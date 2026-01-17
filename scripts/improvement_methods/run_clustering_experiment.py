"""
API-Based Clustering Experiment Runner

Tests if clustering embeddings from multiple prefix generations and continuing
from cluster representatives improves model performance on math problems.

This version runs entirely via OpenRouter API (no local models).

For each problem:
1. Generate k different prefixes of length l tokens via API
2. Embed just the generated text (not prompt) using qwen3-embedding-8b
3. Cluster these k embeddings using HDBSCAN
4. Select one representative from each cluster
5. Continue generation from each representative with standard temperature 0.6
6. Evaluate all continuations

Usage:
    python scripts/improvement_methods/run_clustering_experiment.py --config configs/improvement_methods/clustering/clustering_config.yaml
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hdbscan

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.prompts import MATH_PROMPT
from src.evaluation.math_grader import grade_math
from src.models.api_model import APIModel
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    log.info("Loaded environment variables from .env file")
except ImportError:
    log.warning(".env support not available (python-dotenv not installed)")

class APIClusteringExperiment:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Experiment config
        self.k_values = self.config.get('k_values', [16, 32, 64])
        self.l_values = self.config.get('l_values', [16, 32, 64])
        self.temperature = self.config.get('temperature', 0.6)
        self.max_new_tokens = self.config.get('max_new_tokens', 4096)
        self.top_p = self.config.get('top_p', 1.0)

        # HDBSCAN config
        hdbscan_config = self.config.get('hdbscan', {})
        self.min_cluster_size = hdbscan_config.get('min_cluster_size', 2)
        self.min_samples = hdbscan_config.get('min_samples', 1)

        # API configuration for generation and embedding
        api_config = self.config.get('api', {})
        if not api_config:
            raise ValueError("API configuration required")

        # Get API key from environment variables first (.env file), then config
        api_key_value = os.getenv('OPENROUTER_API_KEY')
        if api_key_value:
            log.info("Using API key from environment variable OPENROUTER_API_KEY")
        elif 'api_key' in api_config and api_config['api_key'] and api_config['api_key'] != 'your-openrouter-api-key-here':
            log.info("Using API key from config file")
            api_key_value = api_config['api_key']
        else:
            raise ValueError(
                "API key not found. Please either:\n"
                "1. Set OPENROUTER_API_KEY in your .env file, or\n"
                "2. Set api_key in the config file, or\n"
                "3. Set the environment variable: export OPENROUTER_API_KEY=your_key_here"
            )

        # Initialize generation model
        gen_model_cfg = DictConfig({
            'type': 'api',
            'provider': 'openrouter',
            'model_name': api_config.get('generation_model', 'qwen/qwen3-8b'),
            'base_url': api_config.get('base_url', 'https://openrouter.ai/api/v1'),
            'api_key': api_key_value
        })

        log.info(f"Initializing generation model: {gen_model_cfg.model_name}")
        self.generation_model = APIModel(gen_model_cfg)

        # Initialize embedding model
        embed_model_cfg = DictConfig({
            'type': 'api',
            'provider': 'openrouter',
            'model_name': api_config.get('embedding_model', 'qwen/qwen3-embedding-8b'),
            'base_url': api_config.get('base_url', 'https://openrouter.ai/api/v1'),
            'api_key': api_key_value
        })

        log.info(f"Initializing embedding model: {embed_model_cfg.model_name}")
        self.embedding_model = APIModel(embed_model_cfg)

        log.info(f"k values: {self.k_values}")
        log.info(f"l values: {self.l_values}")

    def embed_text(self, text, timeout=30):
        """Get embedding for text using the embedding model."""
        try:
            # Use the embedding model's API with timeout
            embedding = self.embedding_model.embed(text, timeout=timeout)
            return embedding
        except Exception as e:
            log.error(f"Error getting embedding: {e}")
            return None

    def generate_prefix_text(self, prompt_text, prefix_length):
        """Generate a prefix text of approximately prefix_length tokens via API."""
        try:
            # Generate with max_tokens to get roughly the right length
            # We'll truncate to approximately prefix_length tokens later
            prefix_completion = self.generation_model.generate(
                prompt_text,
                temperature=self.temperature,
                max_new_tokens=prefix_length * 2,  # Generate extra to account for tokenization differences
                top_p=self.top_p,
                timeout=60,  # 1 minute timeout for prefix generation
                stop_sequences=["\n\n", "###"]  # Stop at natural breaks
            )

            # Extract just the generated part (remove the original prompt)
            if prefix_completion.startswith(prompt_text):
                generated_text = prefix_completion[len(prompt_text):]
            else:
                generated_text = prefix_completion

            # Truncate to approximately prefix_length tokens by word count (rough approximation)
            words = generated_text.split()
            # Estimate ~4 characters per token on average
            target_chars = prefix_length * 4
            if len(generated_text) > target_chars:
                truncated = generated_text[:target_chars]
                # Try to cut at word boundary
                last_space = truncated.rfind(' ')
                if last_space > 0:
                    truncated = truncated[:last_space]
                generated_text = truncated

            return generated_text.strip()

        except Exception as e:
            log.error(f"Error generating prefix: {e}")
            return None

    def continue_generation_with_api(self, base_prompt, prefix_text, max_tokens):
        """Continue generation from prefix using OpenRouter API."""
        continuation = self.api_model.generate(
            base_prompt,
            temperature=self.temperature,
            max_new_tokens=max_tokens,
            top_p=self.top_p,
            prefix=prefix_text  # Pass prefix as prefill
        )
        return continuation

    def process_problem(self, item, k, l):
        """Process a single problem with API-based clustering."""
        try:
            if 'original_item' in item:
                problem_text = item['original_item'].get('problem') or item['original_item'].get('question')
                base_prompt = MATH_PROMPT.format(problem=problem_text)
            elif 'prompt' in item:
                base_prompt = item['prompt']
            else:
                return {'id': item['id'], 'k': k, 'l': l, 'error': 'Missing problem text'}

            log.info(f"Processing problem {item['id']} with k={k}, l={l}")

            # Step 1: Generate k different prefixes and get their embeddings
            prefixes = []
            embeddings = []

            for i in range(k):
                prefix_text = self.generate_prefix_text(base_prompt, l)
                if prefix_text is None:
                    continue

                # Embed just the generated text (not the full prompt)
                embedding = self.embed_text(prefix_text, timeout=30)  # 30 second timeout for embeddings
                if embedding is not None:
                    prefixes.append(prefix_text)
                    embeddings.append(embedding)
                    log.debug(f"Generated and embedded prefix {i}")
                else:
                    log.warning(f"Failed to embed prefix {i}")

            if len(embeddings) < k:
                return {'id': item['id'], 'k': k, 'l': l, 'error': f'Only got {len(embeddings)}/{k} embeddings'}

            # Step 2: Cluster embeddings using HDBSCAN
            embeddings_array = np.array(embeddings)  # [k, embedding_dim]

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            )
            cluster_labels = clusterer.fit_predict(embeddings_array)

            # Step 3: Select one representative from each cluster
            unique_clusters = set(cluster_labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)  # Remove noise cluster

            cluster_representatives = []
            for cluster_id in unique_clusters:
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                # Select first index in cluster as representative
                rep_idx = cluster_indices[0]
                cluster_representatives.append(rep_idx)

            # If there are noise points, also include one of them
            if -1 in cluster_labels:
                noise_indices = np.where(cluster_labels == -1)[0]
                if len(noise_indices) > 0:
                    cluster_representatives.append(noise_indices[0])

            # Step 4: Continue generation from each representative using API
            def generate_continuation(rep_idx):
                """Generate continuation for a single representative."""
                try:
                    prefix_text = prefixes[rep_idx]
                    continuation = self.generation_model.generate(
                        base_prompt,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        top_p=self.top_p,
                        timeout=120,  # 2 minute timeout for continuation generation
                        prefix=prefix_text  # Pass prefix as prefill
                    )
                    # Reconstruct full output: prompt + prefix + continuation
                    full_output = base_prompt + prefix_text + continuation
                    is_correct = grade_math(full_output, item['gold'])
                    return {
                        'rep_idx': rep_idx,
                        'prefix_text': prefix_text,
                        'continuation': continuation,
                        'full_output': full_output,
                        'is_correct': is_correct
                    }
                except Exception as e:
                    log.error(f"Error generating continuation for rep {rep_idx}: {e}")
                    return {
                        'rep_idx': rep_idx,
                        'error': str(e),
                        'is_correct': False
                    }

            # Use ThreadPoolExecutor for parallel API calls
            # Limit to 5 threads per problem to avoid overwhelming the API
            all_outputs = []
            all_correct = []
            with ThreadPoolExecutor(max_workers=min(5, len(cluster_representatives))) as executor:
                futures = [
                    executor.submit(generate_continuation, rep_idx)
                    for rep_idx in cluster_representatives
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=130)  # Slightly longer than API timeout
                        if 'error' not in result:
                            all_outputs.append(result['full_output'])
                            all_correct.append(result['is_correct'])
                        else:
                            all_correct.append(False)
                    except Exception as e:
                        log.error(f"Future result error: {e}")
                        all_correct.append(False)

            # Best result is if any continuation is correct
            best_correct = any(all_correct) if all_correct else False

            return {
                'id': item['id'],
                'k': k,
                'l': l,
                'num_clusters': len(unique_clusters),
                'num_representatives': len(cluster_representatives),
                'outputs': all_outputs,
                'is_correct': all_correct,
                'best_correct': best_correct,
                'num_correct': sum(all_correct)
            }
        except Exception as e:
            log.error(f"Error processing problem {item.get('id', 'unknown')}: {e}")
            return {'id': item['id'], 'k': k, 'l': l, 'error': str(e)}

    def run_experiment(self, max_workers=40):
        # Load task data directly from dataset (filtered by level 5)
        from src.data.loader import load_task_data
        task_cfg = DictConfig(self.config['task'])
        full_data = load_task_data(task_cfg)
        
        log.info(f"Loaded {len(full_data)} Level 5 problems from dataset")
        
        # Convert to candidate format
        candidates = []
        for i, item in enumerate(full_data):
            # Get gold answer (answer field in MATH-500)
            gold = item.get('answer', item.get('solution', ''))
            
            candidate = {
                'id': i,
                'dataset_id': item.get('unique_id', str(i)),
                'gold': gold,
                'original_item': item
            }
            candidates.append(candidate)
        
        log.info(f"Prepared {len(candidates)} Level 5 problems for clustering experiment")
        
        # Setup Experiment
        results = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'num_candidates': len(candidates),
            'results_per_config': {}
        }

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all k, l combinations
        configs_to_test = []
        for k in self.k_values:
            for l in self.l_values:
                configs_to_test.append((k, l))
        
        log.info(f"Testing {len(configs_to_test)} configurations: {configs_to_test}")
        
        checkpoint_file = output_dir / f"checkpoint_clustering.json"
        if checkpoint_file.exists():
            log.info(f"Found checkpoint, loading...")
            with open(checkpoint_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {
                'generation_model': self.generation_model.model_name,
                'embedding_model': self.embedding_model.model_name,
                'results_per_config': {}
            }
        
        for k, l in configs_to_test:
            config_key = f"k{k}_l{l}"
            log.info(f"\n{'='*60}")
            log.info(f"Testing configuration: k={k}, l={l}")
            log.info(f"{'='*60}")
            
            # Initialize or load existing results for this config
            if config_key not in all_results['results_per_config']:
                all_results['results_per_config'][config_key] = {
                    'k': k,
                    'l': l,
                    'details': []
                }
            
            # Get already processed IDs
            processed_ids = set()
            current_details = all_results['results_per_config'][config_key].get('details', [])
            for item in current_details:
                if 'id' in item:
                    processed_ids.add(item['id'])
            
            log.info(f"Previously processed: {len(processed_ids)} items")
            
            # Filter candidates to process
            candidates_to_process = []
            for cand in candidates:
                if cand['id'] not in processed_ids:
                    candidates_to_process.append(cand)
            
            if not candidates_to_process:
                log.info(f"All items already processed for k={k}, l={l}")
            else:
                log.info(f"Items remaining to process: {len(candidates_to_process)}")
                
                # Process remaining
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.process_problem,
                            cand, k, l
                        ): cand['id']
                        for cand in candidates_to_process
                    }
                    
                    items_completed_since_save = 0
                    save_interval = 10
                    
                    for future in tqdm(as_completed(futures), total=len(candidates_to_process), desc=f"k={k}, l={l}"):
                        result = future.result()
                        
                        # Update local state
                        current_details.append(result)
                        processed_ids.add(result['id'])
                        
                        # Update results
                        all_results['results_per_config'][config_key]['details'] = current_details
                        
                        # Periodic Checkpoint
                        items_completed_since_save += 1
                        if items_completed_since_save >= save_interval:
                            try:
                                best_correct_count = sum(1 for r in current_details if r.get('best_correct', False))
                                accuracy = best_correct_count / len(candidates) if candidates else 0
                                all_results['results_per_config'][config_key]['num_correct'] = best_correct_count
                                all_results['results_per_config'][config_key]['accuracy'] = accuracy
                                
                                with open(checkpoint_file, 'w') as f:
                                    json.dump(all_results, f, indent=2)
                                items_completed_since_save = 0
                            except Exception as e:
                                log.error(f"Failed to save checkpoint: {e}")
            
            # Final stats
            best_correct_count = sum(1 for r in current_details if r.get('best_correct', False))
            accuracy = best_correct_count / len(candidates) if candidates else 0
            
            all_results['results_per_config'][config_key]['accuracy'] = accuracy
            all_results['results_per_config'][config_key]['num_correct'] = best_correct_count
            
            log.info(f"Final Accuracy (k={k}, l={l}) = {accuracy:.2%} ({best_correct_count}/{len(candidates)})")
            
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            results['results_per_config'][config_key] = all_results['results_per_config'][config_key]

        # Final Save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = output_dir / f"clustering_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        log.info(f"Saved final results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improvement_methods/clustering/clustering_config.yaml")
    parser.add_argument("--max-workers", type=int, default=20)  # Increased for more parallelism
    args = parser.parse_args()

    experiment = APIClusteringExperiment(args.config)
    experiment.run_experiment(max_workers=args.max_workers)
