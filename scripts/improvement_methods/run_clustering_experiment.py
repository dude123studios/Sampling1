"""
Clustering Experiment Runner

Tests if clustering layer activations from multiple prefix generations and continuing
from cluster representatives improves model performance on math problems.

For each problem:
1. Generate k different prefixes of length l tokens
2. Extract activations from specified layer at position l for each prefix
3. Cluster these k activations using HDBSCAN
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
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
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

class ClusteringExperiment:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model config
        model_config = self.config['model']
        self.device = model_config.get('device', 'cuda')
        self.dtype = torch.float16 if model_config.get('dtype', 'float16') == 'float16' else torch.float32
        
        # Load model and tokenizer
        log.info(f"Loading model: {model_config['model_id']}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            torch_dtype=self.dtype,
            device_map=self.device
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['model_id'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Experiment config
        self.extraction_layer = self.config.get('extraction_layer', 20)
        self.k_values = self.config.get('k_values', [16, 32, 64])
        self.l_values = self.config.get('l_values', [16, 32, 64])
        self.temperature = self.config.get('temperature', 0.6)
        self.max_new_tokens = self.config.get('max_new_tokens', 4096)
        self.top_p = self.config.get('top_p', 1.0)
        
        # HDBSCAN config
        hdbscan_config = self.config.get('hdbscan', {})
        self.min_cluster_size = hdbscan_config.get('min_cluster_size', 2)
        self.min_samples = hdbscan_config.get('min_samples', 1)
        
        # Initialize API model for continuation generation (OpenRouter)
        api_config = self.config.get('api', {})
        if not api_config:
            raise ValueError("API configuration required for continuation generation")
        
        # Get API key environment variable name from config
        api_key_env = api_config.get('api_key_env', 'OPENROUTER_API_KEY')
        
        # Verify API key is available in environment
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"API Key not found in environment variable: {api_key_env}. "
                f"Please set {api_key_env} environment variable with your OpenRouter API key."
            )
        log.info(f"API key found in environment variable: {api_key_env}")
        
        api_model_cfg = DictConfig({
            'type': 'api',
            'provider': 'openrouter',
            'model_name': api_config.get('model_name', 'qwen/qwen3-8b'),
            'base_url': api_config.get('base_url', 'https://openrouter.ai/api/v1'),
            'api_key_env': api_key_env
        })
        
        log.info(f"Initializing API model for continuation: {api_model_cfg.model_name}")
        log.info(f"Using API base URL: {api_model_cfg.base_url}")
        self.api_model = APIModel(api_model_cfg)
        
        log.info(f"Extraction layer: {self.extraction_layer}")
        log.info(f"k values: {self.k_values}")
        log.info(f"l values: {self.l_values}")

    def get_layer_activation(self, input_ids, position):
        """Get activation from specified layer at given position."""
        activation = None
        
        def hook_fn(module, input, output):
            nonlocal activation
            # Handle both tuple and tensor outputs
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Check dimensions to handle both 2D and 3D tensors
            if len(hidden.shape) == 3:
                # [batch, seq, hidden]
                if position < hidden.shape[1]:
                    activation = hidden[0, position, :].detach().cpu()
            elif len(hidden.shape) == 2:
                # [seq, hidden] - batch dimension squeezed
                if position < hidden.shape[0]:
                    activation = hidden[position, :].detach().cpu()
            else:
                log.warning(f"Unexpected hidden state shape: {hidden.shape}")
        
        # Register hook on specified layer
        layer = self.model.model.layers[self.extraction_layer]
        hook = layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = self.model(input_ids)
        finally:
            hook.remove()
        
        return activation

    def generate_prefix(self, prompt_ids, prefix_length):
        """Generate a prefix of specified length."""
        current_ids = prompt_ids.clone()
        
        for _ in range(prefix_length):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                logits = logits / self.temperature
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return current_ids

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
        """Process a single problem with clustering."""
        try:
            if 'original_item' in item:
                problem_text = item['original_item'].get('problem') or item['original_item'].get('question')
                base_prompt = MATH_PROMPT.format(problem=problem_text)
            elif 'prompt' in item:
                base_prompt = item['prompt']
            else:
                return {'id': item['id'], 'k': k, 'l': l, 'error': 'Missing problem text'}

            # Tokenize prompt
            prompt_ids = self.tokenizer.encode(base_prompt, return_tensors="pt").to(self.model.device)
            prompt_len = prompt_ids.shape[1]
            
            # Step 1: Generate k prefixes of length l and extract activations
            # Store prefix_ids temporarily, but we'll only keep representatives after clustering
            activations = []
            all_prefix_ids = []  # Temporary storage during generation
            
            for i in range(k):
                prefix_ids = self.generate_prefix(prompt_ids, l)
                all_prefix_ids.append(prefix_ids)
                
                # Extract activation at position l (after generating l tokens)
                # prefix_ids has shape [1, prompt_len + l]
                # We want activation at the last position (after l tokens generated)
                extract_position = prefix_ids.shape[1] - 1
                activation = self.get_layer_activation(prefix_ids, extract_position)
                if activation is not None:
                    activations.append(activation)
                else:
                    log.warning(f"Failed to extract activation at position {extract_position} for prefix {i}")
            
            if len(activations) < k:
                return {'id': item['id'], 'k': k, 'l': l, 'error': f'Only got {len(activations)}/{k} activations'}
            
            # Step 2: Cluster activations using HDBSCAN
            activations_stack = torch.stack(activations).numpy()  # [k, hidden_size]
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            )
            cluster_labels = clusterer.fit_predict(activations_stack)
            
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
            
            # Step 4: Keep only representative prefix_ids and decode to text
            # Extract representative prefixes and move to CPU immediately to free GPU memory
            prefix_texts = []
            for rep_idx in cluster_representatives:
                prefix_ids = all_prefix_ids[rep_idx]
                # Move to CPU and extract only the generated tokens (not the prompt)
                # prefix_ids contains [prompt + l generated tokens]
                # We want only the generated l tokens as prefix
                generated_prefix_ids = prefix_ids[0, prompt_len:].cpu().tolist()
                prefix_text = self.tokenizer.decode(generated_prefix_ids, skip_special_tokens=True)
                prefix_texts.append((rep_idx, prefix_text))
            
            # Clear all prefixes and activations from memory (representatives already decoded)
            del all_prefix_ids, activations, activations_stack
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Generate continuations in parallel using multithreading
            def generate_continuation(rep_idx, prefix_text):
                """Generate continuation for a single representative."""
                try:
                    continuation = self.continue_generation_with_api(
                        base_prompt, 
                        prefix_text, 
                        self.max_new_tokens
                    )
                    # Reconstruct full output: prompt + prefix + continuation
                    # The API returns only the continuation (after prefix), so we combine them
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
            all_outputs = []
            all_correct = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(generate_continuation, rep_idx, prefix_text): rep_idx
                    for rep_idx, prefix_text in prefix_texts
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if 'error' not in result:
                        all_outputs.append(result['full_output'])
                        all_correct.append(result['is_correct'])
                    else:
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
                'model_id': self.config['model']['model_id'],
                'extraction_layer': self.extraction_layer,
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
    parser.add_argument("--max-workers", type=int, default=40)
    args = parser.parse_args()
    
    experiment = ClusteringExperiment(args.config)
    experiment.run_experiment(max_workers=args.max_workers)
