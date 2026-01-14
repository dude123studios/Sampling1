"""
Self-Correct Prefix Experiment Runner

Tests if providing prefixes of a model's own correct attempts helps it solve problems more consistently.
This experiment:
1.  Identifies problems where the model got exactly 1-2 correct attempts out of 10.
2.  Extracts the prefix from one of those correct attempts.
3.  Evaluates pass@1 with that prefix.

Usage:
    python scripts/run_self_correct_prefix_experiment.py --config configs/self_correct_prefix_experiment.yaml
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
import random
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.api_model import APIModel
from src.data.prompts import MATH_PROMPT
from src.evaluation.math_grader import grade_math
from dotenv import load_dotenv
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SelfCorrectPrefixExperiment:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        log.info("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
        load_dotenv()

    def get_token_prefix(self, text, num_tokens):
        if num_tokens == 0:
            return ""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        prefix_tokens = tokens[:num_tokens]
        return self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)

    def find_baseline_logs(self):
        """Find appropriate baseline logs or None."""
        if self.config.get('source_results'):
            return self.config['source_results']
            
        # Try to find recent matching sweep results
        sweeps_dir = Path("results/sweeps")
        # Look for baseline sweep or temperature sweep folders
        # This is a heuristic. 
        # Ideally, we should generate them if not specified, which is handled in run_experiment.
        return None

    def process_problem(self, item, prefix_length, model, temperature, max_new_tokens, top_p):
        """Process a single problem with a given prefix."""
        try:
            prefix = self.get_token_prefix(item['correct_solution_text'], prefix_length)
            
            # Use original problem text if available, or reconstruct prompt?
            # The 'item' here comes from our filtered list.
            # Ideally we have the original 'problem' text.
            
            if 'original_item' in item:
                # Reconstruct prompt
                problem_text = item['original_item'].get('problem') or item['original_item'].get('question')
                base_prompt = MATH_PROMPT.format(problem=problem_text)
            elif 'prompt' in item:
                 base_prompt = item['prompt'] # Might contain the prompt
            else:
                 # Fallback: assume the 'prompt' field in log.jsonl is the full prompt
                 # But log.jsonl usually keys: id, outputs, scores, gold, metrics, dataset_id...
                 # It doesn't always have the full prompt text unless logged.
                 # We might need to reload the dataset to get the prompt.
                 # Let's hope the filtered item contains 'problem'
                 return {'error': 'Missing problem text'}

            output_continuation = model.generate(
                base_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                prefix=prefix if prefix_length > 0 else None
            )

            # Reconstruct full output
            if prefix_length > 0:
                output = prefix + output_continuation
            else:
                output = output_continuation

            is_correct = grade_math(output, item['gold'])

            return {
                'id': item['id'],
                'prefix_length': prefix_length,
                'output': output,
                'is_correct': is_correct
            }
        except Exception as e:
            return {'id': item['id'], 'error': str(e)}

    def run_experiment(self, max_workers=40):
        # 1. Load or Generate Baseline Data
        source_file = self.find_baseline_logs()
        
        candidates = []
        
        if source_file and os.path.exists(source_file):
            log.info(f"Loading baseline from {source_file}")
            with open(source_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if 'scores' in entry:
                            correct_count = sum(entry['scores'])
                            min_c = self.config['filter_criteria']['min_correct']
                            max_c = self.config['filter_criteria']['max_correct']
                            
                            if min_c <= correct_count <= max_c:
                                # Found a candidate!
                                correct_indices = [i for i, s in enumerate(entry['scores']) if s > 0]
                                if correct_indices:
                                    idx = correct_indices[0] # Pick first correct solution
                                    candidate = {
                                        'id': entry['id'],
                                        'dataset_id': entry.get('dataset_id'),
                                        'gold': entry['gold'],
                                        'correct_solution_text': entry['outputs'][idx],
                                        'baseline_accuracy': correct_count / len(entry['scores'])
                                    }
                                    candidates.append(candidate)
                    except Exception as e:
                        pass
        else:
            # Fallback: Is there a default generation?
            # For now, let's try to search deeper or ask user.
            # But let's check one more common location if source_file was None
            sweeps_base = Path("results/sweeps")
            if sweeps_base.exists():
                # Find most recent log.jsonl in any subdir
                all_logs = sorted(sweeps_base.glob("**/log.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if all_logs:
                    log.info(f"Auto-detected recent sweep log: {all_logs[0]}")
                    source_file = all_logs[0]
                    # Recursive call or copy-paste logic? Copy-paste for simplicity now.
                    with open(source_file, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                if 'scores' in entry:
                                    correct_count = sum(entry['scores'])
                                    min_c = self.config['filter_criteria']['min_correct']
                                    max_c = self.config['filter_criteria']['max_correct']
                                    if min_c <= correct_count <= max_c:
                                        correct_indices = [i for i, s in enumerate(entry['scores']) if s > 0]
                                        if correct_indices:
                                            idx = correct_indices[0]
                                            candidate = {
                                                'id': entry['id'],
                                                'dataset_id': entry.get('dataset_id'),
                                                'gold': entry['gold'],
                                                'correct_solution_text': entry['outputs'][idx],
                                                'baseline_accuracy': correct_count / len(entry['scores'])
                                            }
                                            candidates.append(candidate)
                            except: pass
            
            if not candidates:
                 log.error("No baseline logs found and no candidates extracted. Please run a baseline sweep first.")
                 return

        log.info(f"Found {len(candidates)} candidate problems matching criteria {self.config['filter_criteria']}")
        
        # Load full task data to get problem text
        from src.data.loader import load_task_data
        task_cfg = DictConfig(self.config['task'])
        full_data = load_task_data(task_cfg)
        
        # Maps
        id_to_item = {}
        for i, item in enumerate(full_data):
            # Prefer unique_id or dataset_id
            # Sweep logs use format "test/{subject}/{diff_idx}.json" or similar.
            # MATH-500 usually has 'level', 'subject', 'problem', 'solution'.
            # We can try to reconstruct a comparable ID if needed, or rely on problem content matching if IDs fail.
            # But here, we will index by the integer index 'i' and string 'i' as fallback,
            # and crucially, check if the loaded item has an 'id' or 'unique_id' field.
            
            # Construct a synthetic ID to match sweep logs if possible
            # The sweep likely used a custom loader or version that injected these IDs.
            # Let's assume the sweep log ID might be an index or a path.
            
            # Index by simple integer index as a strong fallback
            id_to_item[str(i)] = item
            id_to_item[i] = item 
            
            # Also index by exact text content hash/preview if we strictly need to match? 
            # No, that's too heavy.
            # Let's hope the 'id' in candidate (which is an int like 38) matches the index 'i' in the loaded dataset.
            # The sweep log has "id": 38. If that corresponds to the 38th item in the dataset, we are good.
            # MATH-500 test set has 500 items. If loaded in same order (sorted), indices should match.


        valid_candidates = []
        for cand in candidates:
             original = None
             if 'dataset_id' in cand and cand['dataset_id']: 
                 original = id_to_item.get(cand['dataset_id']) or id_to_item.get(str(cand['dataset_id']))
             
             if not original:
                 original = id_to_item.get(cand['id'])
                 
             if original:
                 cand['original_item'] = original
                 valid_candidates.append(cand)
        
        candidates = valid_candidates
        log.info(f"Matched {len(candidates)} candidates with original problem text.")
        
        # Setup Experiment
        prefix_lengths = self.config['prefix_lengths']
        target_models = self.config['target_models']
        temperature = self.config['temperature']
        max_new_tokens = self.config['max_new_tokens']
        top_p = self.config['top_p']
        
        results = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'num_candidates': len(candidates),
            'results_per_model': {}
        }

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_info in target_models:
            model_name = model_info['name']
            model_id = model_info['model_name']
            
            log.info(f"\n{'='*60}")
            log.info(f"Testing model: {model_name}")
            log.info(f"{'='*60}")
            
            model_cfg = DictConfig({
                "type": "api",
                "provider": "openrouter",
                "model_name": model_id,
                "base_url": self.config['api']['base_url'],
                "api_key_env": self.config['api']['api_key_env']
            })
            model = APIModel(model_cfg)
            
            # Checkpoint Logic
            checkpoint_file = output_dir / f"checkpoint_{model_name}.json"
            if checkpoint_file.exists():
                log.info(f"Found checkpoint for {model_name}, loading...")
                with open(checkpoint_file, 'r') as f:
                    model_results = json.load(f)
            else:
                model_results = {
                    'model_id': model_id,
                    'results_per_prefix': {}
                }
            
            for prefix_length in prefix_lengths:
                prefix_str = str(prefix_length)
                
                # Initialize or load existing results for this prefix
                if prefix_str not in model_results['results_per_prefix']:
                    model_results['results_per_prefix'][prefix_str] = {
                        'accuracy': 0.0,
                        'num_correct': 0,
                        'details': []
                    }
                
                # Get already processed IDs
                processed_ids = set()
                current_details = model_results['results_per_prefix'][prefix_str].get('details', [])
                for item in current_details:
                    if 'id' in item:
                        processed_ids.add(item['id'])
                
                log.info(f"\nPrefix length: {prefix_length} tokens")
                log.info(f"Previously processed: {len(processed_ids)} items")
                
                # Filter candidates to process
                candidates_to_process = []
                for cand in candidates:
                    if cand['id'] not in processed_ids:
                        candidates_to_process.append(cand)
                
                if not candidates_to_process:
                    log.info(f"All items already processed for prefix {prefix_length}")
                    continue
                    
                log.info(f"Items remaining to process: {len(candidates_to_process)}")
                
                # Process remaining
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.process_problem,
                            cand, prefix_length, model,
                            temperature, max_new_tokens, top_p
                        ): cand['id']
                        for cand in candidates_to_process
                    }
                    
                    items_completed_since_save = 0
                    save_interval = 10
                    
                    for future in tqdm(as_completed(futures), total=len(candidates_to_process), desc=f"Prefix {prefix_length}"):
                        result = future.result()
                        
                        # Update local state
                        current_details.append(result)
                        processed_ids.add(result['id'])
                        
                        # Update model_results
                        model_results['results_per_prefix'][prefix_str]['details'] = current_details
                        
                        # Periodic Checkpoint
                        items_completed_since_save += 1
                        if items_completed_since_save >= save_interval:
                            try:
                                correct_count = sum(1 for r in current_details if r.get('is_correct', False))
                                accuracy = correct_count / len(candidates) if candidates else 0
                                model_results['results_per_prefix'][prefix_str]['num_correct'] = correct_count
                                model_results['results_per_prefix'][prefix_str]['accuracy'] = accuracy
                                
                                with open(checkpoint_file, 'w') as f:
                                    json.dump(model_results, f, indent=2)
                                items_completed_since_save = 0
                            except Exception as e:
                                log.error(f"Failed to save checkpoint: {e}")
                        
                # Stats
                correct_count = sum(1 for r in current_details if r.get('is_correct', False))
                accuracy = correct_count / len(candidates) if candidates else 0
                
                model_results['results_per_prefix'][prefix_str]['accuracy'] = accuracy
                model_results['results_per_prefix'][prefix_str]['num_correct'] = correct_count
                
                log.info(f"Prefix {prefix_length}: Accuracy = {accuracy:.2%} ({correct_count}/{len(candidates)})")
                
                # Save checkpoint
                with open(checkpoint_file, 'w') as f:
                    json.dump(model_results, f, indent=2)
            
            results['results_per_model'][model_name] = model_results

        # Final Save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = output_dir / f"self_correct_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        log.info(f"Saved final results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/self_correct_prefix_experiment.yaml")
    parser.add_argument("--max-workers", type=int, default=40)
    args = parser.parse_args()
    
    experiment = SelfCorrectPrefixExperiment(args.config)
    experiment.run_experiment(max_workers=args.max_workers)
