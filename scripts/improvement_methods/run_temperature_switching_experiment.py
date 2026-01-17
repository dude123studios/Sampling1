"""
Temperature Switching Experiment Runner

Tests if generating the first N tokens with temperature=0 (deterministic) and then
switching to temperature=0.6 improves model performance on math problems.

Usage:
    python scripts/improvement_methods/run_temperature_switching_experiment.py --config configs/improvement_methods/temperature_switching_config.yaml
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
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.models.api_model import APIModel
from src.data.prompts import MATH_PROMPT
from src.evaluation.math_grader import grade_math
from dotenv import load_dotenv
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TemperatureSwitchingExperiment:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        log.info("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
        load_dotenv()
        
        # Get switch token count from config
        self.switch_token_count = self.config.get('switch_token_count', 16)
        self.initial_temperature = self.config.get('initial_temperature', 0.0)
        self.continuation_temperature = self.config.get('continuation_temperature', 0.6)

    def find_baseline_logs(self):
        """Find appropriate baseline logs or None."""
        if self.config.get('source_results'):
            return self.config['source_results']
            
        # Try to find recent matching sweep results
        sweeps_dir = Path("results/sweeps")
        if sweeps_dir.exists():
            all_logs = sorted(sweeps_dir.glob("**/log.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
            if all_logs:
                log.info(f"Auto-detected recent sweep log: {all_logs[0]}")
                return str(all_logs[0])
        return None

    def process_problem(self, item, model, max_new_tokens, top_p):
        """Process a single problem with temperature switching."""
        try:
            if 'original_item' in item:
                problem_text = item['original_item'].get('problem') or item['original_item'].get('question')
                base_prompt = MATH_PROMPT.format(problem=problem_text)
            elif 'prompt' in item:
                base_prompt = item['prompt']
            else:
                return {'id': item['id'], 'error': 'Missing problem text'}

            # Stage 1: Generate first N tokens with temperature=0
            first_stage_output = model.generate(
                base_prompt,
                temperature=self.initial_temperature,
                max_new_tokens=self.switch_token_count,
                top_p=top_p
            )
            
            # Stage 2: Continue generation with temperature=0.6
            # Combine prompt + first stage output as new prompt
            continuation_prompt = base_prompt + first_stage_output
            continuation_output = model.generate(
                continuation_prompt,
                temperature=self.continuation_temperature,
                max_new_tokens=max_new_tokens - self.switch_token_count,
                top_p=top_p
            )
            
            # Full output is first stage + continuation
            full_output = first_stage_output + continuation_output

            is_correct = grade_math(full_output, item['gold'])

            return {
                'id': item['id'],
                'switch_token_count': self.switch_token_count,
                'first_stage_output': first_stage_output,
                'continuation_output': continuation_output,
                'output': full_output,
                'is_correct': is_correct
            }
        except Exception as e:
            return {'id': item['id'], 'error': str(e)}

    def run_experiment(self, max_workers=40):
        # 1. Load Baseline Data
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
                                    idx = correct_indices[0]  # Pick first correct solution
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
            sweeps_base = Path("results/sweeps")
            if sweeps_base.exists():
                all_logs = sorted(sweeps_base.glob("**/log.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if all_logs:
                    log.info(f"Auto-detected recent sweep log: {all_logs[0]}")
                    source_file = str(all_logs[0])
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
            id_to_item[str(i)] = item
            id_to_item[i] = item

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
        target_models = self.config['target_models']
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
                    'switch_token_count': self.switch_token_count,
                    'details': []
                }
            
            # Get already processed IDs
            processed_ids = set()
            current_details = model_results.get('details', [])
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
                log.info(f"All items already processed for {model_name}")
            else:
                log.info(f"Items remaining to process: {len(candidates_to_process)}")
                
                # Process remaining
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.process_problem,
                            cand, model,
                            max_new_tokens, top_p
                        ): cand['id']
                        for cand in candidates_to_process
                    }
                    
                    items_completed_since_save = 0
                    save_interval = 10
                    
                    for future in tqdm(as_completed(futures), total=len(candidates_to_process), desc=f"Processing {model_name}"):
                        result = future.result()
                        
                        # Update local state
                        current_details.append(result)
                        processed_ids.add(result['id'])
                        
                        # Update model_results
                        model_results['details'] = current_details
                        
                        # Periodic Checkpoint
                        items_completed_since_save += 1
                        if items_completed_since_save >= save_interval:
                            try:
                                correct_count = sum(1 for r in current_details if r.get('is_correct', False))
                                accuracy = correct_count / len(candidates) if candidates else 0
                                model_results['num_correct'] = correct_count
                                model_results['accuracy'] = accuracy
                                
                                with open(checkpoint_file, 'w') as f:
                                    json.dump(model_results, f, indent=2)
                                items_completed_since_save = 0
                            except Exception as e:
                                log.error(f"Failed to save checkpoint: {e}")
            
            # Final stats
            correct_count = sum(1 for r in current_details if r.get('is_correct', False))
            accuracy = correct_count / len(candidates) if candidates else 0
            
            model_results['accuracy'] = accuracy
            model_results['num_correct'] = correct_count
            
            log.info(f"Final Accuracy = {accuracy:.2%} ({correct_count}/{len(candidates)})")
            
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(model_results, f, indent=2)
            
            results['results_per_model'][model_name] = model_results

        # Final Save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = output_dir / f"temperature_switching_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        log.info(f"Saved final results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improvement_methods/temperature_switching_config.yaml")
    parser.add_argument("--max-workers", type=int, default=40)
    args = parser.parse_args()
    
    experiment = TemperatureSwitchingExperiment(args.config)
    experiment.run_experiment(max_workers=args.max_workers)
