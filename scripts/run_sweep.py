"""
Sweep Runner

Runs systematic parameter sweeps based on YAML configurations.

Usage:
    python scripts/run_sweep.py --config configs/sweep/baseline_sweep.yaml
    python scripts/run_sweep.py --config configs/sweep/temperature_sweep.yaml --dry-run
"""

import argparse
import sys
import os
from pathlib import Path
import json
import itertools
from datetime import datetime
from tqdm import tqdm
import logging
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.api_model import APIModel
from src.models.hf_model import HFModel
from src.data.loader import load_task_data
from src.data.prompts import get_prompt
from src.sampling.engine import run_sampling
from src.evaluation.math_grader import grade_math
from src.evaluation.metrics import estimate_pass_at_k
from src.utils.logging import ExperimentLogger
from dotenv import load_dotenv
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_sweep_config(config_path):
    """Load sweep configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_sweep_runs(sweep_config):
    """Generate all run configurations from sweep parameters."""
    models = sweep_config['models']
    parameters = sweep_config['parameters']

    # Get all parameter combinations
    param_names = list(parameters.keys())
    param_values = [parameters[name] for name in param_names]

    runs = []
    for model in models:
        for param_combo in itertools.product(*param_values):
            run_config = {
                'model': model,
                'task': sweep_config['task'].copy(),
                'method': sweep_config['method'].copy(),
                'parameters': dict(zip(param_names, param_combo))
            }

            # Override task level_filter if specified in parameters
            if 'level_filter' in run_config['parameters']:
                run_config['task']['level_filter'] = run_config['parameters']['level_filter']
                
            # Override limit if specified in parameters
            if 'limit' in run_config['parameters']:
                # Note: limit might be None, which is fine
                pass

            runs.append(run_config)

    return runs

def process_item(i, item, model, cfg):
    """Process a single problem."""
    prompt = get_prompt(cfg['task']['name'], item)
    problem_outputs = []
    problem_scores = []

    try:
        # Generate n samples
        for sample_idx in range(cfg['parameters']['num_samples']):
            # Convert method to DictConfig
            method_cfg = DictConfig(cfg['method'])
            method_cfg.temperature = cfg['parameters']['temperature']
            # Default defaults if not present
            if not hasattr(method_cfg, 'top_p'): method_cfg.top_p = 1.0
            if not hasattr(method_cfg, 'max_new_tokens'): method_cfg.max_new_tokens = 4096

            output, _ = run_sampling(model, prompt, method_cfg)
            problem_outputs.append(output)

            # Grade
            gold = item.get('answer', item.get('solution', ''))
            is_correct = grade_math(output, gold)
            score = 1 if is_correct else 0
            problem_scores.append(score)

        # Calculate metrics
        num_correct = sum(problem_scores)
        metrics = {"num_correct": num_correct}

        k_list = [k for k in [1, 5, 10, 25, 50, 100] if k <= cfg['parameters']['num_samples']]
        if not k_list:
            k_list = [1]
        pass_k_scores = estimate_pass_at_k(cfg['parameters']['num_samples'], num_correct, k_list)
        metrics.update(pass_k_scores)
        metrics["one@k"] = 1.0 if num_correct > 0 else 0.0

        # Build result
        result = {
            "id": i,
            "outputs": problem_outputs,
            "scores": problem_scores,
            "gold": gold,
            "metrics": metrics,
            "dataset_id": item.get('unique_id', item.get('problem_id', i)),
            "level": item.get('level'),
            "subject": item.get('subject')
        }

        return result

    except Exception as e:
        log.error(f"Error processing item {i}: {e}", exc_info=True)
        return {"id": i, "error": str(e)}

def run_single_experiment(run_config, output_base_dir, max_workers=15):
    """Run a single experiment configuration."""
    load_dotenv()

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = run_config['model']['name']
    temp = run_config['parameters']['temperature']
    n_samples = run_config['parameters']['num_samples']
    level_filter = run_config['parameters'].get('level_filter')
    
    # Safely handle None for level
    if level_filter is None:
        level_str = "_all_levels"
    else:
        level_str = f"_level{level_filter}"
        
    run_name = f"{model_name}_temp{temp}_n{n_samples}{level_str}_{timestamp}"
    
    # Support Resuming: Check for existing incomplete runs with same configuration
    # We ignore the timestamp part of the folder name for matching
    candidate_pattern = f"{model_name}_temp{temp}_n{n_samples}{level_str}_*"
    existing_dirs = sorted(Path(output_base_dir).glob(candidate_pattern))
    
    # Sort by modification time (most recent first) to resume the latest one
    existing_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    output_dir = None
    processed_ids = set()
    
    for d in existing_dirs:
        log_file = d / "log.jsonl"
        if log_file.exists():
            # Check if run is fully complete (has summary)
            has_summary = False
            current_processed_ids = set()
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("type") == "summary":
                                has_summary = True
                                break
                            if "id" in entry:
                                current_processed_ids.add(entry["id"])
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
                
            if has_summary:
                log.info(f"Skipping already completed run: {d.name}")
                # If we found a completed one, we probably shouldn't resume a DIFFERENT incomplete one 
                # for the exact same config unless we really mean to. 
                # But strict resuming policy: if a complete one exists, we are done.
                return {
                    "run_name": d.name,
                    "output_dir": str(d),
                    "summary": {"status": "skipped_already_done"} 
                }
            
            # Found an incomplete run - process it
            output_dir = d
            processed_ids = current_processed_ids
            log.info(f"Resuming incomplete run: {d.name} ({len(processed_ids)} items done)")
            break
    
    if output_dir is None:
        # Create new run
        output_dir = Path(output_base_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)


    logger = ExperimentLogger(str(output_dir))

    # Initialize model
    log.info(f"Initializing model: {model_name}")
    model_cfg = DictConfig(run_config['model'])
    # Ensure URL/Key are set if not present in model config (generic overrides)
    if not hasattr(model_cfg, 'base_url'): model_cfg.base_url = "https://openrouter.ai/api/v1"
    if not hasattr(model_cfg, 'api_key_env'): model_cfg.api_key_env = "OPENROUTER_API_KEY"

    if model_cfg.type == "api":
        model = APIModel(model_cfg)
    elif model_cfg.type == "local":
        model = HFModel(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.type}")

    # Load data
    task_cfg = DictConfig(run_config['task'])
    limit = run_config['parameters'].get('limit')
    log.info(f"Loading task data... (Limit: {limit})")
    data = load_task_data(task_cfg, limit=limit, seed=42)
    log.info(f"Loaded {len(data)} items")

    # Filter out already processed items
    if processed_ids:
        original_len = len(data)
        # Assuming data is a list of items where 'id' corresponds to index 'i' in enumerate(data)
        # In process_item, we passed 'i' as the ID in the result dictionary.
        # Wait, process_item uses: "id": i
        # So processed_ids are INDICES into the original data list.
        # We need to filter based on INDEX.
        
        # We can't easily remove items from the list because indices would shift?
        # No, we're using Enumerate(data).
        # Better: keep data as is, but in ThreadPoolExecutor submission, skip processed indices.
        pass
        
    # Run experiment
    all_results = []
    
    # Pre-populate all_results with processed ones?
    # No, we don't need to reload them unless we want to recalc summary.
    # To recalc summary correctly at the end, we SHOULD reload them or just append new ones?
    # Actually, current script RE-CALCULATES summary from all_results.
    # If we don't load previous results into all_results, the Final Summary will only be for the NEW items!
    # This is bad.
    
    # We should restart with all_results containing previous results if resuming.
    # We already scanned file to get IDs. We should scan again or reuse logic to populate all_results.
    
    # Let's refactor the scan loop above slightly to populate all_results too.
    # But I can't easily jump back in this edit.
    # I'll modify the logic below to populate all_results from log file if reusing dir.
    
    if output_dir and (output_dir / "log.jsonl").exists():
        with open(output_dir / "log.jsonl", 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "metrics" in entry and "id" in entry:
                         all_results.append(entry)
                except:
                    pass
    
    log.info(f"Loaded {len(all_results)} existing results from logs")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit valid tasks (skipping processed ones)
        futures = {}
        for i, item in enumerate(data):
            if i in processed_ids:
                continue
            futures[executor.submit(process_item, i, item, model, run_config)] = i

        if not futures:
             log.info("No new items to process.")
        else:
             for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {run_name}"):
                result = future.result()
                if "metrics" in result:
                    all_results.append(result)
                    logger.log(result)

    # Calculate summary
    summary = None
    if all_results:
        # 1. Global Averages
        summary = {
            "type": "summary", 
            "run_config": run_config,
            "timestamp": timestamp,
            "total_items": len(all_results)
        }
        metric_keys = all_results[0]["metrics"].keys()

        for k_key in metric_keys:
            avg_val = sum(r["metrics"][k_key] for r in all_results) / len(all_results)
            summary[f"avg_{k_key}"] = avg_val

        # 2. Per-level Breakdown
        per_level_metrics = {}
        # Levels are typically 1-5 for MATH
        levels_found = set(r.get('level') for r in all_results if r.get('level') is not None)
        
        for level in sorted(list(levels_found)):
            level_results = [r for r in all_results if r.get('level') == level]
            if level_results:
                level_key = f"level_{level}"
                per_level_metrics[level_key] = {
                    "count": len(level_results)
                }
                for k_key in metric_keys:
                    avg_val = sum(r["metrics"][k_key] for r in level_results) / len(level_results)
                    per_level_metrics[level_key][k_key] = avg_val
        
        summary["per_level_metrics"] = per_level_metrics
        
        # 3. Save Summary
        logger.log(summary)

        log.info(f"Run complete: {run_name}")
        log.info(f"Average pass@1: {summary.get('avg_pass@1', 0):.4f}")
        
        # Also log per-level pass@1 for immediate visibility
        if per_level_metrics:
            log.info("Per-level Pass@1:")
            for lvl, mets in per_level_metrics.items():
                if "pass@1" in mets:
                    log.info(f"  {lvl}: {mets['pass@1']:.4f} (n={mets['count']})")

    return {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "summary": summary
    }

def run_sweep(config_path, output_dir="results/sweeps", dry_run=False, max_workers=15):
    """Run a full parameter sweep."""
    log.info(f"Loading sweep config from: {config_path}")
    sweep_config = load_sweep_config(config_path)
    
    sweep_name = sweep_config.get('name', 'unnamed_sweep')
    log.info(f"Running sweep: {sweep_name}")
    log.info(f"Description: {sweep_config.get('description', 'No description')}")

    # Generate all run configurations
    runs = generate_sweep_runs(sweep_config)
    log.info(f"Generated {len(runs)} run configurations")

    if dry_run:
        log.info("\nDRY RUN - Would execute the following runs:")
        for i, run in enumerate(runs, 1):
            model_name = run['model']['name']
            temp = run['parameters']['temperature']
            n_samples = run['parameters']['num_samples']
            level_filter = run['parameters'].get('level_filter')
            print(f"{i}. {model_name} | temp={temp} | n={n_samples} | level={level_filter}")
        return

    # Create sweep output directory
    sweep_output_dir = Path(output_dir) / sweep_name
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of config used
    with open(sweep_output_dir / "sweep_config.yaml", 'w') as f:
        yaml.dump(sweep_config, f)

    # Run all experiments
    sweep_results = []
    for i, run_config in enumerate(runs, 1):
        log.info(f"\n{'='*80}")
        log.info(f"Running experiment {i}/{len(runs)}")
        log.info(f"{'='*80}")

        try:
            result = run_single_experiment(run_config, sweep_output_dir, max_workers)
            sweep_results.append(result)
        except Exception as e:
            log.error(f"Error running experiment {i}: {e}", exc_info=True)
            sweep_results.append({
                "run_config": run_config,
                "error": str(e)
            })

    # Save sweep summary
    sweep_summary_file = sweep_output_dir / "sweep_summary.json"
    
    # We need to serialize just the summary dicts, handling non-serializable objects if any
    # Since 'run_config' might contain basic types, should be fine.
    
    with open(sweep_summary_file, 'w') as f:
        json.dump({
            "sweep_name": sweep_name,
            "description": sweep_config.get('description'),
            "total_runs": len(runs),
            "results": sweep_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    log.info(f"\n{'='*80}")
    log.info(f"Sweep complete!")
    log.info(f"Results saved to: {sweep_output_dir}")
    log.info(f"Summary: {sweep_summary_file}")
    log.info(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Run parameter sweeps")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--output", type=str, default="results/sweeps", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=45, help="Number of parallel workers")

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    run_sweep(args.config, output_dir=args.output, dry_run=args.dry_run, max_workers=args.max_workers)

if __name__ == "__main__":
    main()
