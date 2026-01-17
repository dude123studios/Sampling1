"""
HumanEval Code Generation Sweep Runner

Runs HumanEval code generation benchmark with pass@k evaluation.
Follows the scientifically rigorous protocol from Chen et al. (2021).

Usage:
    python scripts/run_humaneval_sweep.py --config configs/sweep/humaneval_sweep.yaml
    python scripts/run_humaneval_sweep.py --config configs/sweep/humaneval_sweep.yaml --dry-run
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
from src.data.prompts import get_prompt, HUMANEVAL_SYSTEM_PROMPT
from src.sampling.engine import run_sampling
from src.evaluators.code_eval import HumanEvalEvaluator, compute_pass_at_k_metrics
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
            runs.append(run_config)

    return runs

def process_item(i, item, model, cfg, evaluator):
    """Process a single HumanEval problem."""
    prompt = get_prompt(cfg['task']['name'], item)
    problem_outputs = []

    try:
        # Generate n samples using batched API calls
        num_samples_to_generate = cfg['parameters']['num_samples']
        batch_size = 5 # Request 5 samples per call to allow for large tokens
        
        while len(problem_outputs) < num_samples_to_generate:
            needed = num_samples_to_generate - len(problem_outputs)
            current_batch_size = min(batch_size, needed)
            
            # Convert method to DictConfig
            method_cfg = DictConfig(cfg['method'])
            method_cfg.temperature = cfg['parameters']['temperature']
            method_cfg.n = current_batch_size # Pass n to API

            # Set top_k and top_p from config if specified
            if 'top_k' in cfg['parameters']:
                method_cfg.top_k = cfg['parameters']['top_k']
            if 'top_p' in cfg['parameters']:
                method_cfg.top_p = cfg['parameters']['top_p']

            # Default values if not present
            if not hasattr(method_cfg, 'top_p'): method_cfg.top_p = 0.9
            if not hasattr(method_cfg, 'top_k'): method_cfg.top_k = 50
            if not hasattr(method_cfg, 'max_new_tokens'): method_cfg.max_new_tokens = 2048

            # Generate code - may return list or string
            output, _ = run_sampling(model, prompt, method_cfg)
            
            if isinstance(output, list):
                problem_outputs.extend(output)
            else:
                problem_outputs.append(output)

        # Evaluate all solutions for this problem
        eval_result = evaluator.evaluate_problem(
            item,
            problem_outputs,
            timeout=5
        )

        # Build result
        result = {
            "id": i,
            "task_id": item['task_id'],
            "outputs": problem_outputs,
            "correctness": eval_result['correctness'],
            "num_samples": eval_result['num_samples'],
            "num_correct": eval_result['num_correct'],
            "dataset_id": item.get('task_id', i)
        }

        return result

    except Exception as e:
        log.error(f"Error processing item {i}: {e}", exc_info=True)
        return {"id": i, "task_id": item.get('task_id', ''), "error": str(e)}

def run_single_experiment(run_config, output_base_dir, max_workers=15):
    """Run a single HumanEval experiment configuration."""
    load_dotenv()

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = run_config['model']['name']
    temp = run_config['parameters']['temperature']
    n_samples = run_config['parameters']['num_samples']

    run_name = f"{model_name}_temp{temp}_n{n_samples}_humaneval_{timestamp}"

    # Support Resuming: Check for existing incomplete runs
    candidate_pattern = f"{model_name}_temp{temp}_n{n_samples}_humaneval_*"
    existing_dirs = sorted(Path(output_base_dir).glob(candidate_pattern))
    existing_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    output_dir = None
    processed_ids = set()

    for d in existing_dirs:
        log_file = d / "log.jsonl"
        if log_file.exists():
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
                return {
                    "run_name": d.name,
                    "output_dir": str(d),
                    "summary": {"status": "skipped_already_done"}
                }

            output_dir = d
            processed_ids = current_processed_ids
            log.info(f"Resuming incomplete run: {d.name} ({len(processed_ids)} items done)")
            break

    if output_dir is None:
        output_dir = Path(output_base_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(str(output_dir))

    # Initialize model
    log.info(f"Initializing model: {model_name}")
    model_cfg = DictConfig(run_config['model'])
    if not hasattr(model_cfg, 'base_url'): model_cfg.base_url = "https://openrouter.ai/api/v1"
    if not hasattr(model_cfg, 'api_key_env'): model_cfg.api_key_env = "OPENROUTER_API_KEY"

    if model_cfg.type == "api":
        model = APIModel(model_cfg)
    elif model_cfg.type == "local":
        model = HFModel(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.type}")

    # Initialize HumanEval evaluator
    evaluator = HumanEvalEvaluator()

    # Load data
    task_cfg = DictConfig(run_config['task'])
    limit = run_config['parameters'].get('limit')
    log.info(f"Loading HumanEval data... (Limit: {limit})")
    data = load_task_data(task_cfg, limit=limit, seed=42)
    log.info(f"Loaded {len(data)} HumanEval problems")

    # Load existing results if resuming
    all_results = []
    if output_dir and (output_dir / "log.jsonl").exists():
        with open(output_dir / "log.jsonl", 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "correctness" in entry and "id" in entry:
                         all_results.append(entry)
                except:
                    pass

    log.info(f"Loaded {len(all_results)} existing results from logs")

    # Run experiment
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(data):
            if i in processed_ids:
                continue
            futures[executor.submit(process_item, i, item, model, run_config, evaluator)] = i

        if not futures:
             log.info("No new items to process.")
        else:
             for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {run_name}"):
                result = future.result()
                if "correctness" in result:
                    all_results.append(result)
                    logger.log(result)

    # Calculate summary with pass@k metrics
    summary = None
    if all_results:
        # Compute pass@k metrics using Chen et al. (2021) estimator
        k_values = [1, 5, 10, 20]
        # Filter to valid k values based on num_samples
        n_samples = run_config['parameters']['num_samples']
        k_values = [k for k in k_values if k <= n_samples]

        pass_at_k_metrics = compute_pass_at_k_metrics(all_results, k_values)

        summary = {
            "type": "summary",
            "run_config": run_config,
            "timestamp": timestamp,
            "total_problems": len(all_results),
            "total_samples": sum(r['num_samples'] for r in all_results),
            "total_correct": sum(r['num_correct'] for r in all_results),
        }

        # Add pass@k metrics
        summary.update(pass_at_k_metrics)

        # Overall accuracy (fraction of correct samples)
        total_samples = summary['total_samples']
        total_correct = summary['total_correct']
        summary['overall_accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0

        logger.log(summary)

        log.info(f"Run complete: {run_name}")
        log.info(f"Total problems: {summary['total_problems']}")
        log.info(f"Total samples: {summary['total_samples']}")
        log.info(f"Total correct: {summary['total_correct']}")
        log.info(f"Overall accuracy: {summary['overall_accuracy']:.4f}")

        for k_key, k_val in pass_at_k_metrics.items():
            log.info(f"{k_key}: {k_val:.4f}")

    return {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "summary": summary
    }

def run_sweep(config_path, output_dir="results/sweeps", dry_run=False, max_workers=15):
    """Run a full HumanEval parameter sweep."""
    log.info(f"Loading sweep config from: {config_path}")
    sweep_config = load_sweep_config(config_path)

    sweep_name = sweep_config.get('name', 'humaneval_sweep')
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
            print(f"{i}. {model_name} | temp={temp} | n={n_samples}")
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
    parser = argparse.ArgumentParser(description="Run HumanEval sweeps")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--output", type=str, default="results/sweeps", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=15, help="Number of parallel workers")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    run_sweep(args.config, output_dir=args.output, dry_run=args.dry_run, max_workers=args.max_workers)

if __name__ == "__main__":
    main()
