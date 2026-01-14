import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.api_model import APIModel
from src.models.hf_model import HFModel
from src.data.loader import load_task_data
from src.data.prompts import get_prompt
from src.sampling.engine import run_sampling
from src.evaluation.math_grader import grade_math
from src.evaluation.gpqa_grader import grade_gpqa
from src.evaluation.code_grader import prepare_code_eval
from src.evaluation.metrics import estimate_pass_at_k
from src.utils.logging import ExperimentLogger

from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

log = logging.getLogger(__name__)

def process_item(i, item, model, cfg, logger):
    prompt = get_prompt(cfg.task.name, item)
    problem_outputs = []
    problem_prompts = []
    problem_scores = []
    
    try:
        # Generate n samples
        for sample_idx in range(cfg.num_samples):
            output, prompt_used = run_sampling(model, prompt, cfg.method)
            problem_outputs.append(output)
            problem_prompts.append(prompt_used)
            
            # Grade immediately
            score = 0
            gold = ""

            if cfg.task.name == "math":
                # MATH-500 uses 'answer' field for gold answer
                gold = item.get('answer', item.get('solution', ''))
                is_correct = grade_math(output, gold)
                score = 1 if is_correct else 0
            elif cfg.task.name == "gpqa":
                gold = item['Correct Answer']
                is_correct = grade_gpqa(output, gold)
                score = 1 if is_correct else 0
            elif cfg.task.name == "code":
                gold = item['solution']
                score = -1 # deferred

            problem_scores.append(score)
        
        # Calculate Pass@k
        metrics = {}
        if cfg.task.name != "code":
            num_correct = sum(problem_scores)
            metrics["num_correct"] = num_correct
            k_list = [k for k in [1, 5, 10, 25, 100] if k <= cfg.num_samples]
            if not k_list: k_list = [1]
            pass_k_scores = estimate_pass_at_k(cfg.num_samples, num_correct, k_list)
            metrics.update(pass_k_scores)
            
            # One@k metric: Did ANY of the samples get it right?
            metrics["one@k"] = 1.0 if num_correct > 0 else 0.0
        
        # Build log entry with metadata
        log_entry = {
            "id": i,
            "original_prompt": prompt,
            "prompts_used": problem_prompts, # List of prompts for each rollout
            "outputs": problem_outputs,
            "scores": problem_scores,
            "gold": gold,
            "metrics": metrics,
            "dataset_id": item.get('unique_id', item.get('problem_id', i))
        }

        # Add MATH-500 specific metadata
        if cfg.task.name == "math":
            log_entry["level"] = item.get('level')
            log_entry["subject"] = item.get('subject')

        logger.log(log_entry)
        return log_entry
        
    except Exception as e:
        log.error(f"Error processing item {i}: {e}", exc_info=True)
        err_entry = {"id": i, "error": str(e)}
        logger.log(err_entry)
        return err_entry

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    load_dotenv()
    log.info(OmegaConf.to_yaml(cfg))
    
    # 1. Setup Logger
    logger = ExperimentLogger(cfg.output_dir)
    
    # 2. Load Model
    log.info("Loading model...")
    if cfg.model.type == "api":
        model = APIModel(cfg.model)
    elif cfg.model.type == "local":
        model = HFModel(cfg.model)
    else:
        raise ValueError("Unknown model type")
        
    # Determine workers
    max_workers = cfg.get("max_workers", 15)
    log.info(f"Running with {max_workers} workers")
        
    # 3. Load Data
    data = load_task_data(cfg.task, cfg.limit, cfg.seed)
    log.info(f"Loaded {len(data)} items")
    
    # 4. Experiment Loop with Multithreading
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, i, item, model, cfg, logger): i for i, item in enumerate(data)}
        
        for future in tqdm(as_completed(futures), total=len(data), desc="Running samples"):
            result = future.result()
            if "metrics" in result:
                all_results.append(result)
    
    # 5. Global Metrics Aggregation
    if all_results and cfg.task.name != "code":
        summary = {"type": "summary"}
        # Calculate averages for each pass@k metric found
        metric_keys = all_results[0]["metrics"].keys()
        for k_key in metric_keys:
            avg_val = sum(r["metrics"][k_key] for r in all_results) / len(all_results)
            summary[f"avg_{k_key}"] = avg_val
            log.info(f"Summary {k_key}: {avg_val:.4f}")

        # Calculate per-level metrics for MATH
        if cfg.task.name == "math":
            per_level_metrics = {}
            for level in range(1, 6):  # Levels 1-5
                level_results = [r for r in all_results if r.get('level') == level]
                if level_results:
                    per_level_metrics[f"level_{level}"] = {}
                    for k_key in metric_keys:
                        avg_val = sum(r["metrics"][k_key] for r in level_results) / len(level_results)
                        per_level_metrics[f"level_{level}"][k_key] = avg_val
                    log.info(f"Level {level} ({len(level_results)} problems): pass@1 = {per_level_metrics[f'level_{level}'].get('pass@1', 0):.4f}")

            summary["per_level_metrics"] = per_level_metrics

        logger.log(summary)
        log.info(f"Experiment complete. Summary logged to {logger.output_dir}")

if __name__ == "__main__":
    main()
