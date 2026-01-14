"""
Generate oracle/gold solutions using a strong model (deepseek-r1-llama-70b) via OpenRouter.
These solutions will be used as prefixes in the oracle prefix experiment.

Usage:
    python scripts/generate_oracle_solutions.py --num_problems 500 --output data/oracle_solutions.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.api_model import APIModel
from src.data.loader import load_task_data
from src.data.prompts import get_prompt
from src.evaluation.math_grader import grade_math
from dotenv import load_dotenv
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def process_oracle_item(i, item, model, task_name):
    """Process a single problem to generate oracle solution."""
    try:
        prompt = get_prompt(task_name, item)

        # Generate solution with low temperature for consistency
        solution = model.generate(prompt, temperature=0.6, max_new_tokens=4096)

        # Grade the solution (MATH-500 uses 'answer' field)
        gold = item.get('answer', item.get('solution', ''))
        is_correct = grade_math(solution, gold)

        oracle_entry = {
            "id": i,
            "dataset_id": item.get('unique_id', item.get('problem_id', i)),
            "problem": item['problem'],
            "oracle_solution": solution,
            "gold_answer": gold,
            "is_correct": is_correct,
            "problem_level": item.get('level', 5),
            "subject": item.get('subject', 'Unknown')
        }

        return oracle_entry

    except Exception as e:
        log.error(f"Error processing problem {i}: {e}")
        return {
            "id": i,
            "dataset_id": item.get('unique_id', item.get('problem_id', i)),
            "problem": item['problem'],
            "error": str(e)
        }

def generate_oracle_solutions(num_problems=500, output_path="data/oracle_solutions.json", task_name="math", max_workers=10):
    """Generate oracle solutions using a strong model."""

    load_dotenv()

    # Create model config for deepseek-r1-llama-70b
    model_cfg = DictConfig({
        "type": "api",
        "provider": "openrouter",
        "model_name": "deepseek/deepseek-r1-distill-llama-70b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY"
    })

    # Create task config
    task_cfg = DictConfig({
        "name": task_name,
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "level_filter": None  # Use all levels
    })

    log.info(f"Loading model: {model_cfg.model_name}")
    model = APIModel(model_cfg)

    log.info(f"Loading {num_problems} problems from {task_cfg.dataset}")
    data = load_task_data(task_cfg, limit=num_problems, seed=42)

    log.info(f"Generating oracle solutions with {max_workers} workers...")
    oracle_solutions = []
    save_lock = threading.Lock()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_oracle_item, i, item, model, task_name): i
                   for i, item in enumerate(data)}

        # Process completed tasks
        for future in tqdm(as_completed(futures), total=len(data), desc="Generating solutions"):
            result = future.result()

            # Thread-safe append
            with save_lock:
                oracle_solutions.append(result)

                # Save incrementally every 50 problems
                if len(oracle_solutions) % 50 == 0:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(sorted(oracle_solutions, key=lambda x: x['id']), f, indent=2)
                    log.info(f"Saved {len(oracle_solutions)} solutions to {output_path}")

    # Sort by ID for consistent ordering
    oracle_solutions.sort(key=lambda x: x['id'])

    # Final save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(oracle_solutions, f, indent=2)

    # Calculate statistics
    num_correct = sum(1 for s in oracle_solutions if s.get('is_correct', False))
    accuracy = num_correct / len(oracle_solutions) if oracle_solutions else 0

    log.info(f"Generated {len(oracle_solutions)} oracle solutions")
    log.info(f"Oracle model accuracy: {accuracy:.2%} ({num_correct}/{len(oracle_solutions)})")
    log.info(f"Saved to {output_path}")

    return oracle_solutions

def main():
    parser = argparse.ArgumentParser(description="Generate oracle solutions for prefix experiments")
    parser.add_argument("--num_problems", type=int, default=500,
                        help="Number of problems to generate solutions for")
    parser.add_argument("--output", type=str, default="data/oracle_solutions.json",
                        help="Output path for oracle solutions")
    parser.add_argument("--task", type=str, default="math",
                        help="Task name (currently only 'math' is supported)")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")

    args = parser.parse_args()

    generate_oracle_solutions(
        num_problems=args.num_problems,
        output_path=args.output,
        task_name=args.task,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()
