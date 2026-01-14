"""
Oracle Prefix Experiment Runner

This experiment tests how model performance improves when given prefixes of oracle/gold
solutions at different token lengths. The hypothesis is that the first few tokens (e.g., 32)
provide the most gain per token in performance.

Usage:
    python scripts/run_oracle_prefix_experiment.py --config configs/oracle_prefix_experiment.yaml --limit 100
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
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.api_model import APIModel
from src.data.prompts import MATH_PROMPT
from src.evaluation.math_grader import grade_math
from dotenv import load_dotenv
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class OraclePrefixExperiment:
    def __init__(self, config_path):
        """Initialize the experiment with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize tokenizer for counting tokens (using a common tokenizer)
        log.info("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            log.warning(f"Could not load gpt2 tokenizer, trying minimal default: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        load_dotenv()

    def get_token_prefix(self, text, num_tokens):
        """Get the first num_tokens tokens from text."""
        if num_tokens == 0:
            return ""

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        prefix_tokens = tokens[:num_tokens]
        prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
        return prefix_text

    def process_oracle_problem(self, oracle_item, prefix_length, model, temperature, max_new_tokens, top_p):
        """Process a single oracle problem with a given prefix length."""
        try:
            # Get prefix from oracle solution
            oracle_solution = oracle_item['oracle_solution']
            prefix = self.get_token_prefix(oracle_solution, prefix_length)

            # Create base prompt (without prefix)
            problem = oracle_item['problem']
            base_prompt = MATH_PROMPT.format(problem=problem)

            # Generate solution
            # If prefix exists, we pass it as a prefill to the API
            # and then prepend it to the output for grading/saving
            
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

            # Grade the solution
            gold_answer = oracle_item['gold_answer']
            is_correct = grade_math(output, gold_answer)

            return {
                'id': oracle_item['id'],
                'dataset_id': oracle_item['dataset_id'],
                'prefix_length': prefix_length,
                'prefix_text': prefix,
                'output': output,
                'is_correct': is_correct
            }

        except Exception as e:
            log.error(f"Error processing problem {oracle_item['id']}: {e}")
            return {
                'id': oracle_item['id'],
                'dataset_id': oracle_item['dataset_id'],
                'prefix_length': prefix_length,
                'error': str(e)
            }

    def run_experiment(self, limit=None, max_workers=10):
        """Run the oracle prefix experiment."""

        # Load oracle solutions
        oracle_file = self.config['oracle_file']
        if not os.path.exists(oracle_file):
            log.error(f"Oracle file not found: {oracle_file}")
            log.error("Please run: python scripts/generate_oracle_solutions.py")
            return

        log.info(f"Loading oracle solutions from {oracle_file}")
        with open(oracle_file, 'r') as f:
            oracle_data = json.load(f)

        # Filter to only correct oracle solutions
        oracle_data = [d for d in oracle_data if d.get('is_correct', False)]
        log.info(f"Using {len(oracle_data)} correct oracle solutions")

        # Apply limit if specified
        if limit:
            oracle_data = oracle_data[:limit]
            log.info(f"Limited to {limit} problems")

        # Get configuration
        prefix_lengths = self.config['prefix_lengths']
        target_models = self.config['target_models']
        temperature = self.config['temperature']
        max_new_tokens = self.config['max_new_tokens']
        top_p = self.config['top_p']

        # Results storage
        results = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'num_problems': len(oracle_data),
            'results_per_model': {}
        }

        # Test each model
        for model_info in target_models:
            model_name = model_info['name']
            model_id = model_info['model_name']

            log.info(f"\n{'='*60}")
            log.info(f"Testing model: {model_name} ({model_id})")
            log.info(f"{'='*60}")

            # Initialize model
            model_cfg = DictConfig({
                "type": "api",
                "provider": "openrouter",
                "model_name": model_id,
                "base_url": self.config['api']['base_url'],
                "api_key_env": self.config['api']['api_key_env']
            })
            model = APIModel(model_cfg)

            model_results = {
                'model_id': model_id,
                'results_per_prefix': {}
            }

            # Test each prefix length
            for prefix_length in prefix_lengths:
                log.info(f"\nTesting prefix length: {prefix_length} tokens")

                prefix_results = []

                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = {
                        executor.submit(
                            self.process_oracle_problem,
                            oracle_item, prefix_length, model,
                            temperature, max_new_tokens, top_p
                        ): oracle_item['id']
                        for oracle_item in oracle_data
                    }

                    # Process completed tasks
                    for future in tqdm(as_completed(futures), total=len(oracle_data), desc=f"Prefix {prefix_length}"):
                        result = future.result()
                        prefix_results.append(result)

                # Sort results by ID for consistent ordering
                prefix_results.sort(key=lambda x: x['id'])

                # Count correct answers
                correct_count = sum(1 for r in prefix_results if r.get('is_correct', False))

                # Calculate accuracy for this prefix length
                accuracy = correct_count / len(oracle_data) if oracle_data else 0

                model_results['results_per_prefix'][prefix_length] = {
                    'accuracy': accuracy,
                    'num_correct': correct_count,
                    'num_total': len(oracle_data),
                    'details': prefix_results
                }

                log.info(f"Prefix {prefix_length} tokens: Accuracy = {accuracy:.2%} ({correct_count}/{len(oracle_data)})")

            results['results_per_model'][model_name] = model_results

        # Save results
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = output_dir / f"oracle_prefix_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        log.info(f"\n{'='*60}")
        log.info(f"Results saved to {output_file}")
        log.info(f"{'='*60}")

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results):
        """Print a summary of the results."""
        print("\n" + "="*80)
        print("ORACLE PREFIX EXPERIMENT SUMMARY")
        print("="*80)

        for model_name, model_data in results['results_per_model'].items():
            print(f"\nModel: {model_name}")
            print("-" * 60)
            print(f"{'Prefix Length':<15} {'Accuracy':<15} {'Correct/Total':<20}")
            print("-" * 60)

            for prefix_length in sorted(model_data['results_per_prefix'].keys()):
                prefix_data = model_data['results_per_prefix'][prefix_length]
                accuracy = prefix_data['accuracy']
                correct = prefix_data['num_correct']
                total = prefix_data['num_total']
                print(f"{prefix_length:<15} {accuracy:>6.2%}{'':<9} {correct}/{total:<15}")

        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Run Oracle Prefix Experiment")
    parser.add_argument("--config", type=str,
                        default="configs/oracle_prefix_experiment.yaml",
                        help="Path to experiment config file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of problems to test")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")

    args = parser.parse_args()

    experiment = OraclePrefixExperiment(args.config)
    experiment.run_experiment(limit=args.limit, max_workers=args.max_workers)

if __name__ == "__main__":
    main()
