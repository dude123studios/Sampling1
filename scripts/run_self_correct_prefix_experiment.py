"""
Self-Correct Prefix Experiment Runner

This experiment finds problems where a model achieves pass@k with exactly 1-2 correct attempts,
then tests whether providing prefixes of the correct attempt helps the model solve more consistently.
The hypothesis is that grounding the first few tokens matters most.

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
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.api_model import APIModel
from src.data.loader import load_task_data
from src.data.prompts import MATH_PROMPT, get_prompt
from src.evaluation.math_grader import grade_math
from src.sampling.engine import run_sampling
from src.utils.logging import ExperimentLogger
from dotenv import load_dotenv
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SelfCorrectPrefixExperiment:
    def __init__(self, config_path):
        """Initialize the experiment with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize tokenizer for counting tokens
        log.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        load_dotenv()

    def get_token_prefix(self, text, num_tokens):
        """Get the first num_tokens tokens from text."""
        if num_tokens == 0:
            return ""

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        prefix_tokens = tokens[:num_tokens]
        prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=True)
        return prefix_text

    def process_self_correct_problem(self, problem, prefix_length, model, temperature, max_new_tokens, top_p, num_samples):
        """Process a single problem with a given prefix length."""
        try:
            # Use the first correct output as the prefix source
            correct_output = problem['correct_outputs'][0]
            prefix = self.get_token_prefix(correct_output, prefix_length)

            # Create prompt with prefix
            problem_text = problem['problem']
            base_prompt = MATH_PROMPT.format(problem=problem_text)

            # Generate multiple samples
            outputs = []
            scores = []

            for _ in range(num_samples):
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
                
                outputs.append(output)

                # Grade
                gold_answer = problem['gold']
                is_correct = grade_math(output, gold_answer)
                scores.append(1 if is_correct else 0)

            return {
                'id': problem['id'],
                'dataset_id': problem['dataset_id'],
                'prefix_length': prefix_length,
                'prefix_text': prefix,
                'outputs': outputs,
                'scores': scores,
                'num_correct': sum(scores)
            }

        except Exception as e:
            log.error(f"Error processing problem {problem['id']}: {e}")
            return {
                'id': problem['id'],
                'dataset_id': problem['dataset_id'],
                'prefix_length': prefix_length,
                'error': str(e)
            }

    def load_or_generate_baseline(self):
        """Load existing results or generate new baseline data."""
        source_results = self.config.get('source_results')

        if source_results and os.path.exists(source_results):
            log.info(f"Loading existing results from {source_results}")
            with open(source_results, 'r') as f:
                results = [json.loads(line) for line in f if line.strip() and 'type' not in json.loads(line)]
            return results

        # Generate baseline if needed
        baseline_cfg = self.config.get('baseline_generation', {})
        if not baseline_cfg.get('enabled', False):
            raise ValueError("No source results found and baseline generation is disabled")

        log.info("Generating baseline results...")
        return self.generate_baseline_data(baseline_cfg)

    def generate_baseline_data(self, baseline_cfg):
        """Generate baseline pass@k data for filtering."""
        log.info("Generating baseline pass@k data...")

        # Setup model
        model_cfg = DictConfig({
            "type": "api",
            "provider": "openrouter",
            "model_name": baseline_cfg['model_name'],
            "base_url": self.config['api']['base_url'],
            "api_key_env": self.config['api']['api_key_env']
        })
        model = APIModel(model_cfg)

        # Load task data
        task_cfg = DictConfig(self.config['task'])
        data = load_task_data(task_cfg, limit=baseline_cfg.get('limit'), seed=42)

        # Baseline method config
        method_cfg = DictConfig({
            "name": "baseline",
            "temperature": baseline_cfg['temperature'],
            "top_p": 1.0
        })

        results = []
        num_samples = baseline_cfg['num_samples']

        for i, item in enumerate(tqdm(data, desc="Generating baseline")):
            try:
                prompt = get_prompt(self.config['task']['name'], item)
                outputs = []
                scores = []

                # Generate multiple samples
                for _ in range(num_samples):
                    output, _ = run_sampling(model, prompt, method_cfg)
                    outputs.append(output)

                    # Grade (MATH-500 uses 'answer' field)
                    gold = item.get('answer', item.get('solution', ''))
                    is_correct = grade_math(output, gold)
                    scores.append(1 if is_correct else 0)

                # Store result
                result = {
                    'id': i,
                    'dataset_id': item.get('unique_id', item.get('problem_id', i)),
                    'original_prompt': prompt,
                    'problem': item['problem'],
                    'outputs': outputs,
                    'scores': scores,
                    'gold': gold,
                    'num_correct': sum(scores),
                    'level': item.get('level'),
                    'subject': item.get('subject')
                }
                results.append(result)

            except Exception as e:
                log.error(f"Error generating baseline for problem {i}: {e}")

        # Save baseline results
        output_dir = Path(self.config['output_dir']) / "baseline_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        baseline_file = output_dir / f"baseline_{timestamp}.json"

        with open(baseline_file, 'w') as f:
            json.dump(results, f, indent=2)

        log.info(f"Saved baseline results to {baseline_file}")
        return results

    def filter_problems(self, baseline_results):
        """Filter problems based on criteria (e.g., exactly 1-2 correct)."""
        min_correct = self.config['filter_criteria']['min_correct']
        max_correct = self.config['filter_criteria']['max_correct']

        filtered = []
        for result in baseline_results:
            num_correct = result.get('num_correct', sum(result.get('scores', [])))
            if min_correct <= num_correct <= max_correct:
                # Find the correct solution(s)
                correct_outputs = [
                    output for output, score in zip(result['outputs'], result['scores'])
                    if score == 1
                ]
                if correct_outputs:
                    result['correct_outputs'] = correct_outputs
                    filtered.append(result)

        log.info(f"Filtered {len(filtered)} problems (from {len(baseline_results)} total)")
        log.info(f"Criteria: {min_correct} <= num_correct <= {max_correct}")

        return filtered

    def run_experiment(self, max_workers=10):
        """Run the self-correct prefix experiment."""

        # Load or generate baseline data
        baseline_results = self.load_or_generate_baseline()

        # Filter problems
        filtered_problems = self.filter_problems(baseline_results)

        if not filtered_problems:
            log.error("No problems match the filter criteria!")
            return

        # Get configuration
        prefix_lengths = self.config['prefix_lengths']
        target_models = self.config['target_models']
        temperature = self.config['temperature']
        max_new_tokens = self.config['max_new_tokens']
        top_p = self.config['top_p']
        num_samples = self.config['num_samples']

        # Results storage
        results = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'num_problems': len(filtered_problems),
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
                            self.process_self_correct_problem,
                            problem, prefix_length, model,
                            temperature, max_new_tokens, top_p, num_samples
                        ): problem['id']
                        for problem in filtered_problems
                    }

                    # Process completed tasks
                    for future in tqdm(as_completed(futures), total=len(filtered_problems), desc=f"Prefix {prefix_length}"):
                        result = future.result()
                        prefix_results.append(result)

                # Sort results by ID for consistent ordering
                prefix_results.sort(key=lambda x: x['id'])

                # Calculate totals
                total_correct = sum(result.get('num_correct', 0) for result in prefix_results)
                total_attempts = sum(len(result.get('scores', [])) for result in prefix_results)

                # Calculate accuracy for this prefix length
                accuracy = total_correct / total_attempts if total_attempts > 0 else 0

                model_results['results_per_prefix'][prefix_length] = {
                    'accuracy': accuracy,
                    'total_correct': total_correct,
                    'total_attempts': total_attempts,
                    'num_problems': len(filtered_problems),
                    'details': prefix_results
                }

                log.info(f"Prefix {prefix_length} tokens: Accuracy = {accuracy:.2%} ({total_correct}/{total_attempts})")

            results['results_per_model'][model_name] = model_results

        # Save results
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = output_dir / f"self_correct_prefix_results_{timestamp}.json"

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
        print("SELF-CORRECT PREFIX EXPERIMENT SUMMARY")
        print("="*80)

        for model_name, model_data in results['results_per_model'].items():
            print(f"\nModel: {model_name}")
            print("-" * 60)
            print(f"{'Prefix Length':<15} {'Accuracy':<15} {'Correct/Total':<20}")
            print("-" * 60)

            for prefix_length in sorted(model_data['results_per_prefix'].keys()):
                prefix_data = model_data['results_per_prefix'][prefix_length]
                accuracy = prefix_data['accuracy']
                correct = prefix_data['total_correct']
                total = prefix_data['total_attempts']
                print(f"{prefix_length:<15} {accuracy:>6.2%}{'':<9} {correct}/{total:<15}")

        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Run Self-Correct Prefix Experiment")
    parser.add_argument("--config", type=str,
                        default="configs/self_correct_prefix_experiment.yaml",
                        help="Path to experiment config file")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")

    args = parser.parse_args()

    experiment = SelfCorrectPrefixExperiment(args.config)
    experiment.run_experiment(max_workers=args.max_workers)

if __name__ == "__main__":
    main()
