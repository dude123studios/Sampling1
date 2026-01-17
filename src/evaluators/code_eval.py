"""
HumanEval Code Generation Evaluation with pass@k metrics.

Implements the scientifically rigorous evaluation protocol from:
Chen et al. (2021) "Evaluating Large Language Models Trained on Code"

Uses the unbiased pass@k estimator to compute pass rates.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import tempfile
import subprocess
import signal
from contextlib import contextmanager
import multiprocessing as mp
from scipy.special import comb
import logging

log = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for timing out code execution."""
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")

    # Set the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def execute_code_safely(code: str, test_code: str, timeout: int = 5) -> bool:
    """
    Execute generated code with test cases in a sandboxed environment.

    Args:
        code: The generated code to test
        test_code: The test cases to run
        timeout: Maximum execution time in seconds

    Returns:
        True if all tests pass, False otherwise
    """
    import sys

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write the complete test program
        full_program = code + "\n\n" + test_code
        f.write(full_program)
        temp_file = f.name

    try:
        # Run in subprocess with timeout
        # Use sys.executable to ensure we use the same Python interpreter
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )

        # Check if execution succeeded (return code 0 means all assertions passed)
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        log.debug(f"Code execution timed out after {timeout}s")
        return False
    except Exception as e:
        log.debug(f"Code execution failed: {e}")
        return False
    finally:
        # Clean up
        try:
            Path(temp_file).unlink()
        except:
            pass


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k using the unbiased estimator from Chen et al. (2021).

    Formula: pass@k = E[1 - comb(n-c, k) / comb(n, k)]

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of attempts to consider

    Returns:
        Estimated pass@k rate (between 0 and 1)
    """
    if n < k:
        # Cannot compute pass@k if we don't have enough samples
        return 0.0

    if c == 0:
        return 0.0

    # Check if we can't choose k incorrect samples (not enough incorrect samples)
    # In this case, pass@k = 1.0 (guaranteed to get at least one correct)
    if n - c < k:
        return 1.0

    # Unbiased estimator
    numerator = comb(n - c, k, exact=True)
    denominator = comb(n, k, exact=True)

    if denominator == 0:
        return 0.0

    return 1.0 - (numerator / denominator)


def compute_pass_at_k_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute pass@k metrics across all problems.

    Args:
        results: List of evaluation results, each with:
            - problem_id: str
            - correctness: List[bool] for each generated sample
        k_values: List of k values to compute pass@k for

    Returns:
        Dict mapping 'pass@k' to the average pass@k rate
    """
    metrics = {}

    for k in k_values:
        pass_at_k_values = []

        for result in results:
            correctness = result['correctness']
            n = len(correctness)
            c = sum(correctness)

            # Compute pass@k for this problem
            pass_k = estimate_pass_at_k(n, c, k)
            pass_at_k_values.append(pass_k)

        # Average across all problems
        avg_pass_k = np.mean(pass_at_k_values)
        metrics[f'pass@{k}'] = avg_pass_k

    return metrics


class HumanEvalEvaluator:
    """
    Evaluator for HumanEval code generation benchmark.

    Follows the protocol from Chen et al. (2021) and uses the prompt
    format from Wei et al. (2024).
    """

    SYSTEM_PROMPT = "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."

    USER_PROMPT_TEMPLATE = "@@ Instruction\n{instruction}"

    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            dataset_path: Path to HumanEval dataset (if None, will load from HuggingFace)
        """
        self.dataset_path = dataset_path
        self.problems = self.load_dataset()

    def load_dataset(self) -> List[Dict]:
        """
        Load HumanEval dataset.

        Returns:
            List of problems, each with:
                - task_id: str
                - prompt: str (function signature + docstring)
                - canonical_solution: str
                - test: str (test cases)
                - entry_point: str (function name to test)
        """
        if self.dataset_path and Path(self.dataset_path).exists():
            # Load from local file
            with open(self.dataset_path) as f:
                data = [json.loads(line) for line in f if line.strip()]
            log.info(f"Loaded {len(data)} problems from {self.dataset_path}")
            return data
        else:
            # Load from HuggingFace
            try:
                from datasets import load_dataset
                dataset = load_dataset("openai_humaneval", split="test")
                problems = [item for item in dataset]
                log.info(f"Loaded {len(problems)} problems from HuggingFace")
                return problems
            except Exception as e:
                log.error(f"Failed to load HumanEval dataset: {e}")
                raise

    def format_prompt(self, problem: Dict) -> Tuple[str, str]:
        """
        Format the problem into the prompt format from Wei et al. (2024).

        Args:
            problem: Problem dict from HumanEval

        Returns:
            (system_prompt, user_prompt) tuple
        """
        # The instruction is the function signature + docstring
        instruction = problem['prompt']

        user_prompt = self.USER_PROMPT_TEMPLATE.format(instruction=instruction)

        return self.SYSTEM_PROMPT, user_prompt

    def check_solution(
        self,
        problem: Dict,
        generated_code: str,
        timeout: int = 5
    ) -> bool:
        """
        Check if a generated solution is correct.

        Args:
            problem: Problem dict from HumanEval
            generated_code: The model's generated code
            timeout: Maximum execution time in seconds

        Returns:
            True if solution passes all tests, False otherwise
        """
        try:
            # The generated code should complete the function
            # Extract code from markdown blocks if present
            completion = generated_code

            # Strip markdown code blocks (```python ... ```)
            if "```python" in completion:
                start_idx = completion.find("```python") + len("```python")
                completion = completion[start_idx:]

            # Cut off at closing backticks
            if "```" in completion:
                end_idx = completion.find("```")
                completion = completion[:end_idx]

            # Combine prompt (signature) with generated code
            prompt = problem['prompt']

            # Check if completion already includes the prompt/signature
            if completion.strip().startswith(prompt.strip()):
                # Completion already has full function, use as-is
                full_code = completion
            else:
                # Completion is just the body, prepend prompt
                full_code = prompt + completion

            # Get test code
            test_code = problem['test']

            # Add check function call
            check_program = test_code + f"\ncheck({problem['entry_point']})"

            # Execute and check
            return execute_code_safely(full_code, check_program, timeout)

        except Exception as e:
            log.debug(f"Error checking solution: {e}")
            return False

    def evaluate_problem(
        self,
        problem: Dict,
        generated_solutions: List[str],
        timeout: int = 5
    ) -> Dict:
        """
        Evaluate all generated solutions for a problem.

        Args:
            problem: Problem dict from HumanEval
            generated_solutions: List of generated code solutions
            timeout: Maximum execution time per solution

        Returns:
            Dict with:
                - task_id: str
                - correctness: List[bool]
                - num_samples: int
                - num_correct: int
        """
        correctness = []

        for solution in generated_solutions:
            is_correct = self.check_solution(problem, solution, timeout)
            correctness.append(is_correct)

        return {
            'task_id': problem['task_id'],
            'correctness': correctness,
            'num_samples': len(generated_solutions),
            'num_correct': sum(correctness)
        }

    def evaluate_all(
        self,
        results: List[Dict],
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict:
        """
        Evaluate all problems and compute pass@k metrics.

        Args:
            results: List of dicts, each with:
                - task_id: str
                - generated_solutions: List[str]
            k_values: List of k values to compute pass@k for

        Returns:
            Dict with:
                - problem_results: List of per-problem results
                - pass_at_k: Dict of pass@k metrics
                - summary: Overall statistics
        """
        problem_results = []

        # Create lookup for problems
        problems_dict = {p['task_id']: p for p in self.problems}

        # Evaluate each problem
        for result in results:
            task_id = result['task_id']
            generated_solutions = result['generated_solutions']

            if task_id not in problems_dict:
                log.warning(f"Problem {task_id} not found in dataset")
                continue

            problem = problems_dict[task_id]

            # Evaluate all solutions
            eval_result = self.evaluate_problem(
                problem,
                generated_solutions
            )

            problem_results.append(eval_result)

        # Compute pass@k metrics
        pass_at_k_metrics = compute_pass_at_k_metrics(problem_results, k_values)

        # Summary statistics
        total_samples = sum(r['num_samples'] for r in problem_results)
        total_correct = sum(r['num_correct'] for r in problem_results)

        summary = {
            'num_problems': len(problem_results),
            'total_samples': total_samples,
            'total_correct': total_correct,
            'overall_accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }

        return {
            'problem_results': problem_results,
            'pass_at_k': pass_at_k_metrics,
            'summary': summary
        }
