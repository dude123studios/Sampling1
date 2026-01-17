"""
Quick test script for HumanEval evaluator.

Verifies that:
1. Dataset loads correctly
2. Prompts format correctly
3. Code execution works
4. Pass@k estimation is accurate

Usage:
    python scripts/test_humaneval.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluators.code_eval import (
    HumanEvalEvaluator,
    execute_code_safely,
    estimate_pass_at_k,
    compute_pass_at_k_metrics
)
import numpy as np


def test_dataset_loading():
    """Test that HumanEval dataset loads correctly."""
    print("\n" + "="*60)
    print("TEST 1: Dataset Loading")
    print("="*60)

    evaluator = HumanEvalEvaluator()
    problems = evaluator.problems

    print(f"✓ Loaded {len(problems)} problems")
    assert len(problems) == 164, f"Expected 164 problems, got {len(problems)}"

    # Check first problem structure
    p0 = problems[0]
    required_keys = ['task_id', 'prompt', 'test', 'entry_point']
    for key in required_keys:
        assert key in p0, f"Missing key: {key}"

    print(f"✓ First problem: {p0['task_id']}")
    print(f"✓ Entry point: {p0['entry_point']}")
    print("\n✅ Dataset loading test PASSED")


def test_prompt_formatting():
    """Test prompt formatting."""
    print("\n" + "="*60)
    print("TEST 2: Prompt Formatting")
    print("="*60)

    evaluator = HumanEvalEvaluator()
    problem = evaluator.problems[0]

    system_prompt, user_prompt = evaluator.format_prompt(problem)

    print(f"System prompt length: {len(system_prompt)}")
    print(f"User prompt length: {len(user_prompt)}")

    assert "coding assistant" in system_prompt.lower(), "System prompt doesn't contain expected text"
    assert "@@ Instruction" in user_prompt, "User prompt missing instruction marker"
    assert problem['prompt'] in user_prompt, "User prompt missing problem text"

    print("\n✓ System prompt formatted correctly")
    print("\n✓ User prompt formatted correctly")
    print("\n✅ Prompt formatting test PASSED")


def test_code_execution():
    """Test code execution with known examples."""
    print("\n" + "="*60)
    print("TEST 3: Code Execution")
    print("="*60)

    # Test 1: Correct solution
    correct_code = """
def add(a, b):
    return a + b
"""
    test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0

check(add)
"""
    result = execute_code_safely(correct_code, test_code)
    assert result == True, "Correct code should pass"
    print("✓ Correct code passes tests")

    # Test 2: Incorrect solution
    incorrect_code = """
def add(a, b):
    return a - b  # Wrong operation
"""
    result = execute_code_safely(incorrect_code, test_code)
    assert result == False, "Incorrect code should fail"
    print("✓ Incorrect code fails tests")

    # Test 3: Timeout (infinite loop)
    timeout_code = """
def add(a, b):
    while True:
        pass
    return a + b
"""
    result = execute_code_safely(timeout_code, test_code, timeout=1)
    assert result == False, "Timeout code should fail"
    print("✓ Timeout handling works")

    # Test 4: Syntax error
    syntax_error_code = """
def add(a, b)
    return a + b  # Missing colon
"""
    result = execute_code_safely(syntax_error_code, test_code)
    assert result == False, "Syntax error should fail"
    print("✓ Syntax errors handled correctly")

    print("\n✅ Code execution test PASSED")


def test_pass_at_k_estimator():
    """Test pass@k estimation formula."""
    print("\n" + "="*60)
    print("TEST 4: Pass@k Estimator")
    print("="*60)

    # Test cases from Chen et al. (2021)
    test_cases = [
        # (n, c, k, expected)
        (10, 5, 1, 0.5),           # 5/10 correct, pass@1 = 50%
        (10, 10, 1, 1.0),          # All correct, pass@1 = 100%
        (10, 0, 1, 0.0),           # None correct, pass@1 = 0%
        (100, 50, 5, None),        # Will compute
    ]

    for n, c, k, expected in test_cases:
        result = estimate_pass_at_k(n, c, k)
        if expected is not None:
            assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"
            print(f"✓ pass@{k} with n={n}, c={c}: {result:.4f} (expected {expected})")
        else:
            print(f"✓ pass@{k} with n={n}, c={c}: {result:.4f}")

    # Edge cases
    assert estimate_pass_at_k(5, 3, 10) == 0.0, "k > n should return 0"
    print("✓ Edge case: k > n returns 0")

    assert estimate_pass_at_k(10, 7, 5) == 1.0, "c >= k should return 1"
    print("✓ Edge case: c >= k returns 1")

    print("\n✅ Pass@k estimator test PASSED")


def test_real_problem():
    """Test evaluation on a real HumanEval problem."""
    print("\n" + "="*60)
    print("TEST 5: Real Problem Evaluation")
    print("="*60)

    evaluator = HumanEvalEvaluator()

    # Get problem 0 (two_sum or similar)
    problem = evaluator.problems[0]
    print(f"Testing on: {problem['task_id']}")

    # Generate some mock solutions (mix of correct and incorrect)
    # For HumanEval/0, the problem is typically about has_close_elements
    # We'll use the canonical solution as a correct example

    canonical = problem.get('canonical_solution', '')

    # Create test solutions
    solutions = [
        problem['prompt'] + canonical,  # Should be correct
        problem['prompt'] + "    return False\n",  # Likely incorrect
        problem['prompt'] + "    return True\n",   # Likely incorrect
    ]

    result = evaluator.evaluate_problem(problem, solutions)

    print(f"\n✓ Evaluated {result['num_samples']} solutions")
    print(f"✓ Correctness: {result['correctness']}")
    print(f"✓ Number correct: {result['num_correct']}")

    assert result['num_samples'] == 3, "Should have 3 samples"
    assert result['num_correct'] >= 0, "Should have non-negative correct count"

    # If canonical solution is provided, first should likely be correct
    if canonical and result['correctness'][0]:
        print("✓ Canonical solution passed (as expected)")

    print("\n✅ Real problem evaluation test PASSED")


def test_compute_metrics():
    """Test metric computation across multiple problems."""
    print("\n" + "="*60)
    print("TEST 6: Metrics Computation")
    print("="*60)

    # Mock results for 3 problems
    results = [
        {
            'task_id': 'test/0',
            'correctness': [True, True, False, False, False],  # 2/5 correct
            'num_samples': 5,
            'num_correct': 2
        },
        {
            'task_id': 'test/1',
            'correctness': [True, True, True, False, False],  # 3/5 correct
            'num_samples': 5,
            'num_correct': 3
        },
        {
            'task_id': 'test/2',
            'correctness': [False, False, False, False, False],  # 0/5 correct
            'num_samples': 5,
            'num_correct': 0
        },
    ]

    metrics = compute_pass_at_k_metrics(results, k_values=[1, 2, 3])

    print("\nComputed metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    assert 'pass@1' in metrics, "Should compute pass@1"
    assert 0.0 <= metrics['pass@1'] <= 1.0, "pass@1 should be in [0, 1]"

    # Average pass@1 should be around (2/5 + 3/5 + 0/5) / 3 = 5/15 = 0.333
    expected_pass1 = (2/5 + 3/5 + 0/5) / 3
    assert abs(metrics['pass@1'] - expected_pass1) < 0.01, f"pass@1 should be ~{expected_pass1}"

    print(f"\n✓ pass@1 matches expected value: {expected_pass1:.4f}")
    print("\n✅ Metrics computation test PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("HumanEval Evaluator Test Suite")
    print("="*60)

    tests = [
        test_dataset_loading,
        test_prompt_formatting,
        test_code_execution,
        test_pass_at_k_estimator,
        test_real_problem,
        test_compute_metrics,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED: {e}")
            failed.append((test.__name__, e))
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {len(tests) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return False
    else:
        print("\n🎉 All tests PASSED!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
