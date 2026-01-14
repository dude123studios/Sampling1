import argparse
import json
import os
import glob
from collections import defaultdict

def load_results(result_dir):
    """
    Loads results from a directory.
    Returns a dict mapping problem_id to the result object.
    It looks for 'log.jsonl' in the directory.
    """
    # Find log.jsonl
    log_path = os.path.join(result_dir, "log.jsonl")
    if not os.path.exists(log_path):
        # Try finding any .jsonl file
        jsonls = glob.glob(os.path.join(result_dir, "*.jsonl"))
        if not jsonls:
            raise FileNotFoundError(f"No log.jsonl found in {result_dir}")
        log_path = jsonls[0]

    results = {}
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if "type" in data and data["type"] == "summary":
                    continue
                # Use 'dataset_id' as the unique key for the problem
                if "dataset_id" in data:
                    results[data["dataset_id"]] = data
                elif "id" in data:
                     results[data["id"]] = data
            except json.JSONDecodeError:
                pass
    return results

def compare_runs(baseline_dir, experiment_dir):
    print(f"Loading Baseline: {baseline_dir}")
    baseline = load_results(baseline_dir)
    print(f"Loading Experiment: {experiment_dir}")
    experiment = load_results(experiment_dir)

    common_ids = set(baseline.keys()) & set(experiment.keys())
    print(f"Found {len(common_ids)} common problems.")

    new_solves = []
    lost_solves = []
    both_solved = []
    both_failed = []

    for pid in common_ids:
        b_res = baseline[pid]
        e_res = experiment[pid]

        # Check if solved at least once (one@k > 0)
        # Some older logs might not have one@k, fallback to num_correct > 0
        b_solved = b_res.get("metrics", {}).get("one@k", 0) > 0 or b_res.get("metrics", {}).get("num_correct", 0) > 0
        e_solved = e_res.get("metrics", {}).get("one@k", 0) > 0 or e_res.get("metrics", {}).get("num_correct", 0) > 0

        if not b_solved and e_solved:
            new_solves.append(pid)
        elif b_solved and not e_solved:
            lost_solves.append(pid)
        elif b_solved and e_solved:
            both_solved.append(pid)
        else:
            both_failed.append(pid)

    print("\n" + "="*40)
    print("COMPARISON REPORT")
    print("="*40)
    print(f"Total Common Problems: {len(common_ids)}")
    print(f"Both Solved:           {len(both_solved)}")
    print(f"Both Failed:           {len(both_failed)}")
    print("-" * 20)
    print(f"New Solves (Exp Solved, Base Failed): {len(new_solves)}")
    print(f"Lost Solves (Base Solved, Exp Failed): {len(lost_solves)}")
    print("="*40)

    if new_solves:
        print("\nIDs of New Solves:")
        print(new_solves)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two experiment runs.")
    parser.add_argument("baseline_dir", help="Path to the baseline results directory")
    parser.add_argument("experiment_dir", help="Path to the experiment results directory")
    args = parser.parse_args()

    compare_runs(args.baseline_dir, args.experiment_dir)
