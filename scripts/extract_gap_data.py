import argparse
import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def load_results(result_dir):
    """
    Loads results from a directory.
    Returns a dict mapping problem_id to the result object.
    """
    if os.path.isfile(result_dir):
        log_path = result_dir
    else:
        # Find log.jsonl
        log_path = os.path.join(result_dir, "log.jsonl")
        if not os.path.exists(log_path):
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
                # Use 'dataset_id' or 'id'
                key = data.get("dataset_id", data.get("id"))
                results[key] = data
            except json.JSONDecodeError:
                pass
    return results

def is_solved(res):
    if not res: return False
    metrics = res.get("metrics", {})
    return metrics.get("one@k", 0) > 0 or metrics.get("num_correct", 0) > 0

def get_first_correct_solution(res):
    scores = res.get("scores", [])
    outputs = res.get("outputs", [])
    for i, score in enumerate(scores):
        if score > 0:
            return outputs[i]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", help="Path to baseline (weak model) results")
    parser.add_argument("target_dir", help="Path to target (strong model) results")
    parser.add_argument("output_file", help="Path to save the extracted gap data (jsonl)")
    args = parser.parse_args()

    print(f"Loading Baseline: {args.baseline_dir}")
    baseline = load_results(args.baseline_dir)
    print(f"Loading Target: {args.target_dir}")
    target = load_results(args.target_dir)

    common_ids = set(baseline.keys()) & set(target.keys())
    print(f"Found {len(common_ids)} common problems.")

    gap_items = []
    
    for pid in common_ids:
        b_res = baseline[pid]
        t_res = target[pid]

        if not is_solved(b_res) and is_solved(t_res):
            # Target solved it, Baseline failed
            sol = get_first_correct_solution(t_res)
            if sol:
                item = {
                    "dataset_id": pid,
                    "original_prompt": t_res["original_prompt"],
                    "deepseek_solution": sol,
                    "gold": t_res["gold"]
                }
                gap_items.append(item)

    print(f"Identified {len(gap_items)} gap problems (solved by target only).")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for item in gap_items:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
