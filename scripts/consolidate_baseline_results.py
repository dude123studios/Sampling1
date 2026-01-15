
import json
import os
from pathlib import Path

def load_json(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)

def get_key(entry):
    return (entry['problem_id'], entry['cutoff_pos'])

def main():
    base_dir = Path("mech_interp/random_baseline_results")
    full_path = base_dir / "random_baseline_full.json"
    ds_patch_path = base_dir / "random_baseline_deepseek_patching.json"
    ds_grad_path = base_dir / "random_baseline_deepseek_gradient.json"
    
    full_data = load_json(full_path)
    ds_patch_data = load_json(ds_patch_path)
    ds_grad_data = load_json(ds_grad_path)
    
    print(f"Full Data (Qwen Run?): {len(full_data)} entries")
    print(f"DeepSeek Patching: {len(ds_patch_data)} entries")
    print(f"DeepSeek Gradient: {len(ds_grad_data)} entries")
    
    # Identify DeepSeek Logs based on ds_patch_data
    ds_log_keys = set(get_key(e) for e in ds_patch_data)
    
    # Separation
    qwen_on_qwen = []
    qwen_on_deepseek = []
    
    for entry in full_data:
        k = get_key(entry)
        if k in ds_log_keys:
            qwen_on_deepseek.append(entry)
        else:
            qwen_on_qwen.append(entry)
            
    print(f"\nSeparation of 'Full' (Qwen Model Run):")
    print(f"  - Qwen on Qwen Logs: {len(qwen_on_qwen)}")
    print(f"  - Qwen on DeepSeek Logs: {len(qwen_on_deepseek)}")
    
    # Construct Consolidated Results
    # 1. Qwen Baseline (Qwen on Qwen)
    #    Has DLA, Patching, Gradient from full.json
    with open(base_dir / "final_baseline_qwen.json", 'w') as f:
        json.dump(qwen_on_qwen, f, indent=2)
    print(f"\nSaved Qwen Baseline: {base_dir / 'final_baseline_qwen.json'}")
    
    # 2. DeepSeek Baseline (DeepSeek on DeepSeek)
    #    We have Patching (ds_patch_data) and Gradient (ds_grad_path)
    #    We merge them.
    #    We MISS DLA.
    
    ds_baseline = {}
    for e in ds_patch_data:
        k = get_key(e)
        ds_baseline[k] = e.copy()
        
    for e in ds_grad_data:
        k = get_key(e)
        if k in ds_baseline:
            ds_baseline[k].update(e) # Merge gradient info
        else:
            ds_baseline[k] = e
            
    ds_final = list(ds_baseline.values())
    with open(base_dir / "final_baseline_deepseek.json", 'w') as f:
        json.dump(ds_final, f, indent=2)
    print(f"Saved DeepSeek Baseline: {base_dir / 'final_baseline_deepseek.json'}")

if __name__ == "__main__":
    main()
