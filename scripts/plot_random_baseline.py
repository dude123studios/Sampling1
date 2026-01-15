
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set seaborn style for premium look
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = sns.color_palette("viridis")

def load_json(path):
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return []
    with open(path, 'r') as f:
        return json.load(f)

def plot_dla(data, model_name, output_dir):
    """Plot DLA aggregated scores."""
    if not data:
        print(f"No data provided for DLA {model_name}")
        return
        
    # Check if first entry has 'dla'
    if 'dla' not in data[0]:
        print(f"No 'dla' key in data for {model_name}")
        return

    # Aggregate by layer
    layer_scores = {} # layer -> list of scores
    for entry in data:
        scores = entry.get('dla', {})
        for layer, score in scores.items():
            l_idx = int(layer)
            if l_idx not in layer_scores:
                layer_scores[l_idx] = []
            layer_scores[l_idx].append(score)
            
    if not layer_scores:
        print(f"No DLA scores found for {model_name}")
        return

    # Compute mean/std
    layers = sorted(layer_scores.keys())
    means = [np.mean(layer_scores[l]) for l in layers]
    sems = [np.std(layer_scores[l])/np.sqrt(len(layer_scores[l])) for l in layers]
    
    plt.figure(figsize=(12, 7))
    plt.errorbar(layers, means, yerr=sems, fmt='-o', capsize=5, linewidth=2, markersize=8, color=palette[0], label='DLA Score')
    plt.title(f'Direct Logit Attribution (Random Baseline)\n{model_name}', fontsize=16, pad=20)
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Contribution to Logit Difference', fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    
    out_file = output_dir / f"dla_{model_name.lower().replace('-', '_')}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved DLA plot: {out_file}")

def plot_gradient(data, model_name, output_dir):
    """Plot Gradient norms."""
    if not data or 'gradient' not in data[0]:
        print(f"No Gradient data for {model_name}")
        return
        
    layer_norms = {}
    for entry in data:
        grads = entry.get('gradient', {})
        for layer, norm in grads.items():
            l_idx = int(layer)
            if l_idx not in layer_norms:
                layer_norms[l_idx] = []
            layer_norms[l_idx].append(norm)
            
    if not layer_norms:
        return

    layers = sorted(layer_norms.keys())
    means = [np.mean(layer_norms[l]) for l in layers]
    sems = [np.std(layer_norms[l])/np.sqrt(len(layer_norms[l])) for l in layers]
    
    plt.figure(figsize=(12, 7))
    plt.errorbar(layers, means, yerr=sems, fmt='-s', capsize=5, linewidth=2, markersize=8, color=palette[3], label='Gradient Norm')
    plt.title(f'Gradient Attribution (Random Baseline)\n{model_name}', fontsize=16, pad=20)
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Gradient L2 Norm', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    
    out_file = output_dir / f"gradient_{model_name.lower().replace('-', '_')}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved Gradient plot: {out_file}")

def plot_patching(data, model_name, output_dir):
    """Plot Patching effects."""
    if not data or 'patching' not in data[0]:
        print(f"No Patching data for {model_name}")
        return
        
    layer_effects = {}
    for entry in data:
        effs = entry.get('patching', {})
        for layer, effect in effs.items():
            l_idx = int(layer)
            if l_idx not in layer_effects:
                layer_effects[l_idx] = []
            layer_effects[l_idx].append(effect)
            
    if not layer_effects:
        return

    layers = sorted(layer_effects.keys())
    means = [np.mean(layer_effects[l]) for l in layers]
    sems = [np.std(layer_effects[l])/np.sqrt(len(layer_effects[l])) for l in layers]
    
    plt.figure(figsize=(12, 7))
    plt.errorbar(layers, means, yerr=sems, fmt='-^', capsize=5, linewidth=2, markersize=8, color=palette[4], label='Patching Effect')
    plt.title(f'Activation Patching (Random Baseline)\n{model_name}', fontsize=16, pad=20)
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Logit Difference Recovery', fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    
    out_file = output_dir / f"patching_{model_name.lower().replace('-', '_')}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved Patching plot: {out_file}")


def main():
    base_dir = Path("mech_interp/random_baseline_results")
    output_dir = Path("figures/random_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots in {output_dir}...")

    # 1. Qwen Data
    # Prefer consolidated file if available
    qwen_path = base_dir / "final_baseline_qwen.json"
    if qwen_path.exists():
        qwen_data = load_json(qwen_path)
    else:
        # Fallback to reconstructing from partials if needed, or error
        # Assuming user has run everything, let's look for separate files if consolidated missing
        print("Consolidated Qwen file not found, trying separate files...")
        qwen_patch = load_json(base_dir / "random_baseline_qwen_patching.json")
        qwen_grad = load_json(base_dir / "random_baseline_qwen_gradient.json")
        # Just plot separately in this case
        qwen_data = [] # Placeholder to skip combined logic
        if qwen_patch: plot_patching(qwen_patch, "Qwen3-8B", output_dir)
        if qwen_grad: plot_gradient(qwen_grad, "Qwen3-8B", output_dir)

    if qwen_data:
        plot_dla(qwen_data, "Qwen3-8B", output_dir)
        plot_gradient(qwen_data, "Qwen3-8B", output_dir)
        plot_patching(qwen_data, "Qwen3-8B", output_dir)
    
    # 2. DeepSeek Data
    ds_dla = load_json(base_dir / "random_baseline_deepseek_dla.json")
    ds_patch = load_json(base_dir / "random_baseline_deepseek_patching.json")
    ds_grad = load_json(base_dir / "random_baseline_deepseek_gradient.json")
    
    if ds_dla:
        plot_dla(ds_dla, "DeepSeek-R1", output_dir)
    else:
        print("Skipping DeepSeek DLA (No data found)")
        
    if ds_patch:
        plot_patching(ds_patch, "DeepSeek-R1", output_dir)
    
    if ds_grad:
        plot_gradient(ds_grad, "DeepSeek-R1", output_dir)
        
    print("\nDone. All plots generated.")

if __name__ == "__main__":
    main()
