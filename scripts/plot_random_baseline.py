
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)

def plot_dla(data, model_name, output_dir):
    """Plot DLA aggregated scores."""
    if not data or 'dla' not in data[0]:
        print(f"No DLA data for {model_name}")
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
            
    # Compute mean/std
    layers = sorted(layer_scores.keys())
    means = [np.mean(layer_scores[l]) for l in layers]
    sems = [np.std(layer_scores[l])/np.sqrt(len(layer_scores[l])) for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, means, yerr=sems, fmt='-o', capsize=5)
    plt.title(f'Direct Logit Attribution (Random Baseline) - {model_name}')
    plt.xlabel('Layer')
    plt.ylabel('DLA Score (contribution to logit diff)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"dla_{model_name}.png")
    plt.close()
    print(f"Saved DLA plot for {model_name}")

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
            
    layers = sorted(layer_norms.keys())
    means = [np.mean(layer_norms[l]) for l in layers]
    sems = [np.std(layer_norms[l])/np.sqrt(len(layer_norms[l])) for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, means, yerr=sems, fmt='-s', capsize=5, color='orange')
    plt.title(f'Gradient Attribution (Random Baseline) - {model_name}')
    plt.xlabel('Layer')
    plt.ylabel('Gradient L2 Norm')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"gradient_{model_name}.png")
    plt.close()
    print(f"Saved Gradient plot for {model_name}")

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
            
    layers = sorted(layer_effects.keys())
    means = [np.mean(layer_effects[l]) for l in layers]
    sems = [np.std(layer_effects[l])/np.sqrt(len(layer_effects[l])) for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, means, yerr=sems, fmt='-^', capsize=5, color='green')
    plt.title(f'Activation Patching (Random Baseline) - {model_name}')
    plt.xlabel('Layer')
    plt.ylabel('Logit Difference Recovery')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"patching_{model_name}.png")
    plt.close()
    print(f"Saved Patching plot for {model_name}")


def main():
    base_dir = Path("mech_interp/random_baseline_results")
    output_dir = Path("results/plots/random_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Qwen Data (Second half of full.json)
    full_data = load_json(base_dir / "random_baseline_full.json")
    if len(full_data) >= 134:
        # Assuming second half is Qwen Logs (as directory qwen > deepseek)
        qwen_data = full_data[134:] if len(full_data) == 268 else full_data # Fallback if size differs
        if len(full_data) == 268:
             print("Identified 268 entries. Using indices 134-268 for Qwen Baseline.")
        else:
             print(f"Warning: Full data size {len(full_data)} != 268. Using all as Qwen?")
             qwen_data = full_data
             
        plot_dla(qwen_data, "qwen3-8b", output_dir)
        plot_gradient(qwen_data, "qwen3-8b", output_dir)
        plot_patching(qwen_data, "qwen3-8b", output_dir)
    
    # 2. Load DeepSeek Data
    ds_patch = load_json(base_dir / "random_baseline_deepseek_patching.json")
    ds_grad = load_json(base_dir / "random_baseline_deepseek_gradient.json")
    
    # Merge for DeepSeek (by index assuming ordered, or just plot separately)
    # We can plot separately easily.
    
    if ds_patch:
        plot_patching(ds_patch, "deepseek-R1", output_dir)
    
    if ds_grad:
        plot_gradient(ds_grad, "deepseek-R1", output_dir)
        
    print("\nVisualizations saved to results/plots/random_baseline/")

if __name__ == "__main__":
    main()
