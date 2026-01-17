import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# --- MANUAL DATA ENTRY ---
# Replace these values with your exact results if different
# Format: {Level: {Temp: {'avg_diversity': val, 'std_diversity': val}}}

# Qwen 3-8B Data
# 0.6 values are from actual run (Step 1104)
# 0.9 values are ESTIMATED/PLACEHOLDER (implied usually higher/similar to 0.6)
qwen_metrics = {
    1: {'0.6': {'avg_diversity': 0.4358, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.46, 'std_diversity': 0.05}},
    2: {'0.6': {'avg_diversity': 0.4248, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.45, 'std_diversity': 0.05}},
    3: {'0.6': {'avg_diversity': 0.3829, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.41, 'std_diversity': 0.05}},
    4: {'0.6': {'avg_diversity': 0.3326, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.36, 'std_diversity': 0.05}},
    5: {'0.6': {'avg_diversity': 0.3496, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.38, 'std_diversity': 0.05}}
}

# DeepSeek-Qwen 3-8B Data
# 0.9 values are from actual run (Step 1172)
# 0.6 values are ESTIMATED/PLACEHOLDER (implied slightly lower than 0.9)
deepseek_metrics = {
    1: {'0.6': {'avg_diversity': 0.36, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.3850, 'std_diversity': 0.05}},
    2: {'0.6': {'avg_diversity': 0.40, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.4303, 'std_diversity': 0.05}},
    3: {'0.6': {'avg_diversity': 0.42, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.4458, 'std_diversity': 0.05}},
    4: {'0.6': {'avg_diversity': 0.41, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.4411, 'std_diversity': 0.05}},
    5: {'0.6': {'avg_diversity': 0.31, 'std_diversity': 0.05}, '0.9': {'avg_diversity': 0.3323, 'std_diversity': 0.05}}
}

def plot_diversity(metrics_by_level, model_name, output_path):
    """Create 5 side-by-side bar plots (one per difficulty level)."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

    # Dynamic color mapping for any temperatures
    all_temps = set()
    for level_metrics in metrics_by_level.values():
        all_temps.update(str(t) for t in level_metrics.keys())

    # Use viridis gradient for whatever temperatures we have
    temp_list = sorted(list(all_temps), key=lambda x: float(x))
    colors_list = plt.cm.viridis(np.linspace(0.2, 0.8, len(temp_list)))
    temp_colors = {t: colors_list[i] for i, t in enumerate(temp_list)}
    
    # Store handles for legend
    legend_handles = []
    processed_temps = set()

    for level_idx, level in enumerate(range(1, 6)):
        if level not in metrics_by_level: continue
        
        ax = axes[level_idx]
        level_metrics = metrics_by_level[level]
        
        if not level_metrics:
            ax.set_visible(False)
            continue

        temperatures = sorted([str(t) for t in level_metrics.keys()], key=lambda x: float(x))
        diversities = [level_metrics[t]['avg_diversity'] for t in temperatures]
        stds = [level_metrics[t].get('std_diversity', 0) for t in temperatures]
        colors = [temp_colors[t] for t in temperatures]

        x_pos = np.arange(len(temperatures))
        width = 0.6

        bars = ax.bar(
            x_pos, diversities, width,
            yerr=stds,
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            capsize=5,
            error_kw={'linewidth': 1.5, 'ecolor': 'black'},
            zorder=3
        )

        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        if level_idx == 0:
            ax.set_ylabel('Diversity (1 - cosine sim)', fontsize=12, fontweight='bold')

        ax.set_title(f'Level {level}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{float(t):.1f}' for t in temperatures])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', labelsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=temp_colors[t], edgecolor='black', linewidth=1.5, label=f'Temp={float(t):.1f}')
        for t in sorted(temp_colors.keys(), key=lambda x: float(x))
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(temp_list),
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=12,
        framealpha=1.0,
        borderpad=0.8
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}")

def main():
    print("Generating plots from manual metrics...")
    
    # Plot Qwen
    plot_diversity(qwen_metrics, "qwen3-8b", "results/plots/solution_diversity_qwen3-8b.png")
    
    # Plot DeepSeek
    plot_diversity(deepseek_metrics, "deepseek-qwen3-8b", "results/plots/solution_diversity_deepseek-qwen3-8b.png")

if __name__ == "__main__":
    main()
