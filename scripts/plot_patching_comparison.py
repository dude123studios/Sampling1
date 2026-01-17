"""
Plot beautiful patching comparison between qwen3-8b and deepseek-qwen3-8b.

Uses the corrected patching results with layer-specific causal effects.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


def load_patching_data(filepath: str) -> List[Dict]:
    """Load patching results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_patching_by_layer(data: List[Dict]) -> Tuple[List[int], List[float], List[float]]:
    """
    Aggregate patching effects by layer.

    Returns:
        layers: List of layer indices
        mean_effects: Mean patching effect for each layer
        stderr_effects: Standard error for each layer
    """
    # Get all layers
    layers = sorted([int(k) for k in data[0]['patching'].keys()])

    # Aggregate effects per layer
    layer_effects = defaultdict(list)
    for entry in data:
        for layer_str, effect in entry['patching'].items():
            layer = int(layer_str)
            layer_effects[layer].append(effect)

    mean_effects = [np.mean(layer_effects[layer]) for layer in layers]
    stderr_effects = [np.std(layer_effects[layer]) / np.sqrt(len(layer_effects[layer]))
                      for layer in layers]

    return layers, mean_effects, stderr_effects


def create_patching_comparison_bar(
    qwen_data: List[Dict],
    deepseek_data: List[Dict],
    output_path: str = None
):
    """
    Create beautiful bar graph comparing patching effects.
    Shows every 2nd layer for clarity.
    """
    qwen_layers, qwen_mean, qwen_stderr = aggregate_patching_by_layer(qwen_data)
    ds_layers, ds_mean, ds_stderr = aggregate_patching_by_layer(deepseek_data)

    # Select every 2nd layer to avoid crowding
    step = 2
    display_layers = qwen_layers[::step]
    qwen_subset = qwen_mean[::step]
    qwen_err_subset = qwen_stderr[::step]
    ds_subset = ds_mean[::step]
    ds_err_subset = ds_stderr[::step]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Beautiful colors
    qwen_color = '#2E86AB'      # Blue
    deepseek_color = '#F18F01'  # Orange

    # Bar positions
    x = np.arange(len(display_layers))
    width = 0.38

    # Bars with black edges
    bars1 = ax.bar(
        x - width/2,
        qwen_subset,
        width,
        yerr=qwen_err_subset,
        label='qwen3-8b',
        color=qwen_color,
        edgecolor='black',
        linewidth=1.5,
        capsize=6,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )

    bars2 = ax.bar(
        x + width/2,
        ds_subset,
        width,
        yerr=ds_err_subset,
        label='deepseek-qwen3-8b',
        color=deepseek_color,
        edgecolor='black',
        linewidth=1.5,
        capsize=6,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )

    # Styling
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Patching Effect (Δ logit difference)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Activation Patching: Causal Layer Effects Comparison',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display_layers, fontsize=11)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='upper left',
        frameon=True,
        fontsize=12,
        edgecolor='black',
        framealpha=1.0,
        fancybox=False
    )

    # Subtitle
    subtitle = 'Positive values indicate layer promotes top1 token preference'
    ax.text(
        0.5, -0.12,
        subtitle,
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved patching comparison bar plot to: {output_path}")

    return fig, ax


def create_patching_line_comparison(
    qwen_data: List[Dict],
    deepseek_data: List[Dict],
    output_path: str = None
):
    """
    Create beautiful line graph showing patching effects across all layers.
    """
    qwen_layers, qwen_mean, qwen_stderr = aggregate_patching_by_layer(qwen_data)
    ds_layers, ds_mean, ds_stderr = aggregate_patching_by_layer(deepseek_data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Beautiful colors
    qwen_color = '#2E86AB'
    deepseek_color = '#F18F01'

    # Lines with markers
    ax.plot(
        qwen_layers,
        qwen_mean,
        marker='o',
        markersize=7,
        linewidth=2.5,
        color=qwen_color,
        markeredgecolor='black',
        markeredgewidth=1.2,
        label='qwen3-8b',
        zorder=3
    )

    ax.plot(
        ds_layers,
        ds_mean,
        marker='s',
        markersize=7,
        linewidth=2.5,
        color=deepseek_color,
        markeredgecolor='black',
        markeredgewidth=1.2,
        label='deepseek-qwen3-8b',
        zorder=3
    )

    # Shaded error regions
    ax.fill_between(
        qwen_layers,
        [m - s for m, s in zip(qwen_mean, qwen_stderr)],
        [m + s for m, s in zip(qwen_mean, qwen_stderr)],
        alpha=0.2,
        color=qwen_color,
        zorder=2
    )

    ax.fill_between(
        ds_layers,
        [m - s for m, s in zip(ds_mean, ds_stderr)],
        [m + s for m, s in zip(ds_mean, ds_stderr)],
        alpha=0.2,
        color=deepseek_color,
        zorder=2
    )

    # Styling
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Patching Effect (Δ logit difference)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Activation Patching: Layer-wise Causal Effects',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # X-ticks every 4 layers
    ax.set_xticks(range(0, max(qwen_layers) + 1, 4))
    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='upper left',
        frameon=True,
        fontsize=12,
        edgecolor='black',
        framealpha=1.0,
        fancybox=False
    )

    # Subtitle
    subtitle = f'n = {len(qwen_data)} problems per model'
    ax.text(
        0.5, -0.12,
        subtitle,
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved patching line comparison to: {output_path}")

    return fig, ax


def create_heatmap_comparison(
    qwen_data: List[Dict],
    deepseek_data: List[Dict],
    output_path: str = None
):
    """
    Create side-by-side heatmaps showing patching effects per problem.
    """
    # Get layers
    layers = sorted([int(k) for k in qwen_data[0]['patching'].keys()])

    # Build matrices: rows = problems, cols = layers
    n_problems = len(qwen_data)
    qwen_matrix = np.zeros((n_problems, len(layers)))
    ds_matrix = np.zeros((n_problems, len(layers)))

    for i, (qwen_entry, ds_entry) in enumerate(zip(qwen_data, deepseek_data)):
        for j, layer in enumerate(layers):
            qwen_matrix[i, j] = qwen_entry['patching'][str(layer)]
            ds_matrix[i, j] = ds_entry['patching'][str(layer)]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Shared color limits for fair comparison
    vmin = min(qwen_matrix.min(), ds_matrix.min())
    vmax = max(qwen_matrix.max(), ds_matrix.max())

    # Qwen heatmap
    ax = axes[0]
    im1 = ax.imshow(
        qwen_matrix,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Problem Index', fontsize=12, fontweight='bold')
    ax.set_title('qwen3-8b', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(0, len(layers), 4))
    ax.set_xticklabels(layers[::4])

    # Deepseek heatmap
    ax = axes[1]
    im2 = ax.imshow(
        ds_matrix,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Problem Index', fontsize=12, fontweight='bold')
    ax.set_title('deepseek-qwen3-8b', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(0, len(layers), 4))
    ax.set_xticklabels(layers[::4])

    # Shared colorbar
    fig.colorbar(im2, ax=axes, label='Patching Effect', fraction=0.02, pad=0.04)

    # Overall title
    fig.suptitle(
        'Activation Patching Effects: Per-Problem Comparison',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved patching heatmap comparison to: {output_path}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Plot beautiful patching comparison"
    )
    parser.add_argument(
        '--qwen-file',
        type=str,
        default='mech_interp/random_baseline_results/random_baseline_full (2).json',
        help='Path to qwen3-8b patching results'
    )
    parser.add_argument(
        '--deepseek-file',
        type=str,
        default='mech_interp/random_baseline_results/random_baseline_full (3).json',
        help='Path to deepseek-qwen3-8b patching results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/plots',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading patching data...")
    qwen_data = load_patching_data(args.qwen_file)
    deepseek_data = load_patching_data(args.deepseek_file)

    print(f"Loaded qwen: {len(qwen_data)} entries")
    print(f"Loaded deepseek: {len(deepseek_data)} entries")
    print()

    print("Creating beautiful comparison plots...")
    print()

    # Bar comparison
    create_patching_comparison_bar(
        qwen_data,
        deepseek_data,
        output_dir / 'patching_comparison_bars.png'
    )

    # Line comparison
    create_patching_line_comparison(
        qwen_data,
        deepseek_data,
        output_dir / 'patching_comparison_lines.png'
    )

    # Heatmap comparison
    create_heatmap_comparison(
        qwen_data,
        deepseek_data,
        output_dir / 'patching_comparison_heatmaps.png'
    )

    print()
    print("Done! All plots saved to:", output_dir)


if __name__ == '__main__':
    main()
