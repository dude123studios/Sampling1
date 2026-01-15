"""
Plot DLA (Direct Logit Attribution) results for qwen3-8b with publication-quality styling.

Visualizes:
1. Layer-wise DLA scores (which layers contribute to token preference)
2. Gradient sensitivity across layers
3. Patching effects (causal intervention results)
4. Comprehensive overview panel
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


def load_dla_results(filepath: str) -> List[Dict]:
    """Load DLA results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_dla_by_layer(results: List[Dict]) -> Tuple[List[int], List[float], List[float]]:
    """
    Aggregate DLA scores by layer.

    Returns:
        layers: List of layer indices
        mean_dla: Mean DLA score for each layer
        stderr_dla: Standard error for each layer
    """
    # Get all layers
    first_result = results[0]
    layers = sorted([int(k) for k in first_result['dla'].keys()])

    # Collect DLA scores per layer
    layer_scores = defaultdict(list)
    for result in results:
        for layer_str, score in result['dla'].items():
            layer = int(layer_str)
            layer_scores[layer].append(score)

    mean_dla = [np.mean(layer_scores[layer]) for layer in layers]
    stderr_dla = [np.std(layer_scores[layer]) / np.sqrt(len(layer_scores[layer]))
                  for layer in layers]

    return layers, mean_dla, stderr_dla


def aggregate_gradient_by_layer(results: List[Dict]) -> Tuple[List[int], List[float], List[float]]:
    """Aggregate gradient norms by layer."""
    # Get gradient layers
    first_result = results[0]
    layers = sorted([int(k) for k in first_result['gradient'].keys()])

    # Collect gradient scores per layer
    layer_grads = defaultdict(list)
    for result in results:
        for layer_str, score in result['gradient'].items():
            layer = int(layer_str)
            layer_grads[layer].append(score)

    mean_grad = [np.mean(layer_grads[layer]) for layer in layers]
    stderr_grad = [np.std(layer_grads[layer]) / np.sqrt(len(layer_grads[layer]))
                   for layer in layers]

    return layers, mean_grad, stderr_grad


def aggregate_patching_by_layer(results: List[Dict]) -> Tuple[List[int], List[float], List[float]]:
    """Aggregate patching effects by layer."""
    # Get patching layers
    first_result = results[0]
    layers = sorted([int(k) for k in first_result['patching'].keys()])

    # Collect patching scores per layer
    layer_patches = defaultdict(list)
    for result in results:
        for layer_str, score in result['patching'].items():
            layer = int(layer_str)
            layer_patches[layer].append(score)

    mean_patch = [np.mean(layer_patches[layer]) for layer in layers]
    stderr_patch = [np.std(layer_patches[layer]) / np.sqrt(len(layer_patches[layer]))
                    for layer in layers]

    return layers, mean_patch, stderr_patch


def create_dla_line_plot(
    results: List[Dict],
    output_path: str = None
):
    """
    Create line plot of DLA scores across layers.

    Shows which layers contribute most to preferring top1 over top2 token.
    """
    layers, mean_dla, stderr_dla = aggregate_dla_by_layer(results)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Line plot with shaded error region
    ax.plot(
        layers,
        mean_dla,
        marker='o',
        markersize=5,
        linewidth=2.5,
        color='#2E86AB',
        markeredgecolor='black',
        markeredgewidth=0.8,
        label='Mean DLA Score',
        zorder=3
    )

    # Shaded error region
    ax.fill_between(
        layers,
        [m - s for m, s in zip(mean_dla, stderr_dla)],
        [m + s for m, s in zip(mean_dla, stderr_dla)],
        alpha=0.2,
        color='#2E86AB',
        zorder=2
    )

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

    # Styling
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('DLA Score (logit contribution)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Direct Logit Attribution: Layer-wise Contribution to Token Preference (qwen3-8b)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Set x-ticks to show every 2 layers
    ax.set_xticks(range(0, max(layers) + 1, 2))
    ax.tick_params(axis='both', labelsize=11)

    # Subtitle
    n_samples = len(results)
    subtitle = f'n = {n_samples} samples across 134 problems'
    ax.text(
        0.5, -0.15,
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
        print(f"Saved DLA line plot to: {output_path}")

    return fig, ax


def create_gradient_bar_plot(
    results: List[Dict],
    output_path: str = None
):
    """
    Create bar plot of gradient norms across layers.

    Shows which layers are most sensitive to changes.
    """
    layers, mean_grad, stderr_grad = aggregate_gradient_by_layer(results)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Bar plot with gradient colors
    colors = plt.cm.Reds(np.array(mean_grad) / max(mean_grad))

    bars = ax.bar(
        layers,
        mean_grad,
        yerr=stderr_grad,
        color=colors,
        edgecolor='black',
        linewidth=1.5,
        capsize=6,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )

    # Styling
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gradient Norm (L2)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Gradient Sensitivity: Layer Activation Sensitivity to Logit Difference (qwen3-8b)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.set_xticks(layers)
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved gradient bar plot to: {output_path}")

    return fig, ax


def create_patching_line_plot(
    results: List[Dict],
    output_path: str = None
):
    """
    Create line plot of patching effects across layers.

    Shows which layers have the largest causal effect when intervened upon.
    """
    layers, mean_patch, stderr_patch = aggregate_patching_by_layer(results)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Line plot
    ax.plot(
        layers,
        mean_patch,
        marker='s',
        markersize=5,
        linewidth=2.5,
        color='#F18F01',  # Orange
        markeredgecolor='black',
        markeredgewidth=0.8,
        label='Mean Patching Effect',
        zorder=3
    )

    # Shaded error region
    ax.fill_between(
        layers,
        [m - s for m, s in zip(mean_patch, stderr_patch)],
        [m + s for m, s in zip(mean_patch, stderr_patch)],
        alpha=0.2,
        color='#F18F01',
        zorder=2
    )

    # Styling
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Patching Effect (logit difference change)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Activation Patching: Causal Layer Importance (qwen3-8b)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.set_xticks(range(0, max(layers) + 1, 2))
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved patching line plot to: {output_path}")

    return fig, ax


def create_comprehensive_panel(
    results: List[Dict],
    output_path: str = None
):
    """
    Create comprehensive 3-panel figure showing DLA, gradients, and patching.
    """
    layers_dla, mean_dla, stderr_dla = aggregate_dla_by_layer(results)
    layers_grad, mean_grad, stderr_grad = aggregate_gradient_by_layer(results)
    layers_patch, mean_patch, stderr_patch = aggregate_patching_by_layer(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left: DLA ---
    ax = axes[0]
    ax.plot(
        layers_dla, mean_dla,
        marker='o', markersize=4, linewidth=2,
        color='#2E86AB', markeredgecolor='black', markeredgewidth=0.8,
        zorder=3
    )
    ax.fill_between(
        layers_dla,
        [m - s for m, s in zip(mean_dla, stderr_dla)],
        [m + s for m, s in zip(mean_dla, stderr_dla)],
        alpha=0.2, color='#2E86AB', zorder=2
    )
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('DLA Score', fontsize=12, fontweight='bold')
    ax.set_title('Direct Logit Attribution', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(range(0, max(layers_dla) + 1, 4))

    # --- Middle: Gradient ---
    ax = axes[1]
    colors_grad = plt.cm.Reds(np.array(mean_grad) / max(mean_grad))
    ax.bar(
        layers_grad, mean_grad,
        yerr=stderr_grad,
        color=colors_grad, edgecolor='black', linewidth=1.5,
        capsize=6, error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Sensitivity', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(layers_grad)

    # --- Right: Patching ---
    ax = axes[2]
    ax.plot(
        layers_patch, mean_patch,
        marker='s', markersize=4, linewidth=2,
        color='#F18F01', markeredgecolor='black', markeredgewidth=0.8,
        zorder=3
    )
    ax.fill_between(
        layers_patch,
        [m - s for m, s in zip(mean_patch, stderr_patch)],
        [m + s for m, s in zip(mean_patch, stderr_patch)],
        alpha=0.2, color='#F18F01', zorder=2
    )
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Patching Effect', fontsize=12, fontweight='bold')
    ax.set_title('Activation Patching', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(range(0, max(layers_patch) + 1, 4))

    # Overall title
    fig.suptitle(
        'Mechanistic Interpretability Analysis: qwen3-8b',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comprehensive panel to: {output_path}")

    return fig, axes


def create_dla_heatmap(
    results: List[Dict],
    output_path: str = None
):
    """
    Create heatmap showing DLA scores across layers and cutoff positions.
    """
    # Organize data by cutoff position and layer
    cutoff_positions = sorted(set(r['cutoff_pos'] for r in results))
    layers = sorted([int(k) for k in results[0]['dla'].keys()])

    # Build matrix: rows = cutoff positions, cols = layers
    matrix = np.zeros((len(cutoff_positions), len(layers)))

    for i, cutoff in enumerate(cutoff_positions):
        cutoff_results = [r for r in results if r['cutoff_pos'] == cutoff]
        for j, layer in enumerate(layers):
            layer_scores = [r['dla'][str(layer)] for r in cutoff_results]
            matrix[i, j] = np.mean(layer_scores)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Heatmap
    im = ax.imshow(
        matrix,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest',
        vmin=-np.abs(matrix).max(),
        vmax=np.abs(matrix).max()
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean DLA Score', fontsize=12, fontweight='bold')

    # Styling
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cutoff Position', fontsize=13, fontweight='bold')
    ax.set_title(
        'DLA Scores Across Layers and Cutoff Positions (qwen3-8b)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Set ticks
    ax.set_xticks(range(0, len(layers), 2))
    ax.set_xticklabels(layers[::2])
    ax.set_yticks(range(len(cutoff_positions)))
    ax.set_yticklabels(cutoff_positions)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved DLA heatmap to: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description="Plot DLA results for qwen3-8b")
    parser.add_argument(
        '--results',
        type=str,
        default='mech_interp/random_baseline_results/random_baseline_full.json',
        help='Path to DLA results JSON'
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

    print("Loading DLA results for qwen3-8b...")
    results = load_dla_results(args.results)

    n_problems = len(set(r['problem_id'] for r in results))
    print(f"Loaded {len(results)} results from {n_problems} problems")
    print()

    print("Creating publication-quality plots...")
    print()

    # Individual plots
    create_dla_line_plot(
        results,
        output_dir / 'dla_qwen3_8b_layer_contributions.png'
    )

    create_gradient_bar_plot(
        results,
        output_dir / 'dla_qwen3_8b_gradient_sensitivity.png'
    )

    create_patching_line_plot(
        results,
        output_dir / 'dla_qwen3_8b_patching_effects.png'
    )

    create_dla_heatmap(
        results,
        output_dir / 'dla_qwen3_8b_heatmap.png'
    )

    # Comprehensive panel
    create_comprehensive_panel(
        results,
        output_dir / 'dla_qwen3_8b_comprehensive.png'
    )

    print()
    print("Done!")
    print(f"\nView plots in: {output_dir}")


if __name__ == '__main__':
    main()
