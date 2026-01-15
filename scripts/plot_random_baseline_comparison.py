"""
Plot comparison of random baseline mechanistic interpretability results
between qwen3-8b and deepseek-qwen3-8b.

Creates publication-quality bar graphs and line graphs comparing:
1. DLA (Direct Logit Attribution) scores
2. Gradient sensitivity
3. Patching effects
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


def load_data(base_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load qwen and deepseek random baseline results.

    Returns:
        qwen_data, deepseek_data
    """
    # Qwen data (from full file)
    with open(base_dir / 'random_baseline_full.json') as f:
        qwen_data = json.load(f)

    # Deepseek DLA data
    with open(base_dir / 'random_baseline_deepseek_dla.json') as f:
        deepseek_data = json.load(f)

    return qwen_data, deepseek_data


def aggregate_dla_by_layer(qwen_data: List[Dict], deepseek_data: List[Dict]) -> Tuple:
    """Aggregate DLA scores by layer for both models."""
    # Get all layers
    layers = sorted([int(k) for k in qwen_data[0]['dla'].keys()])

    # Aggregate for qwen
    qwen_scores = defaultdict(list)
    for entry in qwen_data:
        for layer_str, score in entry['dla'].items():
            layer = int(layer_str)
            qwen_scores[layer].append(score)

    # Aggregate for deepseek
    ds_scores = defaultdict(list)
    for entry in deepseek_data:
        for layer_str, score in entry['dla'].items():
            layer = int(layer_str)
            ds_scores[layer].append(score)

    qwen_mean = [np.mean(qwen_scores[l]) for l in layers]
    qwen_stderr = [np.std(qwen_scores[l]) / np.sqrt(len(qwen_scores[l])) for l in layers]

    ds_mean = [np.mean(ds_scores[l]) for l in layers]
    ds_stderr = [np.std(ds_scores[l]) / np.sqrt(len(ds_scores[l])) for l in layers]

    return layers, qwen_mean, qwen_stderr, ds_mean, ds_stderr


def aggregate_gradient_by_layer(base_dir: Path) -> Tuple:
    """Aggregate gradient norms by layer for both models."""
    # Load gradient data
    with open(base_dir / 'random_baseline_qwen_gradient.json') as f:
        qwen_grad = json.load(f)

    with open(base_dir / 'random_baseline_deepseek_gradient.json') as f:
        ds_grad = json.load(f)

    # Get gradient layers
    layers = sorted([int(k) for k in qwen_grad[0]['gradient'].keys()])

    # Aggregate qwen
    qwen_grads = defaultdict(list)
    for entry in qwen_grad:
        for layer_str, norm in entry['gradient'].items():
            layer = int(layer_str)
            qwen_grads[layer].append(norm)

    # Aggregate deepseek
    ds_grads = defaultdict(list)
    for entry in ds_grad:
        for layer_str, norm in entry['gradient'].items():
            layer = int(layer_str)
            ds_grads[layer].append(norm)

    qwen_mean = [np.mean(qwen_grads[l]) for l in layers]
    qwen_stderr = [np.std(qwen_grads[l]) / np.sqrt(len(qwen_grads[l])) for l in layers]

    ds_mean = [np.mean(ds_grads[l]) for l in layers]
    ds_stderr = [np.std(ds_grads[l]) / np.sqrt(len(ds_grads[l])) for l in layers]

    return layers, qwen_mean, qwen_stderr, ds_mean, ds_stderr


def aggregate_patching_by_layer(base_dir: Path) -> Tuple:
    """Aggregate patching effects by layer for both models."""
    # Load patching data
    with open(base_dir / 'random_baseline_qwen_patching.json') as f:
        qwen_patch = json.load(f)

    with open(base_dir / 'random_baseline_deepseek_patching.json') as f:
        ds_patch = json.load(f)

    # Get patching layers
    layers = sorted([int(k) for k in qwen_patch[0]['patching'].keys()])

    # Aggregate qwen
    qwen_patches = defaultdict(list)
    for entry in qwen_patch:
        for layer_str, effect in entry['patching'].items():
            layer = int(layer_str)
            qwen_patches[layer].append(effect)

    # Aggregate deepseek
    ds_patches = defaultdict(list)
    for entry in ds_patch:
        for layer_str, effect in entry['patching'].items():
            layer = int(layer_str)
            ds_patches[layer].append(effect)

    qwen_mean = [np.mean(qwen_patches[l]) for l in layers]
    qwen_stderr = [np.std(qwen_patches[l]) / np.sqrt(len(qwen_patches[l])) for l in layers]

    ds_mean = [np.mean(ds_patches[l]) for l in layers]
    ds_stderr = [np.std(ds_patches[l]) / np.sqrt(len(ds_patches[l])) for l in layers]

    return layers, qwen_mean, qwen_stderr, ds_mean, ds_stderr


def create_dla_comparison_bar(
    qwen_data: List[Dict],
    deepseek_data: List[Dict],
    output_path: str = None
):
    """
    Create bar graph comparing DLA scores between models.
    Shows only key layers for clarity.
    """
    layers, qwen_mean, qwen_stderr, ds_mean, ds_stderr = aggregate_dla_by_layer(
        qwen_data, deepseek_data
    )

    # Select key layers to display (every 4th layer to avoid crowding)
    key_layers = list(range(0, max(layers) + 1, 4))
    key_indices = [layers.index(l) for l in key_layers if l in layers]

    qwen_subset = [qwen_mean[i] for i in key_indices]
    qwen_err_subset = [qwen_stderr[i] for i in key_indices]
    ds_subset = [ds_mean[i] for i in key_indices]
    ds_err_subset = [ds_stderr[i] for i in key_indices]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Bar positions
    x = np.arange(len(key_layers))
    width = 0.35

    # Colors
    qwen_color = '#2E86AB'  # Blue
    ds_color = '#F18F01'    # Orange

    # Bars
    bars1 = ax.bar(
        x - width/2,
        qwen_subset,
        width,
        yerr=qwen_err_subset,
        label='qwen3-8b',
        color=qwen_color,
        edgecolor='black',
        linewidth=1.5,
        capsize=5,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )

    bars2 = ax.bar(
        x + width/2,
        ds_subset,
        width,
        yerr=ds_err_subset,
        label='deepseek-qwen3-8b',
        color=ds_color,
        edgecolor='black',
        linewidth=1.5,
        capsize=5,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )

    # Styling
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('DLA Score (logit contribution)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Direct Logit Attribution: Model Comparison',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_xticks(x)
    ax.set_xticklabels(key_layers)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='upper left',
        frameon=True,
        fontsize=11,
        edgecolor='black',
        framealpha=1.0
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved DLA comparison bar plot to: {output_path}")

    return fig, ax


def create_gradient_comparison_line(
    base_dir: Path,
    output_path: str = None
):
    """
    Create line graph comparing gradient sensitivity between models.
    """
    layers, qwen_mean, qwen_stderr, ds_mean, ds_stderr = aggregate_gradient_by_layer(base_dir)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Colors
    qwen_color = '#2E86AB'
    ds_color = '#F18F01'

    # Lines with markers
    ax.plot(
        layers,
        qwen_mean,
        marker='o',
        markersize=10,
        linewidth=3,
        color=qwen_color,
        markeredgecolor='black',
        markeredgewidth=1.5,
        label='qwen3-8b',
        zorder=3
    )

    ax.plot(
        layers,
        ds_mean,
        marker='s',
        markersize=10,
        linewidth=3,
        color=ds_color,
        markeredgecolor='black',
        markeredgewidth=1.5,
        label='deepseek-qwen3-8b',
        zorder=3
    )

    # Shaded error regions
    ax.fill_between(
        layers,
        [m - s for m, s in zip(qwen_mean, qwen_stderr)],
        [m + s for m, s in zip(qwen_mean, qwen_stderr)],
        alpha=0.2,
        color=qwen_color,
        zorder=2
    )

    ax.fill_between(
        layers,
        [m - s for m, s in zip(ds_mean, ds_stderr)],
        [m + s for m, s in zip(ds_mean, ds_stderr)],
        alpha=0.2,
        color=ds_color,
        zorder=2
    )

    # Styling
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gradient Norm (L2)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Gradient Sensitivity: Model Comparison',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_xticks(layers)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='upper right',
        frameon=True,
        fontsize=11,
        edgecolor='black',
        framealpha=1.0
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved gradient comparison line plot to: {output_path}")

    return fig, ax


def create_patching_comparison_bar(
    base_dir: Path,
    output_path: str = None
):
    """
    Create bar graph comparing patching effects between models.
    Shows only key layers.
    """
    layers, qwen_mean, qwen_stderr, ds_mean, ds_stderr = aggregate_patching_by_layer(base_dir)

    # Select key layers (every 4th)
    key_layers = list(range(0, max(layers) + 1, 4))
    key_indices = [layers.index(l) for l in key_layers if l in layers]

    qwen_subset = [qwen_mean[i] for i in key_indices]
    qwen_err_subset = [qwen_stderr[i] for i in key_indices]
    ds_subset = [ds_mean[i] for i in key_indices]
    ds_err_subset = [ds_stderr[i] for i in key_indices]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Bar positions
    x = np.arange(len(key_layers))
    width = 0.35

    # Colors
    qwen_color = '#2E86AB'
    ds_color = '#F18F01'

    # Bars
    bars1 = ax.bar(
        x - width/2,
        qwen_subset,
        width,
        yerr=qwen_err_subset,
        label='qwen3-8b',
        color=qwen_color,
        edgecolor='black',
        linewidth=1.5,
        capsize=5,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )

    bars2 = ax.bar(
        x + width/2,
        ds_subset,
        width,
        yerr=ds_err_subset,
        label='deepseek-qwen3-8b',
        color=ds_color,
        edgecolor='black',
        linewidth=1.5,
        capsize=5,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'},
        zorder=3
    )

    # Styling
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel('Patching Effect (logit difference)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Activation Patching: Model Comparison',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_xticks(x)
    ax.set_xticklabels(key_layers)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='upper left',
        frameon=True,
        fontsize=11,
        edgecolor='black',
        framealpha=1.0
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved patching comparison bar plot to: {output_path}")

    return fig, ax


def create_comprehensive_comparison(
    base_dir: Path,
    qwen_data: List[Dict],
    deepseek_data: List[Dict],
    output_path: str = None
):
    """
    Create 3-panel comprehensive comparison.
    """
    # Get all data
    dla_layers, qwen_dla, qwen_dla_err, ds_dla, ds_dla_err = aggregate_dla_by_layer(
        qwen_data, deepseek_data
    )
    grad_layers, qwen_grad, qwen_grad_err, ds_grad, ds_grad_err = aggregate_gradient_by_layer(base_dir)
    patch_layers, qwen_patch, qwen_patch_err, ds_patch, ds_patch_err = aggregate_patching_by_layer(base_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Colors
    qwen_color = '#2E86AB'
    ds_color = '#F18F01'

    # --- Panel 1: DLA ---
    ax = axes[0]
    # Select key layers
    key_layers_dla = list(range(0, max(dla_layers) + 1, 4))
    key_idx = [dla_layers.index(l) for l in key_layers_dla if l in dla_layers]
    x = np.arange(len(key_layers_dla))
    width = 0.35

    ax.bar(
        x - width/2,
        [qwen_dla[i] for i in key_idx],
        width,
        yerr=[qwen_dla_err[i] for i in key_idx],
        label='qwen3-8b',
        color=qwen_color,
        edgecolor='black',
        linewidth=1.2,
        capsize=4,
        error_kw={'linewidth': 1.2, 'ecolor': 'black'},
        zorder=3
    )
    ax.bar(
        x + width/2,
        [ds_dla[i] for i in key_idx],
        width,
        yerr=[ds_dla_err[i] for i in key_idx],
        label='deepseek-qwen3-8b',
        color=ds_color,
        edgecolor='black',
        linewidth=1.2,
        capsize=4,
        error_kw={'linewidth': 1.2, 'ecolor': 'black'},
        zorder=3
    )
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('DLA Score', fontsize=12, fontweight='bold')
    ax.set_title('Direct Logit Attribution', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(key_layers_dla)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 2: Gradient (line) ---
    ax = axes[1]
    ax.plot(
        grad_layers, qwen_grad,
        marker='o', markersize=8, linewidth=2.5,
        color=qwen_color, markeredgecolor='black', markeredgewidth=1.2,
        label='qwen3-8b', zorder=3
    )
    ax.plot(
        grad_layers, ds_grad,
        marker='s', markersize=8, linewidth=2.5,
        color=ds_color, markeredgecolor='black', markeredgewidth=1.2,
        label='deepseek-qwen3-8b', zorder=3
    )
    ax.fill_between(
        grad_layers,
        [m - s for m, s in zip(qwen_grad, qwen_grad_err)],
        [m + s for m, s in zip(qwen_grad, qwen_grad_err)],
        alpha=0.2, color=qwen_color, zorder=2
    )
    ax.fill_between(
        grad_layers,
        [m - s for m, s in zip(ds_grad, ds_grad_err)],
        [m + s for m, s in zip(ds_grad, ds_grad_err)],
        alpha=0.2, color=ds_color, zorder=2
    )
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Sensitivity', fontsize=13, fontweight='bold')
    ax.set_xticks(grad_layers)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 3: Patching ---
    ax = axes[2]
    key_layers_patch = list(range(0, max(patch_layers) + 1, 4))
    key_idx_p = [patch_layers.index(l) for l in key_layers_patch if l in patch_layers]
    x = np.arange(len(key_layers_patch))

    ax.bar(
        x - width/2,
        [qwen_patch[i] for i in key_idx_p],
        width,
        yerr=[qwen_patch_err[i] for i in key_idx_p],
        label='qwen3-8b',
        color=qwen_color,
        edgecolor='black',
        linewidth=1.2,
        capsize=4,
        error_kw={'linewidth': 1.2, 'ecolor': 'black'},
        zorder=3
    )
    ax.bar(
        x + width/2,
        [ds_patch[i] for i in key_idx_p],
        width,
        yerr=[ds_patch_err[i] for i in key_idx_p],
        label='deepseek-qwen3-8b',
        color=ds_color,
        edgecolor='black',
        linewidth=1.2,
        capsize=4,
        error_kw={'linewidth': 1.2, 'ecolor': 'black'},
        zorder=3
    )
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Patching Effect', fontsize=12, fontweight='bold')
    ax.set_title('Activation Patching', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(key_layers_patch)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fontsize=11,
        edgecolor='black',
        framealpha=1.0
    )

    # Overall title
    fig.suptitle(
        'Mechanistic Interpretability: Model Comparison',
        fontsize=16,
        fontweight='bold',
        y=1.08
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comprehensive comparison to: {output_path}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Plot random baseline comparison between qwen and deepseek"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='mech_interp/random_baseline_results',
        help='Path to random baseline results directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/plots',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading random baseline data...")
    qwen_data, deepseek_data = load_data(base_dir)
    print(f"Loaded qwen: {len(qwen_data)} entries")
    print(f"Loaded deepseek: {len(deepseek_data)} entries")
    print()

    print("Creating publication-quality comparison plots...")
    print()

    # Individual plots
    create_dla_comparison_bar(
        qwen_data,
        deepseek_data,
        output_dir / 'random_baseline_dla_comparison.png'
    )

    create_gradient_comparison_line(
        base_dir,
        output_dir / 'random_baseline_gradient_comparison.png'
    )

    create_patching_comparison_bar(
        base_dir,
        output_dir / 'random_baseline_patching_comparison.png'
    )

    # Comprehensive panel
    create_comprehensive_comparison(
        base_dir,
        qwen_data,
        deepseek_data,
        output_dir / 'random_baseline_comprehensive_comparison.png'
    )

    print()
    print("Done!")
    print(f"\nView plots in: {output_dir}")


if __name__ == '__main__':
    main()
