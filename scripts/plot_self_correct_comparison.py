"""
Plot comparison of self-correct prefix experiments with publication-quality styling.

Compares results from two different seed models (deepseek-qwen3-8b vs qwen3-8b)
both self-correcting with qwen3-8b.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_accuracy_data(results: Dict) -> Tuple[List[int], List[float], List[float]]:
    """
    Extract accuracy data by prefix length.

    Returns:
        prefix_lengths: List of prefix lengths
        accuracies: Mean accuracy for each prefix length
        stderrs: Standard error for each prefix length
    """
    first_model = list(results['results_per_model'].keys())[0]
    model_results = results['results_per_model'][first_model]

    prefix_lengths = sorted([int(k) for k in model_results['results_per_prefix'].keys()])
    accuracies = []
    stderrs = []

    for prefix_len in prefix_lengths:
        prefix_str = str(prefix_len)
        prefix_data = model_results['results_per_prefix'][prefix_str]
        accuracy = prefix_data['accuracy']
        n_samples = len(prefix_data['details'])

        # Standard error: sqrt(p(1-p)/n)
        stderr = np.sqrt(accuracy * (1 - accuracy) / n_samples) if n_samples > 0 else 0

        accuracies.append(accuracy)
        stderrs.append(stderr)

    return prefix_lengths, accuracies, stderrs


def get_source_model_name(results: Dict) -> str:
    """Extract the source model name from config."""
    source_file = results['config']['source_results']
    if 'deepseek-qwen3-8b' in source_file:
        return 'deepseek-qwen3-8b'
    elif 'qwen3-8b' in source_file:
        return 'qwen3-8b'
    else:
        return 'unknown'


def create_comparison_plot(
    file1: str,
    file2: str,
    output_path: str = None
):
    """
    Create publication-quality comparison plot.

    Args:
        file1: Path to first results file (deepseek-qwen3-8b seeds)
        file2: Path to second results file (qwen3-8b seeds)
        output_path: Where to save the plot
    """
    # Load data
    results1 = load_results(file1)
    results2 = load_results(file2)

    prefix_lengths1, acc1, stderr1 = extract_accuracy_data(results1)
    prefix_lengths2, acc2, stderr2 = extract_accuracy_data(results2)

    source_model1 = get_source_model_name(results1)
    source_model2 = get_source_model_name(results2)

    n_problems1 = results1['num_candidates']
    n_problems2 = results2['num_candidates']

    # Verify same prefix lengths
    assert prefix_lengths1 == prefix_lengths2, "Prefix lengths must match"
    prefix_lengths = prefix_lengths1

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Set up x positions for grouped bars
    x = np.arange(len(prefix_lengths))
    width = 0.35

    # Publication-quality colors (colorblind-friendly)
    color1 = '#1f77b4'  # Blue
    color2 = '#ff7f0e'  # Orange
    edge_color = 'black'

    # Plot bars with black edges and error bars
    bars1 = ax.bar(
        x - width/2,
        acc1,
        width,
        yerr=stderr1,
        label=f'{source_model1} seeds (n={n_problems1})',
        color=color1,
        edgecolor=edge_color,
        linewidth=1.2,
        capsize=5,
        error_kw={'linewidth': 1.5, 'ecolor': edge_color},
        zorder=3
    )

    bars2 = ax.bar(
        x + width/2,
        acc2,
        width,
        yerr=stderr2,
        label=f'{source_model2} seeds (n={n_problems2})',
        color=color2,
        edgecolor=edge_color,
        linewidth=1.2,
        capsize=5,
        error_kw={'linewidth': 1.5, 'ecolor': edge_color},
        zorder=3
    )

    # Styling
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prefix Length (tokens)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Self-Correct Performance: Seed Model Comparison\n(Both self-correcting with qwen3-8b)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prefix_lengths], fontsize=11)

    # Y-axis
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='y', labelsize=11)

    # Grid - only horizontal, dashed, low alpha, behind bars
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legend - external, horizontal, at top
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fontsize=11,
        edgecolor='black',
        framealpha=1.0
    )

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # Tight layout
    plt.tight_layout()

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to: {output_path}")

    return fig, ax


def create_detailed_comparison(
    file1: str,
    file2: str,
    output_path: str = None
):
    """
    Create detailed comparison with multiple metrics.

    Shows:
    - Top left: Accuracy comparison
    - Top right: Improvement over baseline (prefix=0)
    - Bottom left: Number of problems correct
    - Bottom right: Relative improvement between models
    """
    # Load data
    results1 = load_results(file1)
    results2 = load_results(file2)

    prefix_lengths1, acc1, stderr1 = extract_accuracy_data(results1)
    prefix_lengths2, acc2, stderr2 = extract_accuracy_data(results2)

    source_model1 = get_source_model_name(results1)
    source_model2 = get_source_model_name(results2)

    n_problems1 = results1['num_candidates']
    n_problems2 = results2['num_candidates']

    prefix_lengths = prefix_lengths1

    # Calculate improvement over baseline
    baseline1 = acc1[0]  # prefix=0
    baseline2 = acc2[0]
    improvement1 = [(a - baseline1) for a in acc1]
    improvement2 = [(a - baseline2) for a in acc2]

    # Get number correct
    first_model1 = list(results1['results_per_model'].keys())[0]
    first_model2 = list(results2['results_per_model'].keys())[0]

    num_correct1 = [
        results1['results_per_model'][first_model1]['results_per_prefix'][str(p)]['num_correct']
        for p in prefix_lengths
    ]
    num_correct2 = [
        results2['results_per_model'][first_model2]['results_per_prefix'][str(p)]['num_correct']
        for p in prefix_lengths
    ]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors
    color1 = '#1f77b4'
    color2 = '#ff7f0e'
    edge_color = 'black'

    x = np.arange(len(prefix_lengths))
    width = 0.35

    # --- Top Left: Accuracy ---
    ax = axes[0, 0]
    ax.bar(
        x - width/2, acc1, width,
        yerr=stderr1,
        label=source_model1,
        color=color1, edgecolor=edge_color, linewidth=1.2,
        capsize=5, error_kw={'linewidth': 1.5, 'ecolor': edge_color},
        zorder=3
    )
    ax.bar(
        x + width/2, acc2, width,
        yerr=stderr2,
        label=source_model2,
        color=color2, edgecolor=edge_color, linewidth=1.2,
        capsize=5, error_kw={'linewidth': 1.5, 'ecolor': edge_color},
        zorder=3
    )
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prefix Length', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Prefix Length', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prefix_lengths])
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Top Right: Improvement over baseline ---
    ax = axes[0, 1]
    ax.bar(
        x - width/2, improvement1, width,
        label=source_model1,
        color=color1, edgecolor=edge_color, linewidth=1.2,
        zorder=3
    )
    ax.bar(
        x + width/2, improvement2, width,
        label=source_model2,
        color=color2, edgecolor=edge_color, linewidth=1.2,
        zorder=3
    )
    ax.set_ylabel('Accuracy Improvement', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prefix Length', fontsize=12, fontweight='bold')
    ax.set_title('Improvement over Baseline (Prefix=0)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prefix_lengths])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, zorder=1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Bottom Left: Number correct ---
    ax = axes[1, 0]
    ax.bar(
        x - width/2, num_correct1, width,
        label=f'{source_model1} (n={n_problems1})',
        color=color1, edgecolor=edge_color, linewidth=1.2,
        zorder=3
    )
    ax.bar(
        x + width/2, num_correct2, width,
        label=f'{source_model2} (n={n_problems2})',
        color=color2, edgecolor=edge_color, linewidth=1.2,
        zorder=3
    )
    ax.set_ylabel('Number Correct', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prefix Length', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Number of Correct Solutions', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prefix_lengths])
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Bottom Right: Relative difference ---
    ax = axes[1, 1]
    # Calculate difference (model2 - model1 in percentage points)
    acc_diff = [(a2 - a1) * 100 for a1, a2 in zip(acc1, acc2)]
    colors = [color2 if d >= 0 else color1 for d in acc_diff]

    bars = ax.bar(
        x, acc_diff, width * 2,
        color=colors, edgecolor=edge_color, linewidth=1.2,
        zorder=3
    )
    ax.set_ylabel('Accuracy Difference (pp)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prefix Length', fontsize=12, fontweight='bold')
    ax.set_title(f'{source_model2} - {source_model1} (percentage points)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prefix_lengths])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, zorder=1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add a legend at the very top of the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        fontsize=11,
        edgecolor='black',
        framealpha=1.0
    )

    # Overall title
    fig.suptitle(
        'Self-Correct Prefix Experiment: Seed Model Comparison\n(Both self-correcting with qwen3-8b)',
        fontsize=15,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved detailed plot to: {output_path}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot self-correct prefix comparison")
    parser.add_argument(
        '--file1',
        type=str,
        default='results/self_correct_prefix/self_correct_results_2026-01-14_11-45-49.json',
        help='First results file (deepseek-qwen3-8b seeds)'
    )
    parser.add_argument(
        '--file2',
        type=str,
        default='results/self_correct_prefix/self_correct_results_2026-01-14_17-11-44.json',
        help='Second results file (qwen3-8b seeds)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Create only the simple comparison plot'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating publication-quality comparison plots...")
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print()

    # Simple comparison
    simple_path = output_dir / 'self_correct_comparison_simple.png'
    create_comparison_plot(args.file1, args.file2, simple_path)
    print()

    if not args.simple:
        # Detailed comparison
        detailed_path = output_dir / 'self_correct_comparison_detailed.png'
        create_detailed_comparison(args.file1, args.file2, detailed_path)
        print()

    print("Done!")
    print(f"\nView plots in: {output_dir}")


if __name__ == '__main__':
    main()
