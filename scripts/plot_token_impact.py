"""
Plot token impact analysis results with publication-quality styling.

Creates visualizations showing:
1. Divergence scores across cutoff positions
2. Layer-wise similarity patterns
3. Distribution of impactful positions
4. Per-problem divergence heatmaps
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


def load_token_impact_results(filepath: str) -> List[Dict]:
    """Load token impact results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_impactful_positions(filepath: str) -> Dict[str, List[int]]:
    """Load impactful positions from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_by_cutoff(results: List[Dict]) -> Tuple[List[int], List[float], List[float]]:
    """
    Aggregate divergence scores by cutoff position.

    Returns:
        cutoff_positions: List of cutoff positions
        mean_divergence: Mean divergence for each position
        std_divergence: Standard deviation for each position
    """
    # Group by cutoff position
    cutoff_data = defaultdict(list)
    for result in results:
        cutoff_data[result['cutoff_position']].append(result['divergence_score'])

    cutoff_positions = sorted(cutoff_data.keys())
    mean_divergence = [np.mean(cutoff_data[pos]) for pos in cutoff_positions]
    std_divergence = [np.std(cutoff_data[pos]) for pos in cutoff_positions]

    return cutoff_positions, mean_divergence, std_divergence


def aggregate_layer_similarities(results: List[Dict]) -> Dict[int, Dict[int, List[float]]]:
    """
    Aggregate layer similarities by cutoff position.

    Returns:
        Dict mapping layer -> cutoff_position -> list of similarities
    """
    layer_data = defaultdict(lambda: defaultdict(list))

    for result in results:
        cutoff = result['cutoff_position']
        for layer_str, similarity in result['layer_similarities'].items():
            layer = int(layer_str)
            layer_data[layer][cutoff].append(similarity)

    return layer_data


def create_divergence_line_plot(
    results: List[Dict],
    output_path: str = None
):
    """
    Create line plot of divergence scores across cutoff positions.

    Shows mean divergence with error bars (stderr).
    """
    cutoff_positions, mean_div, std_div = aggregate_by_cutoff(results)

    # Calculate stderr
    n_problems = len(set(r['problem_id'] for r in results))
    stderr_div = [std / np.sqrt(n_problems) for std in std_div]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Publication-quality line with markers and error bars
    ax.errorbar(
        cutoff_positions,
        mean_div,
        yerr=stderr_div,
        marker='o',
        markersize=8,
        linewidth=2.5,
        capsize=6,
        capthick=2,
        color='#2E86AB',  # Nice blue
        ecolor='black',
        elinewidth=1.5,
        markeredgecolor='black',
        markeredgewidth=1.2,
        label='Mean Divergence Score',
        zorder=3
    )

    # Styling
    ax.set_xlabel('Cutoff Position (tokens)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Divergence Score (1 - cosine similarity)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Token Impact Analysis: Divergence Across Cutoff Positions',
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

    # Set x-ticks to match cutoff positions
    ax.set_xticks(cutoff_positions)
    ax.tick_params(axis='both', labelsize=11)

    # Add subtitle with info
    subtitle = f'n = {n_problems} problems, {len(cutoff_positions)} cutoff positions'
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
        print(f"Saved divergence line plot to: {output_path}")

    return fig, ax


def create_layer_similarity_plot(
    results: List[Dict],
    output_path: str = None
):
    """
    Create line plots showing layer-wise similarities across cutoff positions.

    Each layer gets its own line.
    """
    layer_data = aggregate_layer_similarities(results)
    layers = sorted(layer_data.keys())

    # Calculate means and stderrs for each layer
    cutoff_positions = sorted(set(r['cutoff_position'] for r in results))
    n_problems = len(set(r['problem_id'] for r in results))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Sequential colormap for layers (viridis-like)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))

    for layer, color in zip(layers, colors):
        means = []
        stderrs = []

        for cutoff in cutoff_positions:
            sims = layer_data[layer][cutoff]
            means.append(np.mean(sims))
            stderrs.append(np.std(sims) / np.sqrt(len(sims)))

        ax.errorbar(
            cutoff_positions,
            means,
            yerr=stderrs,
            marker='o',
            markersize=6,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            color=color,
            ecolor='black',
            elinewidth=1,
            markeredgecolor='black',
            markeredgewidth=1,
            label=f'Layer {layer}',
            alpha=0.9,
            zorder=3
        )

    # Styling
    ax.set_xlabel('Cutoff Position (tokens)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=13, fontweight='bold')
    ax.set_title(
        'Layer-wise Activation Similarity: Top1 vs Top2 Token Paths',
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

    ax.set_xticks(cutoff_positions)
    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='upper right',
        frameon=True,
        fontsize=10,
        edgecolor='black',
        framealpha=1.0,
        ncol=1
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved layer similarity plot to: {output_path}")

    return fig, ax


def create_divergence_distribution(
    results: List[Dict],
    output_path: str = None
):
    """
    Create distribution plot of divergence scores.

    Shows histogram with KDE overlay.
    """
    divergence_scores = [r['divergence_score'] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Histogram with black edges
    n, bins, patches = ax.hist(
        divergence_scores,
        bins=30,
        density=True,
        color='#A23B72',  # Purple
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7,
        zorder=2
    )

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(divergence_scores)
    x_range = np.linspace(min(divergence_scores), max(divergence_scores), 200)
    ax.plot(
        x_range,
        kde(x_range),
        color='black',
        linewidth=2.5,
        label='KDE',
        zorder=3
    )

    # Add vertical line for mean
    mean_div = np.mean(divergence_scores)
    ax.axvline(
        mean_div,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean = {mean_div:.4f}',
        zorder=3
    )

    # Styling
    ax.set_xlabel('Divergence Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribution of Divergence Scores Across All Cutoff Positions',
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

    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='upper right',
        frameon=True,
        fontsize=10,
        edgecolor='black',
        framealpha=1.0
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved divergence distribution to: {output_path}")

    return fig, ax


def create_impactful_positions_plot(
    impactful_positions: Dict[str, List[int]],
    output_path: str = None
):
    """
    Create bar plot showing distribution of most impactful positions.

    Counts how often each position appears in top-3 most impactful.
    """
    # Count occurrences
    position_counts = defaultdict(int)
    for positions in impactful_positions.values():
        for pos in positions:
            position_counts[pos] += 1

    positions = sorted(position_counts.keys())
    counts = [position_counts[pos] for pos in positions]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Bar plot with gradient color based on count
    colors = plt.cm.YlOrRd(np.array(counts) / max(counts))

    bars = ax.bar(
        positions,
        counts,
        color=colors,
        edgecolor='black',
        linewidth=1.2,
        zorder=3
    )

    # Styling
    ax.set_xlabel('Cutoff Position (tokens)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency in Top-3 Most Impactful', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribution of High-Impact Cutoff Positions',
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

    ax.set_xticks(positions)
    ax.tick_params(axis='both', labelsize=11)

    # Add subtitle
    subtitle = f'Based on {len(impactful_positions)} problems (top 3 positions per problem)'
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
        print(f"Saved impactful positions plot to: {output_path}")

    return fig, ax


def create_comprehensive_panel(
    results: List[Dict],
    impactful_positions: Dict[str, List[int]],
    output_path: str = None
):
    """
    Create comprehensive 2x2 panel showing all key metrics.
    """
    cutoff_positions, mean_div, std_div = aggregate_by_cutoff(results)
    n_problems = len(set(r['problem_id'] for r in results))
    stderr_div = [std / np.sqrt(n_problems) for std in std_div]

    layer_data = aggregate_layer_similarities(results)
    layers = sorted(layer_data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Top Left: Divergence across cutoffs ---
    ax = axes[0, 0]
    ax.errorbar(
        cutoff_positions, mean_div, yerr=stderr_div,
        marker='o', markersize=8, linewidth=2.5,
        capsize=6, capthick=2,
        color='#2E86AB', ecolor='black', elinewidth=1.5,
        markeredgecolor='black', markeredgewidth=1.2,
        zorder=3
    )
    ax.set_xlabel('Cutoff Position (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Divergence Score', fontsize=12, fontweight='bold')
    ax.set_title('Mean Divergence Across Cutoff Positions', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(cutoff_positions)

    # --- Top Right: Layer similarities ---
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))
    for layer, color in zip(layers, colors):
        means = [np.mean(layer_data[layer][cutoff]) for cutoff in cutoff_positions]
        stderrs = [np.std(layer_data[layer][cutoff]) / np.sqrt(len(layer_data[layer][cutoff]))
                   for cutoff in cutoff_positions]
        ax.errorbar(
            cutoff_positions, means, yerr=stderrs,
            marker='o', markersize=6, linewidth=2,
            capsize=4, capthick=1.5,
            color=color, ecolor='black', elinewidth=1,
            markeredgecolor='black', markeredgewidth=1,
            label=f'Layer {layer}', alpha=0.9, zorder=3
        )
    ax.set_xlabel('Cutoff Position (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Similarities', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(cutoff_positions)
    ax.legend(loc='upper right', frameon=True, fontsize=9, edgecolor='black')

    # --- Bottom Left: Divergence distribution ---
    ax = axes[1, 0]
    divergence_scores = [r['divergence_score'] for r in results]
    ax.hist(
        divergence_scores, bins=30, density=True,
        color='#A23B72', edgecolor='black', linewidth=1.2,
        alpha=0.7, zorder=2
    )
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(divergence_scores)
    x_range = np.linspace(min(divergence_scores), max(divergence_scores), 200)
    ax.plot(x_range, kde(x_range), color='black', linewidth=2.5, zorder=3)
    mean_div_all = np.mean(divergence_scores)
    ax.axvline(mean_div_all, color='red', linestyle='--', linewidth=2, zorder=3)
    ax.set_xlabel('Divergence Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Divergence Scores', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Bottom Right: Impactful positions ---
    ax = axes[1, 1]
    position_counts = defaultdict(int)
    for positions in impactful_positions.values():
        for pos in positions:
            position_counts[pos] += 1
    positions = sorted(position_counts.keys())
    counts = [position_counts[pos] for pos in positions]
    colors_bar = plt.cm.YlOrRd(np.array(counts) / max(counts))
    ax.bar(
        positions, counts,
        color=colors_bar, edgecolor='black', linewidth=1.2, zorder=3
    )
    ax.set_xlabel('Cutoff Position (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency in Top-3', fontsize=12, fontweight='bold')
    ax.set_title('High-Impact Position Distribution', fontsize=13, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(positions)

    # Overall title
    fig.suptitle(
        'Token Impact Analysis: Comprehensive Overview',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comprehensive panel to: {output_path}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot token impact analysis results")
    parser.add_argument(
        '--results',
        type=str,
        default='mech_interp/token_impact_results/token_impact_results.json',
        help='Path to token impact results JSON'
    )
    parser.add_argument(
        '--impactful',
        type=str,
        default='mech_interp/token_impact_results/impactful_positions.json',
        help='Path to impactful positions JSON'
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

    print("Loading token impact results...")
    results = load_token_impact_results(args.results)
    impactful_positions = load_impactful_positions(args.impactful)

    print(f"Loaded {len(results)} results from {len(set(r['problem_id'] for r in results))} problems")
    print(f"Loaded {len(impactful_positions)} impactful position records")
    print()

    print("Creating publication-quality plots...")
    print()

    # Individual plots
    create_divergence_line_plot(
        results,
        output_dir / 'token_impact_divergence.png'
    )

    create_layer_similarity_plot(
        results,
        output_dir / 'token_impact_layer_similarities.png'
    )

    create_divergence_distribution(
        results,
        output_dir / 'token_impact_divergence_distribution.png'
    )

    create_impactful_positions_plot(
        impactful_positions,
        output_dir / 'token_impact_high_impact_positions.png'
    )

    # Comprehensive panel
    create_comprehensive_panel(
        results,
        impactful_positions,
        output_dir / 'token_impact_comprehensive.png'
    )

    print()
    print("Done!")
    print(f"\nView plots in: {output_dir}")


if __name__ == '__main__':
    main()
