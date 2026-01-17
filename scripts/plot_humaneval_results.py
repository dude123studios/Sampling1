"""
Plot HumanEval pass@k results.

Creates publication-quality visualizations of pass@k metrics across models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse


def load_humaneval_results(sweep_dir: Path) -> Dict[str, Dict]:
    """
    Load HumanEval results from sweep directory.

    Returns:
        Dict mapping model_name -> {pass@k metrics}
    """
    results = {}

    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue

        log_file = run_dir / 'log.jsonl'
        if not log_file.exists():
            continue

        # Extract model name from directory
        # Format: {model}_temp{x}_n{y}_humaneval_{timestamp}
        model_name = None
        if 'deepseek' in run_dir.name:
            model_name = 'deepseek-qwen3-8b'
        elif 'qwen3-8b' in run_dir.name:
            model_name = 'qwen3-8b'
        else:
            continue

        # Read summary from log
        with open(log_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'summary':
                        # Extract pass@k metrics
                        metrics = {}
                        for key, val in entry.items():
                            if key.startswith('pass@'):
                                k = int(key.split('@')[1])
                                metrics[k] = val

                        results[model_name] = {
                            'pass_at_k': metrics,
                            'total_problems': entry.get('total_problems', 0),
                            'total_samples': entry.get('total_samples', 0),
                            'overall_accuracy': entry.get('overall_accuracy', 0.0)
                        }
                        break
                except json.JSONDecodeError:
                    continue

    return results


def create_pass_at_k_comparison(results: Dict, output_path: str = None):
    """
    Create bar graph comparing pass@k across models.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Colors
    colors_map = {
        'qwen3-8b': '#2E86AB',
        'deepseek-qwen3-8b': '#F18F01'
    }

    # Get all k values (should be consistent across models)
    all_k_values = set()
    for model_data in results.values():
        all_k_values.update(model_data['pass_at_k'].keys())
    k_values = sorted(all_k_values)

    # Prepare data
    models = sorted(results.keys())
    x = np.arange(len(k_values))
    width = 0.35

    # Plot bars for each model
    for i, model in enumerate(models):
        pass_k_vals = [results[model]['pass_at_k'].get(k, 0.0) for k in k_values]
        offset = (i - 0.5) * width if len(models) == 2 else i * width - width

        bars = ax.bar(
            x + offset,
            pass_k_vals,
            width,
            label=model,
            color=colors_map.get(model, '#888888'),
            edgecolor='black',
            linewidth=1.5,
            zorder=3
        )

        # Add value labels on top of bars
        for bar, val in zip(bars, pass_k_vals):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    # Styling
    ax.set_xlabel('k (Number of Attempts)', fontsize=14, fontweight='bold')
    ax.set_ylabel('pass@k Rate', fontsize=14, fontweight='bold')
    ax.set_title(
        'HumanEval Code Generation: pass@k Performance',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}' for k in k_values])

    ax.set_ylim(0, min(1.0, max(max(results[m]['pass_at_k'].values()) for m in models) * 1.15))

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

    # Subtitle with sample info
    n_samples = list(results.values())[0]['total_samples'] // list(results.values())[0]['total_problems']
    subtitle = f'164 problems, {n_samples} samples per problem (temp=0.6, top_k=50, top_p=0.9)'
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
        print(f"Saved pass@k comparison to: {output_path}")

    return fig, ax


def create_pass_at_k_line_plot(results: Dict, output_path: str = None):
    """
    Create line plot showing pass@k trends.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Colors
    colors_map = {
        'qwen3-8b': '#2E86AB',
        'deepseek-qwen3-8b': '#F18F01'
    }

    # Get all k values
    all_k_values = set()
    for model_data in results.values():
        all_k_values.update(model_data['pass_at_k'].keys())
    k_values = sorted(all_k_values)

    # Plot lines for each model
    models = sorted(results.keys())
    for model in models:
        pass_k_vals = [results[model]['pass_at_k'].get(k, 0.0) for k in k_values]

        ax.plot(
            k_values,
            pass_k_vals,
            marker='o',
            markersize=8,
            linewidth=2.5,
            color=colors_map.get(model, '#888888'),
            markeredgecolor='black',
            markeredgewidth=1.2,
            label=model,
            zorder=3
        )

    # Styling
    ax.set_xlabel('k (Number of Attempts)', fontsize=14, fontweight='bold')
    ax.set_ylabel('pass@k Rate', fontsize=14, fontweight='bold')
    ax.set_title(
        'HumanEval: pass@k Scaling',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    ax.set_ylim(0, 1.0)

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(axis='both', labelsize=11)

    # Legend
    ax.legend(
        loc='lower right',
        frameon=True,
        fontsize=12,
        edgecolor='black',
        framealpha=1.0,
        fancybox=False
    )

    # Subtitle
    subtitle = 'Higher k = more attempts allowed per problem'
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
        print(f"Saved pass@k line plot to: {output_path}")

    return fig, ax


def print_summary_table(results: Dict):
    """Print a nicely formatted summary table."""
    print("\n" + "="*80)
    print("HumanEval Results Summary")
    print("="*80)

    for model, data in sorted(results.items()):
        print(f"\nModel: {model}")
        print(f"  Total Problems: {data['total_problems']}")
        print(f"  Total Samples: {data['total_samples']}")
        print(f"  Overall Accuracy: {data['overall_accuracy']:.4f}")
        print(f"  Pass@k Metrics:")

        for k, val in sorted(data['pass_at_k'].items()):
            print(f"    pass@{k}: {val:.4f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Plot HumanEval pass@k results"
    )
    parser.add_argument(
        '--sweep-dir',
        type=str,
        default='results/sweeps/humaneval_sweep',
        help='Path to HumanEval sweep directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/plots',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    print(f"Loading HumanEval results from: {sweep_dir}")
    results = load_humaneval_results(sweep_dir)

    if not results:
        print("No results found!")
        return

    print(f"Loaded results for {len(results)} models")

    # Print summary table
    print_summary_table(results)

    # Create plots
    print("\nGenerating plots...")

    # Bar comparison
    create_pass_at_k_comparison(
        results,
        output_dir / 'humaneval_pass_at_k_bars.png'
    )

    # Line plot
    create_pass_at_k_line_plot(
        results,
        output_dir / 'humaneval_pass_at_k_lines.png'
    )

    print(f"\nDone! Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
