"""
Plot pass@5 performance across temperature sweeps for each difficulty level.

Creates 2 PNG files (one per model) with 5 side-by-side line graphs showing
pass@5 performance at each of the 5 MATH difficulty levels across 3 temperatures.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import argparse


def extract_sweep_data(sweep_dir: Path) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Extract pass@5 data from temperature sweep results.

    Returns:
        Dict mapping model_name -> temperature -> level -> pass@5
    """
    results = defaultdict(lambda: defaultdict(dict))

    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue

        log_file = run_dir / 'log.jsonl'
        if not log_file.exists():
            continue

        # Parse run name to get model and temperature
        parts = run_dir.name.split('_temp')
        if len(parts) != 2:
            continue

        model_name = parts[0]
        temp_str = parts[1].split('_')[0]

        # Read the last summary entry
        with open(log_file) as f:
            lines = f.readlines()
            for line in reversed(lines):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'summary':
                            # Extract per-level metrics
                            per_level = data.get('per_level_metrics', {})
                            for level_key, metrics in per_level.items():
                                level = int(level_key.split('_')[1])
                                pass_at_5 = metrics.get('pass@5', 0)
                                results[model_name][temp_str][level] = pass_at_5
                            break
                    except json.JSONDecodeError:
                        continue

    return results


def create_5_panel_line_plot(
    model_name: str,
    data: Dict[str, Dict[int, float]],
    output_path: str = None
):
    """
    Create 5 side-by-side bar graphs for each difficulty level.

    Args:
        model_name: Name of the model
        data: Dict mapping temperature -> level -> pass@5
        output_path: Where to save the plot
    """
    # Extract data
    temperatures = sorted([float(t) for t in data.keys()])
    levels = [1, 2, 3, 4, 5]

    # Create figure with 5 subplots side by side
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

    # Beautiful color palette - viridis-like gradient for temperatures
    temp_colors = {
        0.3: '#440154',  # Dark purple
        0.6: '#31688e',  # Teal blue
        0.9: '#35b779'   # Green
    }

    for i, level in enumerate(levels):
        ax = axes[i]

        # Extract pass@5 values for this level across temperatures
        pass_at_5_values = [data[str(temp)][level] for temp in temperatures]
        colors = [temp_colors[temp] for temp in temperatures]

        # Bar positions
        x_pos = np.arange(len(temperatures))
        width = 0.6

        # Bar plot with black edges
        bars = ax.bar(
            x_pos,
            pass_at_5_values,
            width,
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            zorder=3
        )

        # Styling
        ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Pass@5', fontsize=12, fontweight='bold')
        ax.set_title(f'Level {level}', fontsize=13, fontweight='bold', pad=10)

        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{t:.1f}' for t in temperatures])

        # Set y-axis to 0-1 range
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

        # Grid - only horizontal
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.tick_params(axis='both', labelsize=11)

    # Add legend at the top
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=temp_colors[0.3], edgecolor='black', linewidth=1.5, label='Temp=0.3'),
        Patch(facecolor=temp_colors[0.6], edgecolor='black', linewidth=1.5, label='Temp=0.6'),
        Patch(facecolor=temp_colors[0.9], edgecolor='black', linewidth=1.5, label='Temp=0.9')
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
        fontsize=12,
        edgecolor='black',
        framealpha=1.0
    )

    # Overall title
    model_display = model_name.replace('-', ' ').replace('_', '-')
    fig.suptitle(
        f'Pass@5 Performance Across Temperatures: {model_display}',
        fontsize=16,
        fontweight='bold',
        y=1.08
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {model_name} plot to: {output_path}")

    return fig, axes


def create_5_panel_multiline_plot(
    model_name: str,
    data: Dict[str, Dict[int, float]],
    output_path: str = None
):
    """
    Create 5 side-by-side bar graphs, one per temperature showing all levels.
    Alternative visualization style.
    """
    temperatures = sorted([float(t) for t in data.keys()])
    levels = [1, 2, 3, 4, 5]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Beautiful gradient colors for levels
    level_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(levels)))

    for i, temp in enumerate(temperatures):
        ax = axes[i]

        # Extract pass@5 values across all levels for this temperature
        pass_at_5_values = [data[str(temp)][level] for level in levels]

        # Bar positions
        x_pos = np.arange(len(levels))
        width = 0.7

        # Bar plot
        bars = ax.bar(
            x_pos,
            pass_at_5_values,
            width,
            color=level_colors,
            edgecolor='black',
            linewidth=1.5,
            zorder=3
        )

        # Styling
        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Pass@5', fontsize=12, fontweight='bold')
        ax.set_title(f'Temperature = {temp}', fontsize=13, fontweight='bold', pad=10)

        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(l) for l in levels])

        # Set y-axis
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

        # Grid - only horizontal
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.tick_params(axis='both', labelsize=11)

    # Overall title
    model_display = model_name.replace('-', ' ').replace('_', '-')
    fig.suptitle(
        f'Pass@5 Performance by Difficulty Level: {model_display}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {model_name} alternative plot to: {output_path}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Plot pass@5 performance across temperatures by difficulty level"
    )
    parser.add_argument(
        '--sweep-dir',
        type=str,
        default='results/sweeps/temperature_sweep',
        help='Path to temperature sweep directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--style',
        type=str,
        choices=['by-level', 'by-temp', 'both'],
        default='both',
        help='Plot style: by-level (temp on x-axis) or by-temp (level on x-axis)'
    )

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting temperature sweep data...")
    data = extract_sweep_data(sweep_dir)

    print(f"Found data for {len(data)} models")
    print()

    for model_name in sorted(data.keys()):
        print(f"Processing {model_name}...")

        if args.style in ['by-level', 'both']:
            # Style 1: Each panel shows one level, x-axis is temperature
            output_path = output_dir / f'pass_at_5_{model_name}_by_level.png'
            create_5_panel_line_plot(model_name, data[model_name], output_path)

        if args.style in ['by-temp', 'both']:
            # Style 2: Each panel shows one temperature, x-axis is level
            output_path = output_dir / f'pass_at_5_{model_name}_by_temp.png'
            create_5_panel_multiline_plot(model_name, data[model_name], output_path)

        print()

    print("Done!")
    print(f"\nView plots in: {output_dir}")


if __name__ == '__main__':
    main()
