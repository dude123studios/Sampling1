"""
High-quality plotting for prefix experiments

Creates ICML-quality plots showing:
1. Accuracy vs Prefix Length
2. Marginal Gain per Token (showing first 32 tokens matter most)
3. Comparison across models

Usage:
    python scripts/plot_prefix_experiments.py --oracle results/oracle_prefix/oracle_prefix_results_*.json
    python scripts/plot_prefix_experiments.py --self-correct results/self_correct_prefix/self_correct_prefix_results_*.json
    python scripts/plot_prefix_experiments.py --both oracle.json self_correct.json --output figures/
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import seaborn as sns

# Configure matplotlib for high-quality output
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14

# Use a professional color palette
COLORS = sns.color_palette("Set2", 8)

class PrefixExperimentPlotter:
    def __init__(self, output_dir="figures"):
        """Initialize the plotter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, results_file):
        """Load experiment results from JSON."""
        with open(results_file, 'r') as f:
            return json.load(f)

    def extract_data(self, results):
        """Extract prefix lengths and accuracies for each model."""
        data = {}

        for model_name, model_data in results['results_per_model'].items():
            prefix_lengths = []
            accuracies = []

            for prefix_length in sorted(model_data['results_per_prefix'].keys()):
                prefix_data = model_data['results_per_prefix'][prefix_length]
                prefix_lengths.append(prefix_length)
                accuracies.append(prefix_data['accuracy'] * 100)  # Convert to percentage

            data[model_name] = {
                'prefix_lengths': np.array(prefix_lengths),
                'accuracies': np.array(accuracies)
            }

        return data

    def calculate_marginal_gain(self, prefix_lengths, accuracies):
        """Calculate marginal gain per token between consecutive points."""
        gains = []
        token_intervals = []

        for i in range(1, len(prefix_lengths)):
            token_diff = prefix_lengths[i] - prefix_lengths[i-1]
            acc_diff = accuracies[i] - accuracies[i-1]

            if token_diff > 0:
                gain_per_token = acc_diff / token_diff
                gains.append(gain_per_token)
                # Use midpoint for plotting
                token_intervals.append((prefix_lengths[i-1] + prefix_lengths[i]) / 2)

        return np.array(token_intervals), np.array(gains)

    def plot_accuracy_vs_prefix(self, data, title, output_file):
        """Plot accuracy vs prefix length."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for idx, (model_name, model_data) in enumerate(data.items()):
            prefix_lengths = model_data['prefix_lengths']
            accuracies = model_data['accuracies']

            ax.plot(prefix_lengths, accuracies,
                   marker='o', linewidth=2, markersize=7,
                   label=model_name, color=COLORS[idx],
                   markeredgewidth=1.5, markeredgecolor='white')

        ax.set_xlabel('Prefix Length (tokens)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(frameon=True, fancybox=True, shadow=True)

        # Add reference line at 32 tokens
        ax.axvline(x=32, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='32 tokens')

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def plot_marginal_gain(self, data, title, output_file):
        """Plot marginal gain per token."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for idx, (model_name, model_data) in enumerate(data.items()):
            prefix_lengths = model_data['prefix_lengths']
            accuracies = model_data['accuracies']

            token_intervals, gains = self.calculate_marginal_gain(prefix_lengths, accuracies)

            ax.plot(token_intervals, gains,
                   marker='s', linewidth=2, markersize=7,
                   label=model_name, color=COLORS[idx],
                   markeredgewidth=1.5, markeredgecolor='white')

        ax.set_xlabel('Prefix Length (tokens)', fontweight='bold')
        ax.set_ylabel('Marginal Gain (% per token)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(frameon=True, fancybox=True, shadow=True)

        # Add reference line at 32 tokens
        ax.axvline(x=32, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def plot_cumulative_gain(self, data, title, output_file):
        """Plot cumulative gain showing diminishing returns."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for idx, (model_name, model_data) in enumerate(data.items()):
            prefix_lengths = model_data['prefix_lengths']
            accuracies = model_data['accuracies']

            # Calculate gain from baseline (0 tokens)
            baseline_acc = accuracies[0]
            cumulative_gains = accuracies - baseline_acc

            ax.plot(prefix_lengths, cumulative_gains,
                   marker='D', linewidth=2, markersize=7,
                   label=model_name, color=COLORS[idx],
                   markeredgewidth=1.5, markeredgecolor='white')

        ax.set_xlabel('Prefix Length (tokens)', fontweight='bold')
        ax.set_ylabel('Accuracy Gain from Baseline (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(frameon=True, fancybox=True, shadow=True)

        # Highlight the 0-32 token region
        ax.axvspan(0, 32, alpha=0.2, color='green', label='First 32 tokens')

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def plot_comparison(self, oracle_data, self_correct_data, output_file):
        """Plot comparison between oracle and self-correct experiments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot oracle on left
        for idx, (model_name, model_data) in enumerate(oracle_data.items()):
            ax1.plot(model_data['prefix_lengths'], model_data['accuracies'],
                    marker='o', linewidth=2, markersize=7,
                    label=model_name, color=COLORS[idx],
                    markeredgewidth=1.5, markeredgecolor='white')

        ax1.set_xlabel('Prefix Length (tokens)', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Oracle Prefix Experiment', fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.axvline(x=32, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

        # Plot self-correct on right
        for idx, (model_name, model_data) in enumerate(self_correct_data.items()):
            ax2.plot(model_data['prefix_lengths'], model_data['accuracies'],
                    marker='s', linewidth=2, markersize=7,
                    label=model_name, color=COLORS[idx],
                    markeredgewidth=1.5, markeredgecolor='white')

        ax2.set_xlabel('Prefix Length (tokens)', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Self-Correct Prefix Experiment', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.axvline(x=32, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def create_summary_table(self, data, output_file):
        """Create a summary table showing gains."""
        print("\n" + "="*80)
        print("SUMMARY TABLE: Accuracy Gain Analysis")
        print("="*80)

        for model_name, model_data in data.items():
            prefix_lengths = model_data['prefix_lengths']
            accuracies = model_data['accuracies']

            print(f"\nModel: {model_name}")
            print("-" * 60)
            print(f"{'Prefix':<12} {'Accuracy':<12} {'Gain from 0':<15} {'Marginal Gain':<15}")
            print("-" * 60)

            baseline_acc = accuracies[0]

            for i, (length, acc) in enumerate(zip(prefix_lengths, accuracies)):
                gain_from_baseline = acc - baseline_acc

                if i > 0:
                    prev_length = prefix_lengths[i-1]
                    prev_acc = accuracies[i-1]
                    token_diff = length - prev_length
                    acc_diff = acc - prev_acc
                    marginal = acc_diff / token_diff if token_diff > 0 else 0
                    marginal_str = f"{marginal:.4f}% / tok"
                else:
                    marginal_str = "N/A"

                print(f"{length:<12} {acc:>6.2f}%{'':<4} {gain_from_baseline:>6.2f}%{'':<7} {marginal_str:<15}")

        print("="*80 + "\n")

    def plot_oracle_experiment(self, results_file):
        """Generate all plots for oracle experiment."""
        print(f"\nPlotting Oracle Prefix Experiment: {results_file}")
        results = self.load_results(results_file)
        data = self.extract_data(results)

        self.plot_accuracy_vs_prefix(
            data,
            "Oracle Prefix: Accuracy vs Prefix Length",
            "oracle_accuracy_vs_prefix.pdf"
        )

        self.plot_marginal_gain(
            data,
            "Oracle Prefix: Marginal Gain per Token",
            "oracle_marginal_gain.pdf"
        )

        self.plot_cumulative_gain(
            data,
            "Oracle Prefix: Cumulative Accuracy Gain",
            "oracle_cumulative_gain.pdf"
        )

        self.create_summary_table(data, "oracle_summary.txt")

    def plot_self_correct_experiment(self, results_file):
        """Generate all plots for self-correct experiment."""
        print(f"\nPlotting Self-Correct Prefix Experiment: {results_file}")
        results = self.load_results(results_file)
        data = self.extract_data(results)

        self.plot_accuracy_vs_prefix(
            data,
            "Self-Correct Prefix: Accuracy vs Prefix Length",
            "self_correct_accuracy_vs_prefix.pdf"
        )

        self.plot_marginal_gain(
            data,
            "Self-Correct Prefix: Marginal Gain per Token",
            "self_correct_marginal_gain.pdf"
        )

        self.plot_cumulative_gain(
            data,
            "Self-Correct Prefix: Cumulative Accuracy Gain",
            "self_correct_cumulative_gain.pdf"
        )

        self.create_summary_table(data, "self_correct_summary.txt")

    def plot_both_experiments(self, oracle_file, self_correct_file):
        """Generate comparison plots for both experiments."""
        print(f"\nPlotting Comparison: Oracle vs Self-Correct")

        oracle_results = self.load_results(oracle_file)
        self_correct_results = self.load_results(self_correct_file)

        oracle_data = self.extract_data(oracle_results)
        self_correct_data = self.extract_data(self_correct_results)

        self.plot_comparison(
            oracle_data,
            self_correct_data,
            "comparison_oracle_vs_self_correct.pdf"
        )

def main():
    parser = argparse.ArgumentParser(description="Plot prefix experiment results")
    parser.add_argument("--oracle", type=str, default=None,
                        help="Path to oracle prefix experiment results JSON")
    parser.add_argument("--self-correct", type=str, default=None,
                        help="Path to self-correct prefix experiment results JSON")
    parser.add_argument("--both", nargs=2, metavar=('ORACLE', 'SELF_CORRECT'),
                        help="Paths to both experiment results for comparison")
    parser.add_argument("--output", type=str, default="figures",
                        help="Output directory for plots")

    args = parser.parse_args()

    plotter = PrefixExperimentPlotter(output_dir=args.output)

    if args.both:
        oracle_file, self_correct_file = args.both
        plotter.plot_oracle_experiment(oracle_file)
        plotter.plot_self_correct_experiment(self_correct_file)
        plotter.plot_both_experiments(oracle_file, self_correct_file)
    else:
        if args.oracle:
            plotter.plot_oracle_experiment(args.oracle)

        if args.self_correct:
            plotter.plot_self_correct_experiment(args.self_correct)

    if not args.oracle and not args.self_correct and not args.both:
        print("Error: Please specify at least one of --oracle, --self-correct, or --both")
        parser.print_help()

if __name__ == "__main__":
    main()
