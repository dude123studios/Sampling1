"""
High-Quality Plotting for MATH-500 Results

Creates ICML-quality plots showing:
1. Performance by difficulty level
2. Temperature sweep comparisons
3. Model comparisons
4. Subject-wise performance

Usage:
    python scripts/plot_math_results.py --log results/2026-01-12/math/baseline/log.jsonl --output figures/
    python scripts/plot_math_results.py --sweep results/sweeps/temperature_sweep/ --output figures/
    python scripts/plot_math_results.py --compare log1.jsonl log2.jsonl log3.jsonl --output figures/
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import defaultdict

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

# Professional color palette
COLORS = sns.color_palette("Set2", 8)
LEVEL_COLORS = sns.color_palette("YlOrRd", 5)

class MathResultsPlotter:
    def __init__(self, output_dir="figures"):
        """Initialize the plotter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, log_file):
        """Load experiment results from JSONL."""
        results = []
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get('type') != 'summary':
                        results.append(entry)
        return results

    def load_summary(self, log_file):
        """Load summary from JSONL."""
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get('type') == 'summary':
                        return entry
        return None

    def extract_per_level_data(self, results):
        """Extract per-level performance data."""
        level_data = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in results:
            if 'level' in result and 'metrics' in result:
                level = result['level']
                level_data[level]['total'] += 1
                if result['metrics'].get('num_correct', 0) > 0:
                    level_data[level]['correct'] += 1

        # Calculate pass@1 for each level
        per_level_pass1 = {}
        for level, data in level_data.items():
            if data['total'] > 0:
                per_level_pass1[level] = (data['correct'] / data['total']) * 100

        return per_level_pass1

    def plot_per_level_performance(self, log_file, title=None, output_file="per_level_performance.pdf"):
        """Plot performance by difficulty level."""
        results = self.load_results(log_file)
        per_level_pass1 = self.extract_per_level_data(results)

        fig, ax = plt.subplots(figsize=(8, 5))

        levels = sorted(per_level_pass1.keys())
        accuracies = [per_level_pass1[level] for level in levels]

        bars = ax.bar(levels, accuracies, color=LEVEL_COLORS, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Difficulty Level', fontweight='bold')
        ax.set_ylabel('Pass@1 Accuracy (%)', fontweight='bold')
        ax.set_title(title or 'Performance by Difficulty Level', fontweight='bold', pad=15)
        ax.set_xticks(levels)
        ax.set_xticklabels([f'Level {l}' for l in levels])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def plot_temperature_comparison(self, sweep_dir, output_file="temperature_comparison.pdf"):
        """Plot temperature sweep results."""
        sweep_dir = Path(sweep_dir)

        # Find all log files
        log_files = list(sweep_dir.glob("*/log.jsonl"))

        if not log_files:
            print(f"No log files found in {sweep_dir}")
            return

        # Extract temperature and metrics
        data_by_model = defaultdict(lambda: {'temps': [], 'pass1': []})

        for log_file in log_files:
            # Extract model name and temperature from directory name
            dir_name = log_file.parent.name
            parts = dir_name.split('_')

            model_name = parts[0]
            temp = None
            for part in parts:
                if part.startswith('temp'):
                    temp = float(part.replace('temp', ''))
                    break

            if temp is None:
                continue

            summary = self.load_summary(log_file)
            if summary and 'avg_pass@1' in summary:
                data_by_model[model_name]['temps'].append(temp)
                data_by_model[model_name]['pass1'].append(summary['avg_pass@1'] * 100)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, (model_name, data) in enumerate(sorted(data_by_model.items())):
            # Sort by temperature
            sorted_indices = np.argsort(data['temps'])
            temps = np.array(data['temps'])[sorted_indices]
            pass1 = np.array(data['pass1'])[sorted_indices]

            ax.plot(temps, pass1,
                   marker='o', linewidth=2, markersize=8,
                   label=model_name, color=COLORS[idx],
                   markeredgewidth=1.5, markeredgecolor='white')

        ax.set_xlabel('Temperature', fontweight='bold')
        ax.set_ylabel('Pass@1 Accuracy (%)', fontweight='bold')
        ax.set_title('Performance vs Temperature', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def plot_model_comparison(self, log_files, labels=None, output_file="model_comparison.pdf"):
        """Compare multiple models on per-level performance."""
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(log_files))]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(1, 6)  # Levels 1-5
        width = 0.8 / len(log_files)

        for idx, (log_file, label) in enumerate(zip(log_files, labels)):
            results = self.load_results(log_file)
            per_level_pass1 = self.extract_per_level_data(results)

            accuracies = [per_level_pass1.get(level, 0) for level in range(1, 6)]
            offset = (idx - len(log_files)/2 + 0.5) * width

            ax.bar(x + offset, accuracies, width,
                  label=label, color=COLORS[idx],
                  edgecolor='black', linewidth=1.2)

        ax.set_xlabel('Difficulty Level', fontweight='bold')
        ax.set_ylabel('Pass@1 Accuracy (%)', fontweight='bold')
        ax.set_title('Model Comparison Across Difficulty Levels', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Level {l}' for l in range(1, 6)])
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def plot_pass_at_k_curve(self, log_file, title=None, output_file="pass_at_k.pdf"):
        """Plot Pass@k curve."""
        summary = self.load_summary(log_file)

        if not summary:
            print("No summary found in log file")
            return

        # Extract pass@k values
        pass_k_values = {}
        for key, value in summary.items():
            if key.startswith('avg_pass@'):
                k = int(key.split('@')[1])
                pass_k_values[k] = value * 100

        if not pass_k_values:
            print("No pass@k metrics found")
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        k_values = sorted(pass_k_values.keys())
        accuracies = [pass_k_values[k] for k in k_values]

        ax.plot(k_values, accuracies,
               marker='o', linewidth=2.5, markersize=8,
               color=COLORS[0], markeredgewidth=1.5,
               markeredgecolor='white')

        ax.set_xlabel('k (number of samples)', fontweight='bold')
        ax.set_ylabel('Pass@k Accuracy (%)', fontweight='bold')
        ax.set_title(title or 'Pass@k Performance', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def plot_subject_performance(self, log_file, title=None, output_file="subject_performance.pdf"):
        """Plot performance by subject."""
        results = self.load_results(log_file)

        subject_data = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in results:
            if 'subject' in result and 'metrics' in result:
                subject = result['subject']
                subject_data[subject]['total'] += 1
                if result['metrics'].get('num_correct', 0) > 0:
                    subject_data[subject]['correct'] += 1

        # Calculate accuracy per subject
        subjects = []
        accuracies = []
        for subject, data in sorted(subject_data.items()):
            if data['total'] > 0:
                subjects.append(subject)
                accuracies.append((data['correct'] / data['total']) * 100)

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(subjects, accuracies, color=COLORS[2], edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{acc:.1f}%',
                   ha='left', va='center', fontweight='bold', fontsize=9)

        ax.set_xlabel('Pass@1 Accuracy (%)', fontweight='bold')
        ax.set_ylabel('Subject', fontweight='bold')
        ax.set_title(title or 'Performance by Subject', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='x')
        ax.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / output_file}")
        plt.close()

    def create_summary_report(self, log_file, output_file="summary_report.txt"):
        """Create a text summary report."""
        results = self.load_results(log_file)
        summary = self.load_summary(log_file)

        with open(self.output_dir / output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MATH-500 EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")

            if summary:
                f.write("Overall Metrics:\n")
                f.write("-"*60 + "\n")
                for key, value in sorted(summary.items()):
                    if key.startswith('avg_'):
                        metric_name = key.replace('avg_', '')
                        if isinstance(value, float):
                            f.write(f"{metric_name:<20}: {value*100:>6.2f}%\n")
                f.write("\n")

            # Per-level breakdown
            per_level_pass1 = self.extract_per_level_data(results)
            f.write("Per-Level Performance:\n")
            f.write("-"*60 + "\n")
            for level in sorted(per_level_pass1.keys()):
                f.write(f"Level {level:<15}: {per_level_pass1[level]:>6.2f}%\n")
            f.write("\n")

            # Per-subject breakdown
            subject_data = defaultdict(lambda: {'correct': 0, 'total': 0})
            for result in results:
                if 'subject' in result and 'metrics' in result:
                    subject = result['subject']
                    subject_data[subject]['total'] += 1
                    if result['metrics'].get('num_correct', 0) > 0:
                        subject_data[subject]['correct'] += 1

            f.write("Per-Subject Performance:\n")
            f.write("-"*60 + "\n")
            for subject in sorted(subject_data.keys()):
                data = subject_data[subject]
                if data['total'] > 0:
                    acc = (data['correct'] / data['total']) * 100
                    f.write(f"{subject:<30}: {acc:>6.2f}% ({data['correct']}/{data['total']})\n")

            f.write("="*80 + "\n")

        print(f"Saved summary report: {self.output_dir / output_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot MATH-500 results")
    parser.add_argument("--log", type=str, help="Path to log.jsonl file")
    parser.add_argument("--sweep", type=str, help="Path to sweep directory")
    parser.add_argument("--compare", nargs='+', help="Compare multiple log files")
    parser.add_argument("--labels", nargs='+', help="Labels for comparison (optional)")
    parser.add_argument("--output", type=str, default="figures", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Generate all plot types")

    args = parser.parse_args()

    plotter = MathResultsPlotter(output_dir=args.output)

    if args.log:
        log_file = args.log
        if args.all:
            plotter.plot_per_level_performance(log_file)
            plotter.plot_pass_at_k_curve(log_file)
            plotter.plot_subject_performance(log_file)
            plotter.create_summary_report(log_file)
        else:
            plotter.plot_per_level_performance(log_file)

    elif args.sweep:
        plotter.plot_temperature_comparison(args.sweep)

    elif args.compare:
        plotter.plot_model_comparison(args.compare, labels=args.labels)

    else:
        print("Error: Must specify --log, --sweep, or --compare")
        parser.print_help()

if __name__ == "__main__":
    main()
