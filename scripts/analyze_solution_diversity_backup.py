"""
Analyze solution diversity across temperature sweeps.

For each problem with multiple correct solutions:
1. Extract solution summaries using deepseek-r1-llama-70b
2. Embed solutions using text-embedding-3-large
3. Compute pairwise diversity (1 - cosine similarity)
4. Plot diversity metrics by temperature
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse
from tqdm import tqdm
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity


class SolutionDiversityAnalyzer:
    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def extract_solution_summary(self, full_solution: str) -> str:
        """
        Use deepseek-r1-llama-70b to extract the final solution from the full trace.
        """
        prompt = f"""You are given a solution trace from a math problem-solving model. Your task is to extract and summarize ONLY the final mathematical solution (the answer and key approach), not the reasoning process.

Full solution trace:
{full_solution}

Extract the final solution concisely (1-3 sentences maximum). Focus on:
- The final answer/result
- The key mathematical approach or method used

Do not include the step-by-step reasoning, only the final solution."""

        payload = {
            "model": "deepseek/deepseek-r1-distill-llama-70b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error extracting solution: {e}")
            return full_solution[:500]  # Fallback to truncated original

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using text-embedding-3-large via OpenRouter.
        """
        payload = {
            "model": "openai/text-embedding-3-large",
            "input": text
        }

        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return np.array(result['data'][0]['embedding'])
        except Exception as e:
            print(f"Error embedding text: {e}")
            return np.zeros(1024)  # Fallback

    def load_sweep_data(self, sweep_dir: Path, model_name: str) -> Dict:
        """
        Load sweep results for a specific model.

        Returns:
            Dict mapping temperature -> problem_id -> list of correct solutions
        """
        results = defaultdict(lambda: defaultdict(list))

        for run_dir in sweep_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Filter by model
            if model_name == "qwen3-8b":
                if "deepseek" in run_dir.name:
                    continue
                if "qwen3-8b" not in run_dir.name:
                    continue
            elif model_name == "deepseek-qwen3-8b":
                if "deepseek-qwen3-8b" not in run_dir.name:
                    continue

            # Extract temperature
            parts = run_dir.name.split('_temp')
            if len(parts) != 2:
                continue
            temp_str = parts[1].split('_')[0]

            # Read log file
            log_file = run_dir / 'log.jsonl'
            if not log_file.exists():
                continue

            with open(log_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get('type') == 'summary':
                            continue

                        # Check if any outputs are correct
                        if 'outputs' in entry and 'correctness' in entry:
                            problem_id = entry.get('dataset_id', entry.get('id'))
                            for i, correct in enumerate(entry['correctness']):
                                if correct and i < len(entry['outputs']):
                                    results[temp_str][problem_id].append({
                                        'output': entry['outputs'][i],
                                        'problem': entry.get('original_prompt', ''),
                                        'level': entry.get('level', 0)
                                    })
                    except json.JSONDecodeError:
                        continue

        return results

    def compute_diversity_metrics(
        self,
        sweep_data: Dict,
        cache_file: Path = None
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Compute diversity metrics for each temperature and level.

        Returns:
            Dict mapping level -> temperature -> {'avg_diversity': float, 'n_problems': int}
        """
        # Load cache if exists
        cache = {}
        if cache_file and cache_file.exists():
            with open(cache_file) as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached embeddings")

        # Organize by level
        metrics_by_level = {level: {} for level in range(1, 6)}

        for temp in sorted(sweep_data.keys()):
            print(f"\nProcessing temperature {temp}...")

            # Track diversity per level
            level_diversities = {level: [] for level in range(1, 6)}

            for problem_id, solutions in tqdm(sweep_data[temp].items(), desc=f"Temp {temp}"):
                # Need at least 2 correct solutions to measure diversity
                if len(solutions) < 2:
                    continue

                # Get level from first solution (all should be same problem)
                level = solutions[0]['level']
                if level not in range(1, 6):
                    continue

                # Extract and embed solutions
                embeddings = []
                for sol_data in solutions:
                    cache_key = f"{problem_id}_{hash(sol_data['output'])}"

                    if cache_key in cache:
                        embedding = np.array(cache[cache_key])
                    else:
                        # Extract summary
                        summary = self.extract_solution_summary(sol_data['output'])

                        # Embed
                        embedding = self.embed_text(summary)

                        # Cache
                        cache[cache_key] = embedding.tolist()

                    embeddings.append(embedding)

                # Compute pairwise cosine similarities
                if len(embeddings) >= 2:
                    embeddings_matrix = np.array(embeddings)
                    similarities = cosine_similarity(embeddings_matrix)

                    # Get upper triangle (pairwise similarities, excluding diagonal)
                    n = len(embeddings)
                    pairwise_sims = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            pairwise_sims.append(similarities[i, j])

                    # Diversity = 1 - mean similarity
                    avg_similarity = np.mean(pairwise_sims)
                    diversity = 1.0 - avg_similarity
                    level_diversities[level].append(diversity)

            # Save cache
            if cache_file:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)

            # Aggregate metrics per level
            for level in range(1, 6):
                diversities = level_diversities[level]
                if diversities:
                    metrics_by_level[level][temp] = {
                        'avg_diversity': np.mean(diversities),
                        'std_diversity': np.std(diversities),
                        'n_problems': len(diversities)
                    }
                else:
                    metrics_by_level[level][temp] = {
                        'avg_diversity': 0.0,
                        'std_diversity': 0.0,
                        'n_problems': 0
                    }

            # Print summary
            for level in range(1, 6):
                n = metrics_by_level[level][temp]['n_problems']
                if n > 0:
                    div = metrics_by_level[level][temp]['avg_diversity']
                    print(f"  Level {level}: {div:.4f} (n={n})")

        return metrics_by_level

    def plot_diversity(
        self,
        metrics_by_level: Dict[int, Dict[str, Dict[str, float]]],
        model_name: str,
        output_path: str
    ):
        """
        Create 5 side-by-side bar plots (one per difficulty level).
        """
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

        # Color palette - viridis-like gradient
        temp_colors = {
            0.3: '#440154',  # Dark purple
            0.6: '#31688e',  # Teal blue
            0.9: '#35b779'   # Green
        }
        colors = [temp_colors[t] for t in temperatures]

        # Bar positions
        x_pos = np.arange(len(temperatures))
        width = 0.6

        # Bar plot
        bars = ax.bar(
            x_pos,
            diversities,
            width,
            yerr=stds,
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            capsize=6,
            error_kw={'linewidth': 1.5, 'ecolor': 'black'},
            zorder=3
        )

        # Add n_problems labels on bars
        for i, (bar, n) in enumerate(zip(bars, n_problems)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + stds[i] + 0.01,
                f'n={n}',
                ha='center',
                va='bottom',
                fontsize=10,
                color='gray'
            )

        # Styling
        ax.set_xlabel('Temperature', fontsize=13, fontweight='bold')
        ax.set_ylabel('Solution Diversity (1 - cosine similarity)', fontsize=13, fontweight='bold')

        model_display = model_name.replace('-', ' ')
        ax.set_title(
            f'Solution Diversity Across Temperatures: {model_display}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{t:.1f}' for t in temperatures])

        # Set y-axis
        ax.set_ylim(0, min(1.0, max(diversities) * 1.3))

        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.tick_params(axis='both', labelsize=11)

        # Subtitle
        subtitle = 'Higher values indicate more diverse solution strategies'
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze solution diversity across temperature sweeps"
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
        '--cache-dir',
        type=str,
        default='results/diversity_cache',
        help='Directory for caching embeddings'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("Please provide --api-key or set OPENROUTER_API_KEY environment variable")

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SolutionDiversityAnalyzer(api_key)

    # Process each model
    for model_name in ['qwen3-8b', 'deepseek-qwen3-8b']:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")

        # Load data
        print("Loading sweep data...")
        sweep_data = analyzer.load_sweep_data(sweep_dir, model_name)

        if not sweep_data:
            print(f"No data found for {model_name}")
            continue

        # Report statistics
        for temp in sorted(sweep_data.keys()):
            n_problems = len(sweep_data[temp])
            n_multi_correct = sum(1 for sols in sweep_data[temp].values() if len(sols) >= 2)
            print(f"  Temp {temp}: {n_problems} problems, {n_multi_correct} with ≥2 correct")

        # Compute diversity metrics
        cache_file = cache_dir / f'embeddings_{model_name}.json'
        metrics = analyzer.compute_diversity_metrics(sweep_data, cache_file)

        # Plot
        output_path = output_dir / f'solution_diversity_{model_name}.png'
        analyzer.plot_diversity(metrics, model_name, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
