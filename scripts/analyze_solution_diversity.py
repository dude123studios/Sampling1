"""
Analyze solution diversity across temperature sweeps by difficulty level.

For each problem with multiple correct solutions:
1. Extract solution summaries using deepseek-r1-llama-70b
2. Embed solutions using text-embedding-3-large
3. Compute pairwise diversity (1 - cosine similarity)
4. Plot diversity metrics by temperature and level
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
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()


class SolutionDiversityAnalyzer:
    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _make_request_with_retry(self, url, payload, retries=3, timeout=30):
        import time
        for i in range(retries):
            try:
                response = requests.post(
                    url, headers=self.headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if i == retries - 1:
                    raise e
                time.sleep(2 ** i) # Exponential backoff
        return None

    def extract_solution_summary(self, full_solution: str) -> str:
        """Use deepseek-r1-llama-70b to extract the solution methodology from the full trace."""
        prompt = f"""You are analyzing a mathematical solution to identify its methodology. Extract ONLY the solution pathway, approach, and key steps used - NOT the final answer.

Full solution trace:
{full_solution[:6000]}

Summarize the solution methodology concisely (2-4 sentences). Focus ONLY on:
- The mathematical approach/strategy used (e.g., "uses substitution", "applies geometric reasoning")
- Key intermediate steps or techniques employed
- The logical pathway taken to solve the problem

IGNORE the final numerical answer entirely. Only describe HOW the problem was solved, not WHAT the answer was."""

        payload = {
            "model": "deepseek/deepseek-r1-distill-llama-70b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }

        try:
            result = self._make_request_with_retry(f"{self.base_url}/chat/completions", payload, timeout=60)
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error extracting solution: {e}")
            return full_solution[:500]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using text-embedding-3-large via OpenRouter."""
        payload = {
            "model": "openai/text-embedding-3-large",
            "input": text
        }

        try:
            result = self._make_request_with_retry(f"{self.base_url}/embeddings", payload, timeout=30)
            return np.array(result['data'][0]['embedding'])
        except Exception as e:
            print(f"Error embedding text: {e}")
            # text-embedding-3-large uses 3072 dimensions
            return np.zeros(3072)

    def process_and_analyze(self, sweep_dir: Path, model_name: str, 
                          temperatures: List[str] = None, 
                          cache_dir: Path = None,
                          specific_log_file: Path = None) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Load data and compute diversity metrics in a memory-efficient streaming manner."""
        
        # Initialize metrics storage: metrics[level][temp] = [diversities]
        raw_metrics = defaultdict(lambda: defaultdict(list))
        
        # Load embedding cache
        cache_file = cache_dir / f'embeddings_{model_name}.json' if cache_dir else None
        cache = {}
        if cache_file and cache_file.exists():
            try:
                with open(cache_file) as f:
                    cache = json.load(f)
                print(f"Loaded {len(cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}")

        # Find valid run directories
        if specific_log_file:
            # Handle directory path being passed instead of file
            if specific_log_file.is_dir():
                print(f"WARNING: Provided path {specific_log_file} is a directory, appending 'log.jsonl'")
                specific_log_file = specific_log_file / 'log.jsonl'
                
            found_logs = [specific_log_file]
            print(f"Using specific log file: {specific_log_file}")
        else:
            print(f"Searching for logs in {sweep_dir}")
            found_logs = list(sweep_dir.rglob('log.jsonl'))
            
        print(f"DEBUG: Processing {len(found_logs)} log file(s)")
        
        for log_file in tqdm(found_logs, desc=f"Processing logs for {model_name}"):
            run_dir = log_file.parent
            dir_name = run_dir.name
            
            # Extract temperature
            try:
                if "_temp" in dir_name:
                    temp_str = dir_name.split('temp')[1].split('_')[0]
                else:
                    # Fallback or specific file might not have temp in name
                    temp_str = "0.6" # Default/Fallback
                    
                # Validate it's a number
                float(temp_str)
            except (IndexError, ValueError):
                continue

            # Filter by model strict matching ONLY if not using specific file
            if not specific_log_file:
                if model_name == "qwen3-8b":
                    if "deepseek" in dir_name: continue
                    if "qwen3-8b" not in dir_name: continue
                elif model_name == "deepseek-qwen3-8b":
                    if "deepseek-qwen3-8b" not in dir_name: continue
                
                if temperatures and temp_str not in temperatures:
                    continue

            print(f"  Processing log: {dir_name} (Temp {temp_str})")
            
            file_problems = defaultdict(list)
            
            with open(log_file) as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        if entry.get('type') == 'summary': continue
                        
                        # Handle both 'correctness' (list of bools) and 'scores' (list of ints/floats)
                        correctness = entry.get('correctness')
                        if correctness is None:
                            scores = entry.get('scores')
                            if scores:
                                # Assume score > 0 is correct (usually 1.0)
                                correctness = [s > 0 for s in scores]
                        
                        if 'outputs' in entry and correctness:
                            correct_indices = [i for i, c in enumerate(correctness) if c]
                            
                            if len(correct_indices) < 2:
                                continue
                                
                            problem_id = entry.get('dataset_id', entry.get('id'))
                            for i in correct_indices:
                                if i < len(entry['outputs']):
                                    file_problems[problem_id].append({
                                        'output': entry['outputs'][i],
                                        'level': entry.get('level', 0)
                                    })
                    except (json.JSONDecodeError, IndexError):
                        continue
            
            # Prepare tasks for multithreading
            problems_to_process = []
            for pid, sols in file_problems.items():
                if len(sols) >= 2:
                    level = sols[0]['level']
                    if level in range(1, 6):
                        problems_to_process.append((pid, sols, level))
            
            processed_count = 0
            
            def process_single_problem(pid, sols, level):
                local_embeddings = []
                local_cache_updates = {}
                
                try:
                    for sol_data in sols:
                        sol_hash = hash(sol_data['output'])
                        cache_key = f"{pid}_{sol_hash}"
                        
                        # Read from main cache (safe for reading)
                        if cache_key in cache:
                            emb = np.array(cache[cache_key])
                        else:
                            # Expensive API calls here
                            summary = self.extract_solution_summary(sol_data['output'])
                            emb = self.embed_text(summary)
                            local_cache_updates[cache_key] = emb.tolist()
                            
                        local_embeddings.append(emb)
                    
                    if len(local_embeddings) >= 2:
                        mat = np.array(local_embeddings)
                        sims = cosine_similarity(mat)
                        triu = sims[np.triu_indices(len(local_embeddings), k=1)]
                        div = 1.0 - np.mean(triu)
                        return (level, div, local_cache_updates)
                        
                except Exception as e:
                    print(f"Error processing {pid}: {e}")
                    return None
                    
                return None

            # Process in batches using ThreadPoolExecutor
            if problems_to_process:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(process_single_problem, p[0], p[1], p[2]) for p in problems_to_process]
                    
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            level, diversity, updates = result
                            raw_metrics[level][temp_str].append(diversity)
                            processed_count += 1
                            
                            # Update main cache safely in main thread
                            if updates:
                                cache.update(updates)

            if cache_file:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
            
            print(f"    - Computed metrics for {processed_count} problems")
            
            del file_problems
            import gc; gc.collect()

        # Aggregate metrics
        final_stats = {level: {} for level in range(1, 6)}
        
        for level in range(1, 6):
            temps = set(raw_metrics[level].keys())
            for t in temps:
                divs = raw_metrics[level][t]
                if divs:
                    final_stats[level][t] = {
                        'avg_diversity': np.mean(divs),
                        'std_diversity': np.std(divs),
                        'n_problems': len(divs)
                    }
                    print(f"Level {level} Temp {t}: Avg Div {np.mean(divs):.4f} (n={len(divs)})")
                else:
                    final_stats[level][t] = {'avg_diversity': 0.0, 'std_diversity': 0.0, 'n_problems': 0}
                    
        return final_stats

    def plot_diversity(self, metrics_by_level: Dict[int, Dict[str, Dict[str, float]]], model_name: str, output_path: str):
        """Create 5 side-by-side bar plots (one per difficulty level)."""
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

        # Dynamic color mapping for any temperatures
        all_temps = set()
        for level_metrics in metrics_by_level.values():
            all_temps.update(float(t) for t in level_metrics.keys())

        # Use viridis gradient for whatever temperatures we have
        temp_list = sorted(all_temps)
        colors_list = plt.cm.viridis(np.linspace(0.2, 0.8, len(temp_list)))
        temp_colors = {t: colors_list[i] for i, t in enumerate(temp_list)}

        for level_idx, level in enumerate(range(1, 6)):
            ax = axes[level_idx]
            level_metrics = metrics_by_level[level]
            
            if not level_metrics:
                ax.set_visible(False)
                continue

            temperatures = sorted([float(t) for t in level_metrics.keys()])
            diversities = [level_metrics[str(t)]['avg_diversity'] for t in temperatures]
            stds = [level_metrics[str(t)]['std_diversity'] for t in temperatures]
            colors = [temp_colors[t] for t in temperatures]

            x_pos = np.arange(len(temperatures))
            width = 0.6

            bars = ax.bar(
                x_pos, diversities, width,
                yerr=stds,
                color=colors,
                edgecolor='black',
                linewidth=1.5,
                capsize=5,
                error_kw={'linewidth': 1.5, 'ecolor': 'black'},
                zorder=3
            )

            ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
            if level_idx == 0:
                ax.set_ylabel('Diversity (1 - cosine sim)', fontsize=12, fontweight='bold')

            ax.set_title(f'Level {level}', fontsize=13, fontweight='bold', pad=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{t:.1f}' for t in temperatures])
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.tick_params(axis='both', labelsize=11)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=temp_colors[t], edgecolor='black', linewidth=1.5, label=f'Temp={t:.1f}')
            for t in sorted(temp_colors.keys())
        ]
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=True,
            fontsize=11,
            edgecolor='black',
            framealpha=1.0
        )

        model_display = model_name.replace('-', ' ')
        fig.suptitle(
            f'Solution Diversity by Difficulty Level: {model_display}',
            fontsize=15,
            fontweight='bold',
            y=1.05
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze solution diversity across temperature sweeps")
    parser.add_argument('--sweep-dir', type=str, default='results/sweeps/temperature_sweep')
    parser.add_argument('--output-dir', type=str, default='results/plots')
    parser.add_argument('--cache-dir', type=str, default='results/diversity_cache')
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--models', nargs='+', default=['qwen3-8b', 'deepseek-qwen3-8b'],
                        help='Models to process')
    parser.add_argument('--temperatures', nargs='+', default=None,
                        help='Temperatures to include (e.g., 0.3 0.6 0.9)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Specific log file to process (bypasses recursion)')

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("Please provide --api-key or set OPENROUTER_API_KEY environment variable")

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SolutionDiversityAnalyzer(api_key)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")

        print("Processing and analyzing data...")
        specific_log = Path(args.log_file) if args.log_file else None
        
        # Load existing metrics if available to allow incremental updates
        metrics_file = output_dir / f'diversity_metrics_{model_name}.json'
        current_metrics = {}
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    # JSON keys are strings, convert level keys back to ints
                    current_metrics = {int(k): v for k, v in data.items()}
                print(f"Loaded existing metrics for {len(current_metrics)} levels")
            except Exception as e:
                print(f"Warning: Could not load existing metrics: {e}")

        # Compute new metrics for the current run
        new_metrics = analyzer.process_and_analyze(
            sweep_dir, 
            model_name, 
            temperatures=args.temperatures, 
            cache_dir=cache_dir,
            specific_log_file=specific_log
        )

        # Merge new metrics into current_metrics
        # We assume different runs process different temperatures. 
        # If we re-run the same temperature, we overwrite it with the new result.
        for level, temp_dict in new_metrics.items():
            if level not in current_metrics:
                current_metrics[level] = {}
            
            for temp, stats in temp_dict.items():
                # Only update if we actually found data
                if stats.get('n_problems', 0) > 0:
                    current_metrics[level][temp] = stats
        
        # Save updated aggregated metrics
        with open(metrics_file, 'w') as f:
            json.dump(current_metrics, f, indent=2)
        print(f"Saved aggregated metrics to {metrics_file}")

        output_path = output_dir / f'solution_diversity_{model_name}.png'
        analyzer.plot_diversity(current_metrics, model_name, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
