"""
Trajectory Bifurcation Analysis - Embeddings Version

This version runs entirely on OpenRouter API (no local models).
Uses Qwen embedding model to embed generated solutions for bifurcation analysis.

For a given problem:
1. Generate k solutions using OpenRouter API with multithreading
2. Extract just the generated text (not prompt) from each solution
3. Embed each generated text using qwen/qwen3-embedding-8b via OpenRouter
4. Perform PCA on the embeddings
5. Plot bifurcation analysis

Usage:
    python mech_interp/bifurcation_analysis_embeddings.py --config configs/mech_interp/bifurcation_config.yaml
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.math_grader import grade_math


def load_problem_directly(problem_id: int = 24):
    """Directly load problem 24 from MATH-500."""
    log.info(f"Loading problem ID {problem_id} directly from HuggingFace (ignoring sweep data)...")
    from datasets import load_dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Assuming ID corresponds to index.
    # Logic: The sweep assigned IDs sequentially 0..499 matching the dataset index.
    if problem_id >= len(dataset):
        log.error(f"Problem ID {problem_id} out of range (0-{len(dataset)-1})")
        return None

    item = dataset[problem_id]
    return {
        'problem_id': f"id_{problem_id}",
        'problem': item['problem'],
        'answer': item['answer'],
        'outputs': [], # Will generate these
        'correctness': [],
        'level': item.get('level', 0)
    }


class OpenRouterAPI:
    """Simple OpenRouter API client for text generation and embeddings."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://neurips-experiment.com",
            "X-Title": "Sampling Limits NeurIPS",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, **kwargs):
        """Generate text using OpenRouter API."""
        data = {
            "model": kwargs.get('model', 'qwen/qwen3-8b'),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_new_tokens', 4096),
            "top_p": kwargs.get('top_p', 1.0),
            "top_k": kwargs.get('top_k', None)
        }
        if data['top_k'] is None:
            del data['top_k']

        retries = 3
        for i in range(retries):
            try:
                import requests
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=kwargs.get('timeout', 300)
                )
                response.raise_for_status()
                resp_json = response.json()
                return resp_json['choices'][0]['message']['content']
            except Exception as e:
                if i == retries - 1:
                    raise e
                import time
                time.sleep(2 ** i)

    def embed(self, text: str):
        """Get embeddings using qwen/qwen3-embedding-8b."""
        data = {
            "model": "qwen/qwen3-embedding-8b",
            "input": text
        }

        retries = 3
        for i in range(retries):
            try:
                import requests
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=data,
                    timeout=60  # Shorter timeout for embeddings
                )
                response.raise_for_status()
                resp_json = response.json()
                return np.array(resp_json['data'][0]['embedding'])
            except Exception as e:
                if i == retries - 1:
                    raise e
                import time
                time.sleep(2 ** i)


def run_bifurcation_analysis_embeddings(config_path: str):
    """Run bifurcation analysis using embeddings."""
    # Load environment variables
    load_dotenv()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load problem directly
    problem_id = cfg['analysis'].get('problem_id', 24)
    problem = load_problem_directly(problem_id)
    if not problem:
        log.error(f"Failed to load problem {problem_id}")
        return

    # Build prompt
    prompt = f"""You are a helpful mathematical assistant. Solve the following problem step-by-step.
IMPORTANT: You must put your final answer within \\boxed{{}}.

Problem:
{problem['problem']}

Solution:
"""

    outputs = problem['outputs']
    labels = []

    # Get parameters
    max_new_tokens = cfg['analysis'].get('max_new_tokens', 4096)
    temp = cfg['analysis'].get('temperature', 0.6)

    # Setup output directory
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg['model']['name']

    # Check for existing solutions
    solutions_file = output_dir / f"{model_name}_problem_{problem_id}_embeddings_solutions.json"

    if solutions_file.exists():
        log.info(f"Loading existing solutions from {solutions_file}")
        try:
            with open(solutions_file, 'r') as f:
                saved_data = json.load(f)
                if 'solutions' in saved_data and len(saved_data['solutions']) > 0:
                    outputs = saved_data['solutions']
                    labels = saved_data.get('labels', [])
                    log.info(f"Loaded {len(outputs)} existing solutions")
                    if len(labels) != len(outputs):
                        # Regrade if labels don't match
                        log.info("Regrading solutions...")
                        labels = []
                        for sol in outputs:
                            is_correct = grade_math(sol, problem['answer'])
                            labels.append(1 if is_correct else 0)
                else:
                    outputs = []
        except Exception as e:
            log.warning(f"Error loading solutions file: {e}. Will generate new ones.")
            outputs = []

    # Initialize API client
    api_config = cfg.get('api', {})
    if not api_config:
        raise ValueError("API configuration required. Add 'api' section to config with api_key")

    api_key = api_config.get('api_key', None)
    if not api_key:
        api_key_env = api_config.get('api_key_env', 'OPENROUTER_API_KEY')
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found. Set 'api_key' in config or environment variable '{api_key_env}'")

    api = OpenRouterAPI(api_key)

    # Generate samples if needed
    n_samples = cfg['analysis'].get('n_samples', 100)
    if len(outputs) < n_samples:
        if outputs:
            log.info(f"Loaded {len(outputs)} solutions but need {n_samples}, generating additional ones")
            samples_to_generate = n_samples - len(outputs)
        else:
            samples_to_generate = n_samples
            log.info(f"Generating {n_samples} samples for Problem {problem_id} using OpenRouter API (temp={temp}, max_tokens={max_new_tokens})...")

        # Generate solutions in parallel
        def generate_one_solution(idx):
            try:
                solution = api.generate(
                    prompt,
                    temperature=temp,
                    max_new_tokens=max_new_tokens,
                    top_p=0.9,
                    top_k=50
                )
                return idx, solution, None
            except Exception as e:
                log.error(f"Error generating solution {idx}: {e}")
                return idx, "", str(e)

        generated_solutions = [None] * samples_to_generate
        max_workers = api_config.get('max_workers', 15)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_one_solution, i): i for i in range(samples_to_generate)}
            for future in tqdm(as_completed(futures), total=samples_to_generate, desc="Generating"):
                idx, sol, error = future.result()
                if error:
                    log.warning(f"Solution {idx} failed: {error}")
                    generated_solutions[idx] = ""
                else:
                    generated_solutions[idx] = sol

        # Combine with existing solutions
        if outputs:
            outputs.extend(generated_solutions)
        else:
            outputs = generated_solutions

        # Ensure we have exactly n_samples
        if len(outputs) > n_samples:
            outputs = outputs[:n_samples]
        elif len(outputs) < n_samples:
            while len(outputs) < n_samples:
                outputs.append("")

        # Grade all solutions
        log.info("Grading generated solutions...")
        labels = []
        for i, sol in enumerate(outputs):
            if sol and sol.strip():
                is_correct = grade_math(sol, problem['answer'])
                labels.append(1 if is_correct else 0)
            else:
                labels.append(0)
                if i < 5:
                    log.warning(f"Solution {i} was empty/failed, counting as incorrect")

        # Save solutions
        log.info(f"About to save: outputs has {len(outputs)} items, labels has {len(labels)} items")
        log.info(f"First 5 outputs: {[repr(s[:50]) for s in outputs[:5]]}")
        log.info(f"First 5 labels: {labels[:5]}")
        log.info(f"Saving {len(outputs)} solutions to {solutions_file}")
        try:
            valid_outputs = [s for s in outputs if s and s.strip()]
            with open(solutions_file, 'w') as f:
                json.dump({
                    'problem_id': problem_id,
                    'model_name': model_name,
                    'temperature': temp,
                    'n_samples': len(outputs),
                    'valid_samples': len(valid_outputs),
                    'unique_samples': len(set(valid_outputs)),
                    'solutions': outputs,
                    'labels': labels
                }, f, indent=2)
            log.info("Solutions saved successfully")
        except Exception as e:
            log.warning(f"Failed to save solutions: {e}")

    else:
        # Already have enough solutions
        log.info(f"Using {len(outputs)} existing solutions")

    # Verify diversity
    valid_outputs = [s for s in outputs if s and s.strip()]
    unique_solutions = set(valid_outputs)
    log.info(f"Total solutions: {len(outputs)}, Valid: {len(valid_outputs)}, Unique: {len(unique_solutions)}")

    correct_count = sum(labels)
    total_count = len(labels)
    success_rate = correct_count / total_count if total_count > 0 else 0
    log.info(f"Success rate: {correct_count}/{total_count} = {success_rate:.2%}")
    log.info(f"First 10 labels: {labels[:10]}")

    # Generate embeddings for ALL solutions (including empty ones)
    log.info(f"Generating embeddings for {len(outputs)} solutions...")
    embeddings = []

    def embed_one_solution(idx, solution_text):
        try:
            if solution_text and solution_text.strip():
                embedding = api.embed(solution_text)
                return idx, embedding, None
            else:
                # Empty solution - use zero embedding
                log.debug(f"Solution {idx}: Empty text, using zero embedding")
                return idx, np.zeros(1024), "empty_solution"  # Qwen embedding dim is 1024
        except Exception as e:
            log.error(f"Error embedding solution {idx}: {e}")
            return idx, np.zeros(1024), str(e)

    embedding_results = [None] * len(outputs)
    embedding_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_one_solution, i, sol): i for i, sol in enumerate(outputs)}
        for future in tqdm(as_completed(futures), total=len(outputs), desc="Embedding"):
            idx, embedding, error = future.result()
            embedding_count += 1
            if error and error != "empty_solution":
                log.warning(f"Embedding {idx} failed: {error}")
            embedding_results[idx] = embedding

    embeddings = np.array(embedding_results)
    log.info(f"Embeddings completed: processed {embedding_count}/{len(outputs)} solutions, shape: {embeddings.shape}")

    if len(embeddings) != len(outputs):
        log.error(f"EMBEDDING MISMATCH: Expected {len(outputs)} embeddings but got {len(embeddings)}!")
        # This shouldn't happen, but just in case
        while len(embedding_results) < len(outputs):
            embedding_results.append(np.zeros(1024))
        embeddings = np.array(embedding_results)

    # PCA analysis
    if len(embeddings) < 2:
        log.error("Not enough embeddings for PCA")
        return

    try:
        # Check for identical embeddings
        sample_diffs = []
        for i in range(min(5, len(embeddings))):
            for j in range(i+1, min(5, len(embeddings))):
                diff = np.mean(np.abs(embeddings[i] - embeddings[j]))
                sample_diffs.append(diff)
        if sample_diffs:
            mean_diff = np.mean(sample_diffs)
            log.info(f"Mean difference between first 5 embeddings: {mean_diff:.6f}")

        # PCA
        log.info("Running PCA on embeddings...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        log.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        log.info(f"PCA output range: [{np.min(embeddings_2d):.4f}, {np.max(embeddings_2d):.4f}]")

        # Create results dict
        results = {
            'embeddings_2d': embeddings_2d.tolist(),
            'labels': labels,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'n_success': correct_count,
            'n_fail': total_count - correct_count,
            'problem_id': problem['problem_id'],
            'problem_text': problem['problem'],
            'gold_answer': problem['answer'],
            'temperature': temp,
            'n_samples': len(outputs)
        }

        # Save results
        results_file = output_dir / f"{model_name}_problem_{problem_id}_embeddings_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved results to {results_file}")

        # Create improved plot - more dense and visually appealing
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        success_mask = np.array(labels) == 1
        fail_mask = np.array(labels) == 0

        # Use smaller, more transparent points for density
        ax.scatter(embeddings_2d[fail_mask, 0], embeddings_2d[fail_mask, 1],
                   c='#DC143C', s=50, alpha=0.7, edgecolors='white', linewidths=0.5,
                   label=f'Failed ({results["n_fail"]})', zorder=2)

        ax.scatter(embeddings_2d[success_mask, 0], embeddings_2d[success_mask, 1],
                   c='#4169E1', s=50, alpha=0.7, edgecolors='white', linewidths=0.5,
                   label=f'Success ({results["n_success"]})', zorder=3)

        ax.set_xlabel(f'PC1 ({results["explained_variance"][0]:.1%} variance)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({results["explained_variance"][1]:.1%} variance)', fontsize=14, fontweight='bold')

        # Calculate nice axis limits to make it less spread out
        x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()
        y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()
        x_center = (embeddings_2d[:, 0].max() + embeddings_2d[:, 0].min()) / 2
        y_center = (embeddings_2d[:, 1].max() + embeddings_2d[:, 1].min()) / 2

        # Add 20% padding
        padding = 0.2
        ax.set_xlim(x_center - x_range/2 * (1 + padding), x_center + x_range/2 * (1 + padding))
        ax.set_ylim(y_center - y_range/2 * (1 + padding), y_center + y_range/2 * (1 + padding))

        ax.set_title(f'Bifurcation Analysis (Embeddings)\n{model_name} • Problem {problem_id} • Qwen Embedding Model',
                     fontsize=16, fontweight='bold', pad=20)

        # Lighter grid
        ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
        ax.set_axisbelow(True)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Better legend
        legend = ax.legend(loc='upper right', frameon=True, fontsize=12,
                          framealpha=0.9, edgecolor='gray', fancybox=True)
        legend.get_frame().set_linewidth(0.5)

        plt.tight_layout()
        plot_file = output_dir / f"{model_name}_problem_{problem_id}_embeddings_bifurcation.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        log.info(f"Saved plot to {plot_file}")

        return results

    except Exception as e:
        log.error(f"PCA failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mech_interp/bifurcation_config.yaml")
    args = parser.parse_args()

    run_bifurcation_analysis_embeddings(args.config)