"""
Trajectory Bifurcation Analysis via PCA

Analyzes where "good" and "bad" reasoning paths diverge early in generation.

Key Question: Can we predict success/failure from early hidden states?

Method:
1. Pick hard problems (low pass rate)
2. Generate 100 solutions at T=0.6, label as Success/Fail
3. Extract hidden state at token 16, layer 10 for all prefixes (prefill only)
4. PCA to 2D
5. Plot: Blue (success), Red (fail), Green star (greedy)

Hypothesis: If clusters separate, early layers "know" the outcome.
"""

import torch
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import argparse
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.math_grader import grade_math
from src.data.loader import load_task_data
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TrajectoryBifurcationAnalyzer:
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize analyzer with model."""
        self.device = device
        self.model_name = model_name

        log.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model architecture info
        self.num_layers = len(self.model.model.layers)
        log.info(f"Model has {self.num_layers} layers")

    def find_hard_problems(
        self,
        dataset: List[Dict],
        num_samples: int = 5,
        target_pass_rate: float = 0.2
    ) -> List[Dict]:
        """
        Find hard problems with low pass rate.

        Returns problems where exactly 1 out of num_samples is correct.
        """
        log.info(f"Searching for hard problems with ~{target_pass_rate} pass rate...")

        hard_problems = []

        for problem in tqdm(dataset[:50], desc="Scanning problems"):  # Check first 50
            # Generate num_samples solutions
            prompt = f"""You are a helpful mathematical assistant. Solve the following problem step-by-step.
IMPORTANT: You must put your final answer within \\boxed{{}}.

Problem:
{problem['problem']}

Solution:
"""

            correct_count = 0
            for _ in range(num_samples):
                output = self.generate_solution(prompt, temperature=0.6)
                is_correct = grade_math(output, problem['answer'])
                if is_correct:
                    correct_count += 1

            pass_rate = correct_count / num_samples

            if abs(pass_rate - target_pass_rate) < 0.1:  # Within 10% of target
                problem['pass_rate'] = pass_rate
                problem['prompt'] = prompt
                hard_problems.append(problem)
                log.info(f"Found hard problem: {problem.get('unique_id', 'unknown')} (pass rate: {pass_rate:.2f})")

                if len(hard_problems) >= 2:  # Get 2 hard problems (one for each model)
                    break

        return hard_problems

    def generate_solution(self, prompt: str, temperature: float = 0.6) -> str:
        """Generate a single solution."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        solution = full_text[len(prompt):]
        return solution

    def extract_hidden_state_at_position(
        self,
        prefix_text: str,
        token_position: int,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract hidden state at specific token position and layer.

        Uses prefill (no generation), extracts state at token_position in layer_idx.

        Returns:
            Hidden state vector [hidden_dim]
        """
        # Tokenize prefix
        inputs = self.tokenizer(prefix_text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        # Check if we have enough tokens
        seq_len = input_ids.shape[1]
        if seq_len < token_position:
            # Pad or handle short sequences
            log.warning(f"Sequence too short ({seq_len} < {token_position}), using last token")
            token_position = seq_len - 1

        # Register hook to capture hidden state
        hidden_state = None

        def hook_fn(module, input, output):
            nonlocal hidden_state
            # output is (batch, seq, hidden) or tuple
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Extract specific position
            hidden_state = hidden[0, token_position, :].detach().cpu()

        # Register hook on target layer
        target_layer = self.model.model.layers[layer_idx]
        hook = target_layer.register_forward_hook(hook_fn)

        # Forward pass (prefill only)
        with torch.no_grad():
            _ = self.model(input_ids)

        hook.remove()

        return hidden_state

    def run_bifurcation_analysis(
        self,
        problem: Dict,
        n_samples: int = 100,
        token_position: int = 16,
        layer_idx: int = 10,
        temperature: float = 0.6
    ) -> Dict:
        """
        Main bifurcation analysis.

        Returns:
            Dict with PCA results, labels, and metadata
        """
        log.info(f"Running bifurcation analysis on problem: {problem.get('unique_id', 'unknown')}")
        log.info(f"Generating {n_samples} solutions at T={temperature}")

        prompt = problem['prompt']
        gold_answer = problem['answer']

        # Generate solutions and extract hidden states
        hidden_states = []
        labels = []  # 1 for success, 0 for fail
        solutions = []

        for i in tqdm(range(n_samples), desc="Generating solutions"):
            # Generate solution
            solution = self.generate_solution(prompt, temperature=temperature)
            solutions.append(solution)

            # Grade
            is_correct = grade_math(solution, gold_answer)
            labels.append(1 if is_correct else 0)

            # Extract hidden state from prefix (prompt + first part of solution)
            # We want exactly token_position tokens total
            full_text = prompt + solution
            inputs = self.tokenizer(full_text, return_tensors="pt")

            # Get prefix up to token_position
            if inputs['input_ids'].shape[1] >= token_position:
                prefix_ids = inputs['input_ids'][0, :token_position]
                prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            else:
                # Use full text if shorter
                prefix_text = full_text

            # Extract hidden state
            h = self.extract_hidden_state_at_position(prefix_text, token_position - 1, layer_idx)
            hidden_states.append(h.numpy())

        # Also get greedy (T=0) solution
        log.info("Generating greedy solution (T=0)...")
        greedy_solution = self.generate_solution(prompt, temperature=0.0)
        greedy_is_correct = grade_math(greedy_solution, gold_answer)

        # Extract greedy hidden state
        greedy_full = prompt + greedy_solution
        greedy_inputs = self.tokenizer(greedy_full, return_tensors="pt")
        if greedy_inputs['input_ids'].shape[1] >= token_position:
            greedy_prefix_ids = greedy_inputs['input_ids'][0, :token_position]
            greedy_prefix_text = self.tokenizer.decode(greedy_prefix_ids, skip_special_tokens=True)
        else:
            greedy_prefix_text = greedy_full

        greedy_hidden = self.extract_hidden_state_at_position(
            greedy_prefix_text, token_position - 1, layer_idx
        )

        # Convert to numpy array
        hidden_states = np.array(hidden_states)  # [n_samples, hidden_dim]
        labels = np.array(labels)

        log.info(f"Success rate: {labels.sum()}/{n_samples} = {labels.mean():.2%}")
        log.info(f"Greedy solution: {'CORRECT' if greedy_is_correct else 'INCORRECT'}")

        # Run PCA
        log.info("Running PCA to reduce to 2D...")
        pca = PCA(n_components=2)

        # Fit on all hidden states
        hidden_2d = pca.fit_transform(hidden_states)
        greedy_2d = pca.transform(greedy_hidden.numpy().reshape(1, -1))

        explained_var = pca.explained_variance_ratio_
        log.info(f"PCA explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")

        return {
            'hidden_2d': hidden_2d,
            'labels': labels,
            'greedy_2d': greedy_2d[0],
            'greedy_correct': greedy_is_correct,
            'explained_variance': explained_var,
            'n_success': labels.sum(),
            'n_fail': len(labels) - labels.sum(),
            'problem_id': problem.get('unique_id', 'unknown'),
            'layer_idx': layer_idx,
            'token_position': token_position,
            'temperature': temperature
        }

    def plot_bifurcation(
        self,
        results: Dict,
        output_path: str,
        model_display_name: str = None
    ):
        """
        Create trajectory bifurcation plot.

        Blue dots: Success
        Red dots: Fail
        Green star: Greedy
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        hidden_2d = results['hidden_2d']
        labels = results['labels']
        greedy_2d = results['greedy_2d']
        greedy_correct = results['greedy_correct']

        # Separate success and fail
        success_mask = labels == 1
        fail_mask = labels == 0

        success_points = hidden_2d[success_mask]
        fail_points = hidden_2d[fail_mask]

        # Plot fail (red) first so success is on top
        ax.scatter(
            fail_points[:, 0],
            fail_points[:, 1],
            c='#E63946',
            s=80,
            alpha=0.6,
            edgecolors='black',
            linewidths=1.2,
            label=f'Failed ({results["n_fail"]})',
            zorder=2
        )

        # Plot success (blue)
        ax.scatter(
            success_points[:, 0],
            success_points[:, 1],
            c='#457B9D',
            s=80,
            alpha=0.6,
            edgecolors='black',
            linewidths=1.2,
            label=f'Success ({results["n_success"]})',
            zorder=3
        )

        # Plot greedy (green star)
        greedy_color = '#2A9D8F' if greedy_correct else '#F4A261'
        greedy_label = 'Greedy (Correct)' if greedy_correct else 'Greedy (Incorrect)'

        ax.scatter(
            greedy_2d[0],
            greedy_2d[1],
            c=greedy_color,
            s=400,
            alpha=1.0,
            marker='*',
            edgecolors='black',
            linewidths=2.0,
            label=greedy_label,
            zorder=4
        )

        # Styling
        ax.set_xlabel(f'PC1 ({results["explained_variance"][0]:.1%} var)', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'PC2 ({results["explained_variance"][1]:.1%} var)', fontsize=13, fontweight='bold')

        model_name = model_display_name or "Model"
        ax.set_title(
            f'Trajectory Bifurcation Analysis: {model_name}\n'
            f'Layer {results["layer_idx"]}, Token {results["token_position"]} (T={results["temperature"]})',
            fontsize=14,
            fontweight='bold',
            pad=15
        )

        # Grid
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        ax.tick_params(axis='both', labelsize=11)

        # Legend
        ax.legend(
            loc='best',
            frameon=True,
            fontsize=11,
            edgecolor='black',
            framealpha=0.95,
            fancybox=False
        )

        # Add interpretation note
        interpretation = (
            'Interpretation: Separation between blue/red indicates early divergence.\n'
            'Green star position shows if greedy decoding follows the success cluster.'
        )
        fig.text(
            0.5, 0.02,
            interpretation,
            ha='center',
            fontsize=9,
            style='italic',
            color='gray',
            wrap=True
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            log.info(f"Saved bifurcation plot to: {output_path}")

        return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory bifurcation via PCA on early hidden states"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--model-display-name',
        type=str,
        default=None,
        help='Display name for plots'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='HuggingFaceH4/MATH-500',
        help='Dataset to use'
    )
    parser.add_argument(
        '--level',
        type=int,
        default=5,
        help='Difficulty level to analyze (default: 5 = hardest)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of solutions to generate per problem'
    )
    parser.add_argument(
        '--token-position',
        type=int,
        default=16,
        help='Token position to extract hidden state'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=10,
        help='Layer index to extract hidden state'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/bifurcation',
        help='Output directory for plots and results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    log.info(f"Loading dataset: {args.dataset}, level {args.level}")
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, split='test')
    dataset = dataset.filter(lambda x: x['level'] == args.level)
    dataset = list(dataset)
    log.info(f"Loaded {len(dataset)} level {args.level} problems")

    # Initialize analyzer
    analyzer = TrajectoryBifurcationAnalyzer(args.model, device=args.device)

    # Find hard problem
    log.info("Finding hard problems...")
    hard_problems = analyzer.find_hard_problems(dataset, num_samples=5, target_pass_rate=0.2)

    if not hard_problems:
        log.error("No hard problems found! Try different difficulty level or dataset.")
        return

    # Analyze first hard problem
    problem = hard_problems[0]
    log.info(f"Analyzing problem: {problem.get('unique_id', 'unknown')}")

    results = analyzer.run_bifurcation_analysis(
        problem,
        n_samples=args.n_samples,
        token_position=args.token_position,
        layer_idx=args.layer,
        temperature=args.temperature
    )

    # Save results
    results_file = output_dir / f"bifurcation_results_{args.model_display_name or 'model'}.json"
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        'hidden_2d': results['hidden_2d'].tolist(),
        'labels': results['labels'].tolist(),
        'greedy_2d': results['greedy_2d'].tolist(),
        'greedy_correct': bool(results['greedy_correct']),
        'explained_variance': results['explained_variance'].tolist(),
        'n_success': int(results['n_success']),
        'n_fail': int(results['n_fail']),
        'problem_id': results['problem_id'],
        'layer_idx': results['layer_idx'],
        'token_position': results['token_position'],
        'temperature': results['temperature']
    }

    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    log.info(f"Saved results to: {results_file}")

    # Plot
    plot_path = output_dir / f"bifurcation_plot_{args.model_display_name or 'model'}.png"
    analyzer.plot_bifurcation(results, str(plot_path), args.model_display_name)

    log.info("\nAnalysis complete!")
    log.info(f"Success rate: {results['n_success']}/{args.n_samples} = {results['n_success']/args.n_samples:.1%}")
    log.info(f"Greedy: {'CORRECT' if results['greedy_correct'] else 'INCORRECT'}")
    log.info(f"PCA variance explained: PC1={results['explained_variance'][0]:.1%}, PC2={results['explained_variance'][1]:.1%}")


if __name__ == '__main__':
    main()
