"""
Gradient Attribution Framework

Measures how sensitive the logit difference is to changes in layer activations
by computing gradients.

For each layer ℓ:
  ∇_{z^ℓ} (logit[token1] - logit[token2])

The L2 norm |∇z^ℓ|_2 indicates how much small changes in that layer's
activations affect the final decision.

Higher gradient norms indicate:
  - Greater instability/sensitivity at that layer
  - Critical decision points where small activation changes have large effects
"""

import torch
import json
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GradientAttributionExperiment:
    def __init__(self, config_path: str, impactful_positions_path: str):
        """Initialize gradient attribution experiment.

        Args:
            config_path: Path to gradient config YAML
            impactful_positions_path: Path to token_impact_results.json
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load branching points from token impact results
        with open(impactful_positions_path, 'r') as f:
            self.branching_data = json.load(f)

        self.device = self.config['model']['device']
        self.dtype = torch.float16 if self.config['model']['dtype'] == 'float16' else torch.float32

        # Load model and tokenizer
        log.info(f"Loading model: {self.config['model']['model_id']}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['model_id'],
            torch_dtype=self.dtype,
            device_map=self.device
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_id']
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Gradient config
        self.gradient_layers = self.config['gradient_analysis']['layers']

        # Output directory
        self.output_dir = Path(self.config.get('output', {}).get('base_dir', 'mech_interp/gradient_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_gradients(
        self,
        input_ids: torch.Tensor,
        position: int,
        token1: int,
        token2: int,
        layers: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute gradients of logit difference w.r.t. layer activations.

        Args:
            input_ids: Input sequence [1, seq_len]
            position: Token position to analyze
            token1: First token ID
            token2: Second token ID
            layers: Layers to compute gradients for

        Returns:
            Dict mapping layer -> {gradient_norm, gradient_vector}
        """
        # Storage for activations and gradients
        activations = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                # Store activation and enable gradient tracking
                act = output[0]  # [batch, seq, hidden]
                act.requires_grad_(True)
                act.retain_grad()
                activations[layer_idx] = act
                return act
            return hook

        # Register hooks
        hooks = []
        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)

        # Enable gradients
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]

        # Compute logit difference
        logit_diff = logits[0, token1] - logits[0, token2]

        # Backward to compute gradients
        logit_diff.backward()

        # Extract gradients
        results = {}
        for layer_idx in layers:
            if layer_idx in activations:
                act = activations[layer_idx]
                if act.grad is not None:
                    # Gradient at the position of interest
                    grad = act.grad[0, position, :]  # [hidden_size]

                    # Compute L2 norm
                    grad_norm = torch.norm(grad, p=2).item()

                    results[layer_idx] = {
                        'gradient_norm': grad_norm,
                        'gradient_mean': grad.mean().item(),
                        'gradient_std': grad.std().item(),
                        'gradient_max': grad.max().item(),
                        'gradient_min': grad.min().item()
                    }

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Clear gradients
        self.model.zero_grad()

        return results

    def compute_directional_similarity(
        self,
        grad1: torch.Tensor,
        grad2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two gradient vectors.

        Args:
            grad1: First gradient vector
            grad2: Second gradient vector

        Returns:
            Cosine similarity [-1, 1]
        """
        cos_sim = torch.nn.functional.cosine_similarity(
            grad1.unsqueeze(0),
            grad2.unsqueeze(0)
        )
        return cos_sim.item()

    def run_experiment(self):
        """Run gradient attribution analysis on branching points."""
        all_results = []

        # Process each branching point
        for entry in tqdm(self.branching_data, desc="Computing gradients"):
            problem_id = entry['problem_id']
            cutoff_position = entry['cutoff_position']
            top1_token = entry['top1_token']
            top2_token = entry['top2_token']

            log.info(f"Analyzing problem {problem_id}, position {cutoff_position}")

            # TODO: Need input_ids prefix up to cutoff position
            # This requires either storing in token_impact results or reloading

            result = {
                'problem_id': problem_id,
                'cutoff_position': cutoff_position,
                'top1_token': top1_token,
                'top2_token': top2_token,
                'gradient_analysis': {},
                'note': 'Need to implement sequence loading'
            }

            # Compute gradients
            # gradient_results = self.compute_gradients(
            #     input_ids, cutoff_position, top1_token, top2_token, self.gradient_layers
            # )
            # result['gradient_analysis'] = gradient_results

            all_results.append(result)

            # Clean up GPU memory
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

        # Save results
        output_file = self.output_dir / "gradient_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        log.info(f"Gradient results saved to {output_file}")

        # Compute aggregate statistics
        self.compute_statistics(all_results)

    def compute_statistics(self, results: List[Dict]):
        """Compute aggregate gradient statistics."""
        # Group gradient norms by layer
        layer_norms = {layer: [] for layer in self.gradient_layers}

        for result in results:
            grad_analysis = result.get('gradient_analysis', {})
            for layer, data in grad_analysis.items():
                layer_norms[int(layer)].append(data['gradient_norm'])

        # Compute statistics
        stats = {
            'mean_gradient_norm_per_layer': {},
            'std_gradient_norm_per_layer': {},
            'max_gradient_norm_per_layer': {}
        }

        for layer in self.gradient_layers:
            if layer_norms[layer]:
                norms = np.array(layer_norms[layer])
                stats['mean_gradient_norm_per_layer'][layer] = float(np.mean(norms))
                stats['std_gradient_norm_per_layer'][layer] = float(np.std(norms))
                stats['max_gradient_norm_per_layer'][layer] = float(np.max(norms))

        # Identify layers with highest sensitivity
        if stats['mean_gradient_norm_per_layer']:
            most_sensitive = sorted(
                stats['mean_gradient_norm_per_layer'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            stats['most_sensitive_layers'] = most_sensitive

        # Save statistics
        output_file = self.output_dir / "gradient_statistics.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        log.info(f"Gradient statistics saved to {output_file}")

        if 'most_sensitive_layers' in stats:
            log.info("Most sensitive layers (by mean gradient norm):")
            for layer, norm in stats['most_sensitive_layers']:
                log.info(f"  Layer {layer}: {norm:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Gradient Attribution Analysis")
    parser.add_argument("--config", type=str,
                       default="configs/mech_interp/dla_patching_config.yaml",
                       help="Path to gradient config file")
    parser.add_argument("--branching-points", type=str,
                       default="mech_interp/token_impact_results/token_impact_results.json",
                       help="Path to token impact results")

    args = parser.parse_args()

    experiment = GradientAttributionExperiment(args.config, args.branching_points)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
