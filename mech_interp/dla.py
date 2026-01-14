"""
Direct Logit Attribution (DLA) Framework

Decomposes the logit difference between two tokens into layer-wise contributions
using residual stream analysis.

For each layer ℓ:
  DLA_ℓ = u^T · Δz^ℓ
where:
  - u = W_U[token1] - W_U[token2] (unembedding direction difference)
  - Δz^ℓ = z^ℓ - z^(ℓ-1) (residual stream delta at layer ℓ)

This identifies which layers contribute most to preferring token1 over token2.
"""

import torch
import json
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DLAExperiment:
    def __init__(self, config_path: str, impactful_positions_path: str):
        """Initialize DLA experiment.

        Args:
            config_path: Path to DLA config YAML
            impactful_positions_path: Path to token_impact_results.json with branching points
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

        # Get unembedding matrix
        self.W_U = self.model.lm_head.weight  # [vocab_size, hidden_size]

        # Number of layers
        self.num_layers = len(self.model.model.layers)

        # Output directory
        self.output_dir = Path(self.config.get('output', {}).get('base_dir', 'mech_interp/dla_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_residual_streams(
        self,
        input_ids: torch.Tensor,
        layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """
        Get residual stream activations at each layer.

        Args:
            input_ids: Input token IDs [1, seq_len]
            layers: Which layers to extract from

        Returns:
            Dict mapping layer -> residual stream [seq_len, hidden_size]
        """
        residual_streams = {}

        # Hook to capture residual stream outputs
        def hook_fn(layer_idx):
            def hook(module, input, output):
                # output[0] is hidden states [batch, seq, hidden]
                hidden = output[0]
                residual_streams[layer_idx] = hidden[0].detach().cpu()  # [seq, hidden]
            return hook

        # Register hooks
        hooks = []
        for layer_idx in layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return residual_streams

    def compute_dla(
        self,
        input_ids: torch.Tensor,
        position: int,
        token1: int,
        token2: int
    ) -> Dict[int, float]:
        """
        Compute Direct Logit Attribution for all layers.

        Args:
            input_ids: Input sequence [1, seq_len]
            position: Token position to analyze
            token1: First token ID (typically top1, "correct")
            token2: Second token ID (typically top2, "incorrect")

        Returns:
            Dict mapping layer -> DLA score
        """
        # Get residual streams at all layers
        all_layers = list(range(self.num_layers))
        residual_streams = self.get_residual_streams(input_ids, all_layers)

        # Get unembedding direction difference: u = W_U[token1] - W_U[token2]
        u = self.W_U[token1] - self.W_U[token2]  # [hidden_size]

        # Compute DLA for each layer
        dla_scores = {}

        # Layer 0 delta is from embedding to layer 0 output
        # For simplicity, we'll compute deltas between consecutive layers
        prev_z = None

        for layer_idx in all_layers:
            if layer_idx not in residual_streams:
                continue

            # Current residual stream at the position of interest
            z_curr = residual_streams[layer_idx][position]  # [hidden_size]

            if prev_z is None:
                # First layer: delta from input embeddings
                # We approximate this as the layer 0 output itself
                delta_z = z_curr
            else:
                # Delta between consecutive layers
                delta_z = z_curr - prev_z

            # DLA_ℓ = u^T · Δz^ℓ
            dla_score = torch.dot(u, delta_z).item()
            dla_scores[layer_idx] = dla_score

            prev_z = z_curr

        return dla_scores

    def run_experiment(self):
        """Run DLA analysis on branching points."""
        results = []

        # Process each branching point from token impact results
        for entry in tqdm(self.branching_data, desc="Computing DLA"):
            problem_id = entry['problem_id']
            cutoff_position = entry['cutoff_position']
            top1_token = entry['top1_token']
            top2_token = entry['top2_token']

            # We need the prefix sequence up to cutoff
            # This should be stored in the token impact results or we need to reload
            # For now, we'll skip if we don't have the sequence
            # TODO: May need to reload problem data if sequence not in results

            log.info(f"Analyzing problem {problem_id}, position {cutoff_position}")

            # Placeholder: would need actual input_ids here
            # This requires either storing them in token_impact results
            # or reloading the problem data

            results.append({
                'problem_id': problem_id,
                'cutoff_position': cutoff_position,
                'top1_token': top1_token,
                'top2_token': top2_token,
                'dla_scores': {},  # Will be filled when we have input_ids
                'note': 'Need to implement input sequence loading'
            })

            # Clean up GPU memory
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

        # Save results
        output_file = self.output_dir / "dla_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        log.info(f"DLA results saved to {output_file}")

        # Compute aggregate statistics
        self.compute_statistics(results)

    def compute_statistics(self, results: List[Dict]):
        """Compute aggregate DLA statistics."""
        # Group DLA scores by layer across all branching points
        layer_scores = {i: [] for i in range(self.num_layers)}

        for result in results:
            dla_scores = result.get('dla_scores', {})
            for layer, score in dla_scores.items():
                layer_scores[int(layer)].append(score)

        # Compute mean and variance per layer
        stats = {
            'mean_dla_per_layer': {},
            'variance_per_layer': {},
            'std_per_layer': {}
        }

        for layer in range(self.num_layers):
            if layer_scores[layer]:
                scores = np.array(layer_scores[layer])
                stats['mean_dla_per_layer'][layer] = float(np.mean(scores))
                stats['variance_per_layer'][layer] = float(np.var(scores))
                stats['std_per_layer'][layer] = float(np.std(scores))

        # Focus analysis on specific bands
        abstraction_layers = self.config['dla_config']['analysis_bands']['abstraction']
        commitment_layers = self.config['dla_config']['analysis_bands']['commitment']

        stats['abstraction_band'] = {
            'layers': abstraction_layers,
            'mean_dla': float(np.mean([stats['mean_dla_per_layer'].get(l, 0) for l in abstraction_layers])),
            'mean_variance': float(np.mean([stats['variance_per_layer'].get(l, 0) for l in abstraction_layers]))
        }

        stats['commitment_band'] = {
            'layers': commitment_layers,
            'mean_dla': float(np.mean([stats['mean_dla_per_layer'].get(l, 0) for l in commitment_layers])),
            'mean_variance': float(np.mean([stats['variance_per_layer'].get(l, 0) for l in commitment_layers]))
        }

        # Save statistics
        output_file = self.output_dir / "dla_statistics.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        log.info(f"DLA statistics saved to {output_file}")
        log.info(f"Abstraction band (layers {abstraction_layers[0]}-{abstraction_layers[-1]}): "
                f"mean DLA = {stats['abstraction_band']['mean_dla']:.4f}")
        log.info(f"Commitment band (layers {commitment_layers[0]}-{commitment_layers[-1]}): "
                f"mean DLA = {stats['commitment_band']['mean_dla']:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Direct Logit Attribution Analysis")
    parser.add_argument("--config", type=str,
                       default="configs/mech_interp/dla_patching_config.yaml",
                       help="Path to DLA config file")
    parser.add_argument("--branching-points", type=str,
                       default="mech_interp/token_impact_results/token_impact_results.json",
                       help="Path to token impact results with branching points")

    args = parser.parse_args()

    experiment = DLAExperiment(args.config, args.branching_points)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
