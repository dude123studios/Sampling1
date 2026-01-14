"""
Activation Patching Framework

Performs causal interventions by replacing activations from the "incorrect" (top2) path
with activations from the "correct" (top1) path at specific layers.

Measures the change in logit difference to identify which layers are critical
for the model's decision between token1 and token2.

Patch types:
  - residual_stream: Full residual stream after layer
  - attention_output: Just the attention output
  - mlp_output: Just the MLP output
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


class ActivationPatchingExperiment:
    def __init__(self, config_path: str, impactful_positions_path: str):
        """Initialize activation patching experiment.

        Args:
            config_path: Path to patching config YAML
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

        # Number of layers
        self.num_layers = len(self.model.model.layers)

        # Patching config
        self.patch_types = self.config['patching']['patch_types']
        self.patch_layers = self.config['patching']['layers']

        # Output directory
        self.output_dir = Path(self.config.get('output', {}).get('base_dir', 'mech_interp/patching_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for activations
        self.activation_cache = {}

    def cache_activations(
        self,
        input_ids: torch.Tensor,
        layers: List[int],
        patch_type: str
    ) -> Dict[int, torch.Tensor]:
        """
        Cache activations for later patching.

        Args:
            input_ids: Input token IDs [1, seq_len]
            layers: Which layers to cache
            patch_type: Type of activation to cache

        Returns:
            Dict mapping layer -> activations
        """
        cached = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                if patch_type == "residual_stream":
                    # Full output of the layer (residual stream)
                    cached[layer_idx] = output[0].detach().clone()  # [batch, seq, hidden]
                elif patch_type == "attention_output":
                    # This would need access to attention module specifically
                    # For now, placeholder
                    cached[layer_idx] = output[0].detach().clone()
                elif patch_type == "mlp_output":
                    # This would need access to MLP module specifically
                    # For now, placeholder
                    cached[layer_idx] = output[0].detach().clone()
            return hook

        # Register hooks
        hooks = []
        for layer_idx in layers:
            if patch_type == "residual_stream":
                layer = self.model.model.layers[layer_idx]
            elif patch_type == "attention_output":
                # Hook attention output specifically
                layer = self.model.model.layers[layer_idx].self_attn
            elif patch_type == "mlp_output":
                # Hook MLP output specifically
                layer = self.model.model.layers[layer_idx].mlp
            else:
                continue

            hook = layer.register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return cached

    def get_logit_difference(
        self,
        input_ids: torch.Tensor,
        token1: int,
        token2: int
    ) -> float:
        """
        Compute logit difference: logit[token1] - logit[token2] at the last position.

        Args:
            input_ids: Input sequence [1, seq_len]
            token1: First token ID
            token2: Second token ID

        Returns:
            Logit difference
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            logit1 = logits[0, token1].item()
            logit2 = logits[0, token2].item()

            return logit1 - logit2

    def patch_and_measure(
        self,
        input_ids: torch.Tensor,
        patch_layer: int,
        patch_type: str,
        source_activations: Dict[int, torch.Tensor],
        token1: int,
        token2: int
    ) -> float:
        """
        Apply activation patch at a specific layer and measure logit difference.

        Args:
            input_ids: Input sequence to run with patching [1, seq_len]
            patch_layer: Layer to patch
            patch_type: Type of patch
            source_activations: Cached activations to patch in
            token1: First token ID
            token2: Second token ID

        Returns:
            Logit difference after patching
        """
        if patch_layer not in source_activations:
            log.warning(f"No cached activation for layer {patch_layer}")
            return self.get_logit_difference(input_ids, token1, token2)

        patched_activation = source_activations[patch_layer]

        # Hook to replace activation
        def patch_hook(module, input, output):
            # Replace output with cached activation
            if isinstance(output, tuple):
                return (patched_activation.to(self.device),) + output[1:]
            else:
                return patched_activation.to(self.device)

        # Register patch hook
        if patch_type == "residual_stream":
            layer = self.model.model.layers[patch_layer]
        elif patch_type == "attention_output":
            layer = self.model.model.layers[patch_layer].self_attn
        elif patch_type == "mlp_output":
            layer = self.model.model.layers[patch_layer].mlp
        else:
            return self.get_logit_difference(input_ids, token1, token2)

        hook = layer.register_forward_hook(patch_hook)

        # Forward pass with patch
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            logit1 = logits[0, token1].item()
            logit2 = logits[0, token2].item()
            logit_diff = logit1 - logit2

        # Remove hook
        hook.remove()

        return logit_diff

    def run_patching_analysis(
        self,
        input_ids_top1: torch.Tensor,
        input_ids_top2: torch.Tensor,
        token1: int,
        token2: int,
        patch_type: str
    ) -> Dict[int, Dict[str, float]]:
        """
        Run full patching analysis for a branching point.

        Args:
            input_ids_top1: Sequence with top1 token at cutoff
            input_ids_top2: Sequence with top2 token at cutoff
            token1: Top1 token ID
            token2: Top2 token ID
            patch_type: Type of patching

        Returns:
            Dict mapping layer -> {baseline, patched, effect}
        """
        # Cache activations from top1 path (the "correct" path)
        top1_activations = self.cache_activations(
            input_ids_top1,
            self.patch_layers,
            patch_type
        )

        # Get baseline logit difference for top2 path (the "incorrect" path)
        baseline_logit_diff = self.get_logit_difference(input_ids_top2, token1, token2)

        # Patch each layer and measure effect
        results = {}
        for layer in tqdm(self.patch_layers, desc=f"Patching {patch_type}", leave=False):
            # Patch top1 activation into top2 path
            patched_logit_diff = self.patch_and_measure(
                input_ids_top2,
                layer,
                patch_type,
                top1_activations,
                token1,
                token2
            )

            # Effect: how much did patching change the logit difference
            effect = patched_logit_diff - baseline_logit_diff

            results[layer] = {
                'baseline_logit_diff': baseline_logit_diff,
                'patched_logit_diff': patched_logit_diff,
                'patch_effect': effect
            }

            # Clean up GPU memory
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

        return results

    def run_experiment(self):
        """Run activation patching on branching points."""
        all_results = []

        # Process each branching point
        for entry in tqdm(self.branching_data, desc="Patching analysis"):
            problem_id = entry['problem_id']
            cutoff_position = entry['cutoff_position']
            top1_token = entry['top1_token']
            top2_token = entry['top2_token']

            log.info(f"Analyzing problem {problem_id}, position {cutoff_position}")

            # TODO: Need to reconstruct input_ids_top1 and input_ids_top2
            # This requires either storing sequences in token_impact results
            # or reloading problem data

            result = {
                'problem_id': problem_id,
                'cutoff_position': cutoff_position,
                'top1_token': top1_token,
                'top2_token': top2_token,
                'patching_results': {},
                'note': 'Need to implement sequence loading'
            }

            # For each patch type, run patching analysis
            # for patch_type in self.patch_types:
            #     patch_results = self.run_patching_analysis(
            #         input_ids_top1, input_ids_top2, top1_token, top2_token, patch_type
            #     )
            #     result['patching_results'][patch_type] = patch_results

            all_results.append(result)

            # Clean up GPU memory
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

        # Save results
        output_file = self.output_dir / "patching_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        log.info(f"Patching results saved to {output_file}")

        # Compute aggregate statistics
        self.compute_statistics(all_results)

    def compute_statistics(self, results: List[Dict]):
        """Compute aggregate patching statistics."""
        stats = {}

        for patch_type in self.patch_types:
            layer_effects = {layer: [] for layer in self.patch_layers}

            for result in results:
                patch_results = result.get('patching_results', {}).get(patch_type, {})
                for layer, data in patch_results.items():
                    layer_effects[int(layer)].append(data['patch_effect'])

            # Mean effect per layer
            mean_effects = {}
            for layer in self.patch_layers:
                if layer_effects[layer]:
                    mean_effects[layer] = float(np.mean(layer_effects[layer]))

            stats[patch_type] = {
                'mean_effect_per_layer': mean_effects,
                'most_important_layers': sorted(
                    mean_effects.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]  # Top 5 most impactful layers
            }

        # Save statistics
        output_file = self.output_dir / "patching_statistics.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        log.info(f"Patching statistics saved to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Activation Patching Analysis")
    parser.add_argument("--config", type=str,
                       default="configs/mech_interp/dla_patching_config.yaml",
                       help="Path to patching config file")
    parser.add_argument("--branching-points", type=str,
                       default="mech_interp/token_impact_results/token_impact_results.json",
                       help="Path to token impact results")

    args = parser.parse_args()

    experiment = ActivationPatchingExperiment(args.config, args.branching_points)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
