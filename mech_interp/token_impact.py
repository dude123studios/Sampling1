"""
Token Impact Identification Experiment

Identifies which token positions are most impactful by:
1. Prefilling up to a cutoff point from 0.6 temp trajectories
2. Forcing either 1st or 2nd most likely token at cutoff
3. Generating 128 more tokens
4. Comparing cosine similarity of layer activations between the two paths

This identifies branching points where token choice matters most.
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
import glob

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TokenImpactExperiment:
    def __init__(self, config_path: str):
        """Initialize the experiment."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

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

        # Output directory
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_sweep_data(self) -> List[Dict]:
        """Load Level 5 problems from sweep results."""
        sweep_dir = Path(self.config['data_source']['sweep_dir'])
        model_name = self.config['data_source']['model_name']
        temperature = self.config['data_source']['temperature']
        level_filter = self.config['data_source']['level_filter']

        # Find matching sweep directories
        pattern = f"{model_name}_temp{temperature}_*"
        matching_dirs = list(sweep_dir.glob(f"*/{pattern}"))

        if not matching_dirs:
            log.error(f"No sweep results found matching pattern: {pattern}")
            return []

        log.info(f"Found {len(matching_dirs)} matching sweep directories")

        # Load log.jsonl from each directory
        problems = []
        for sweep_result_dir in matching_dirs:
            log_file = sweep_result_dir / "log.jsonl"
            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    # Skip summary entries
                    if entry.get('type') == 'summary':
                        continue

                    # Filter for Level 5
                    if entry.get('level') == level_filter:
                        # Need: problem text, outputs (generations)
                        if 'outputs' in entry and entry['outputs']:
                            problems.append({
                                'problem_id': entry.get('dataset_id', entry.get('id')),
                                'problem': entry.get('original_prompt', ''),
                                'outputs': entry['outputs'],
                                'level': entry['level']
                            })

        log.info(f"Loaded {len(problems)} Level {level_filter} problems")
        return problems

    def get_layer_activations(
        self,
        input_ids: torch.Tensor,
        layers: List[int],
        positions: List[int]
    ) -> Dict[int, torch.Tensor]:
        """
        Get residual stream activations at specific layers and positions.

        Args:
            input_ids: Input token IDs [1, seq_len]
            layers: Which layers to extract activations from
            positions: Which token positions to extract

        Returns:
            Dict mapping layer -> activations at positions [len(positions), hidden_size]
        """
        activations = {}

        # Hook to capture activations
        def hook_fn(layer_idx):
            def hook(module, input, output):
                # output[0] is hidden states [batch, seq, hidden]
                hidden = output[0]
                # Extract positions we care about
                for pos in positions:
                    if pos < hidden.shape[1]:
                        if layer_idx not in activations:
                            activations[layer_idx] = []
                        activations[layer_idx].append(hidden[0, pos, :].detach().cpu())
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

        # Stack activations
        result = {}
        for layer_idx in layers:
            if layer_idx in activations:
                result[layer_idx] = torch.stack(activations[layer_idx])  # [num_positions, hidden_size]

        return result

    def generate_with_forced_token(
        self,
        input_ids: torch.Tensor,
        forced_token_id: int,
        continuation_length: int,
        temperature: float,
        top_p: float
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate continuation with a forced first token.

        Returns:
            Generated token IDs and the list of generated tokens
        """
        # Start with forced token
        current_ids = torch.cat([
            input_ids,
            torch.tensor([[forced_token_id]], device=self.device)
        ], dim=1)

        generated_tokens = [forced_token_id]

        # Generate remaining tokens
        for _ in range(continuation_length - 1):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]

                # Apply temperature
                logits = logits / temperature

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated_tokens.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids, generated_tokens

    def compute_cosine_similarity(
        self,
        acts1: torch.Tensor,
        acts2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two activation tensors."""
        # Flatten if needed
        if len(acts1.shape) > 1:
            acts1 = acts1.flatten()
        if len(acts2.shape) > 1:
            acts2 = acts2.flatten()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            acts1.unsqueeze(0),
            acts2.unsqueeze(0)
        )
        return cos_sim.item()

    def run_experiment(self):
        """Run the token impact identification experiment."""
        problems = self.load_sweep_data()

        if not problems:
            log.error("No problems loaded. Exiting.")
            return

        cutoff_positions = self.config['token_forcing']['cutoff_positions']
        continuation_length = self.config['token_forcing']['continuation_length']
        temperature = self.config['token_forcing']['temperature']
        top_p = self.config['token_forcing']['top_p']

        layers = self.config['layer_analysis']['layers']
        averaging_positions = self.config['layer_analysis']['averaging_positions']

        results = []

        for problem in tqdm(problems, desc="Analyzing problems"):
            problem_id = problem['problem_id']
            # Use first output as the reference trajectory
            if not problem['outputs']:
                continue

            reference_output = problem['outputs'][0]

            # Tokenize reference
            ref_tokens = self.tokenizer.encode(
                problem['problem'] + reference_output,
                return_tensors='pt'
            ).to(self.device)

            # Test each cutoff position
            for cutoff_pos in cutoff_positions:
                if cutoff_pos >= ref_tokens.shape[1]:
                    continue  # Skip if cutoff exceeds reference length

                # Prefix up to cutoff
                prefix = ref_tokens[:, :cutoff_pos]

                # Get top 2 tokens at cutoff position
                with torch.no_grad():
                    outputs = self.model(prefix)
                    logits = outputs.logits[:, -1, :]
                    top_k = torch.topk(logits, k=2, dim=-1)
                    top1_token = top_k.indices[0, 0].item()
                    top2_token = top_k.indices[0, 1].item()

                # Generate with top1 token
                _, gen_top1 = self.generate_with_forced_token(
                    prefix, top1_token, continuation_length, temperature, top_p
                )

                # Generate with top2 token
                _, gen_top2 = self.generate_with_forced_token(
                    prefix, top2_token, continuation_length, temperature, top_p
                )

                # Reconstruct full sequences for layer analysis
                seq_top1 = torch.cat([
                    prefix,
                    torch.tensor([gen_top1], device=self.device)
                ], dim=1)

                seq_top2 = torch.cat([
                    prefix,
                    torch.tensor([gen_top2], device=self.device)
                ], dim=1)

                # Compute activations for both paths
                # We want positions relative to the start of the 128 continuation
                continuation_start = cutoff_pos + 1
                abs_positions = [continuation_start + p for p in averaging_positions
                                 if continuation_start + p < min(seq_top1.shape[1], seq_top2.shape[1])]

                acts_top1 = self.get_layer_activations(seq_top1, layers, abs_positions)
                acts_top2 = self.get_layer_activations(seq_top2, layers, abs_positions)

                # Compute cosine similarities per layer
                layer_similarities = {}
                for layer in layers:
                    if layer in acts_top1 and layer in acts_top2:
                        # Average activations across positions
                        avg_act1 = acts_top1[layer].mean(dim=0)  # [hidden_size]
                        avg_act2 = acts_top2[layer].mean(dim=0)

                        cos_sim = self.compute_cosine_similarity(avg_act1, avg_act2)
                        layer_similarities[layer] = cos_sim

                # Compute divergence score (1 - similarity)
                avg_similarity = np.mean(list(layer_similarities.values())) if layer_similarities else 1.0
                divergence_score = 1.0 - avg_similarity

                results.append({
                    'problem_id': problem_id,
                    'cutoff_position': cutoff_pos,
                    'top1_token': top1_token,
                    'top2_token': top2_token,
                    'top1_text': self.tokenizer.decode([top1_token]),
                    'top2_text': self.tokenizer.decode([top2_token]),
                    'layer_similarities': layer_similarities,
                    'divergence_score': divergence_score
                })

                log.info(f"Problem {problem_id}, cutoff {cutoff_pos}: "
                        f"divergence={divergence_score:.4f}")

                # Clean up GPU memory periodically
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()

        # Save results
        output_file = self.output_dir / "token_impact_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        log.info(f"Results saved to {output_file}")

        # Find most impactful positions
        self.analyze_results(results)

    def analyze_results(self, results: List[Dict]):
        """Analyze results to find most impactful token positions."""
        # Group by problem and find positions with highest divergence
        problem_divergences = {}

        for result in results:
            pid = result['problem_id']
            if pid not in problem_divergences:
                problem_divergences[pid] = []

            problem_divergences[pid].append({
                'cutoff_position': result['cutoff_position'],
                'divergence_score': result['divergence_score']
            })

        # For each problem, find top divergence positions
        impactful_positions = {}
        for pid, divs in problem_divergences.items():
            # Sort by divergence score
            sorted_divs = sorted(divs, key=lambda x: x['divergence_score'], reverse=True)
            # Take top 3 most impactful positions
            impactful_positions[pid] = [d['cutoff_position'] for d in sorted_divs[:3]]

        # Save impactful positions
        output_file = self.output_dir / "impactful_positions.json"
        with open(output_file, 'w') as f:
            json.dump(impactful_positions, f, indent=2)

        log.info(f"Impactful positions saved to {output_file}")
        log.info(f"Found impactful positions for {len(impactful_positions)} problems")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Token Impact Identification")
    parser.add_argument("--config", type=str,
                       default="configs/mech_interp/token_impact_config.yaml",
                       help="Path to config file")

    args = parser.parse_args()

    experiment = TokenImpactExperiment(args.config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
