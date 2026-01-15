
"""
Random Baseline Analysis: DLA, Gradient Attribution, and Patching

This script performs mechanistic interpretability analyses (DLA, Gradient Attribution, Patching)
but instead of using the "most impactful tokens" identified by previous steps, it selects
a random position between index 8 and 16 (inclusive) on all hard problem rollouts.

This serves as a control/baseline experiment.
"""

import torch
import json
import yaml
import logging
import random
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

class RandomBaselineExperiment:
    def __init__(self, config_path: str, modes: List[str] = None):
        """Initialize experiment with configuration."""
        log.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.modes = modes if modes else ["dla", "gradient", "patching"]
        log.info(f"Running modes: {self.modes}")

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

        # Unembedding matrix for DLA
        self.W_U = self.model.lm_head.weight
        self.num_layers = len(self.model.model.layers)
        
        # Output directory
        self.output_dir = Path("mech_interp/random_baseline_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration for specific analyses
        self.dla_layers = "all" # All layers
        self.patch_layers = self.config['patching']['layers']
        self.gradient_layers = self.config['gradient_analysis']['layers']

    def load_sweep_data(self) -> List[Dict]:
        """Load Level 5 problems from sweep results (recursive search)."""
        sweep_dir = Path("results/sweeps")
        log.info(f"Looking for Level 5 problems (recursively) in {sweep_dir.absolute()}")
        
        problems = []
        
        if sweep_dir.exists():
            # Use rglob to find log.jsonl in any subdirectory
            for log_file in sweep_dir.rglob("log.jsonl"):
                path_str = str(log_file)
                
                # Filter for temperature 0.6
                if "temp0.6" not in path_str:
                    continue
                    
                # Filter for correct model if specified in config
                model_filter = self.config.get('data_source', {}).get('model_name')
                if not model_filter:
                    # Try to guess from model config name
                    model_filter = self.config.get('model', {}).get('name')

                if model_filter:
                    # Extract just the directory name to check model
                    dir_name = log_file.parent.name

                    # If filter is "qwen3-8b", exclude "deepseek-qwen3-8b"
                    if model_filter == "qwen3-8b":
                        if "deepseek" in dir_name:
                            log.debug(f"Skipping {dir_name}: contains 'deepseek' but filter is qwen3-8b")
                            continue
                        if "qwen3-8b" not in dir_name:
                            log.debug(f"Skipping {dir_name}: doesn't contain 'qwen3-8b'")
                            continue

                    # If filter is "deepseek-qwen3-8b", only include that
                    elif model_filter == "deepseek-qwen3-8b":
                        if "deepseek-qwen3-8b" not in dir_name:
                            log.debug(f"Skipping {dir_name}: doesn't contain 'deepseek-qwen3-8b'")
                            continue

                    # Generic filter: must contain the filter string
                    else:
                        if model_filter not in dir_name:
                            log.debug(f"Skipping {dir_name}: doesn't contain '{model_filter}'")
                            continue

                log.info(f"Found log file: {log_file}")
                with open(log_file, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            entry = json.loads(line)
                            if entry.get('type') == 'summary': continue
                            if entry.get('level') == 5:
                                if 'outputs' in entry and entry['outputs']:
                                    problems.append({
                                        'problem_id': entry.get('dataset_id', entry.get('id')),
                                        'problem': entry.get('original_prompt', ''),
                                        'outputs': entry['outputs'],
                                        'level': entry['level']
                                    })
                        except json.JSONDecodeError:
                            continue
                            
        log.info(f"Loaded {len(problems)} Level 5 problems")
        return problems

    def get_residual_streams(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Capture residual streams (output of each layer)."""
        residual_streams = {}
        hooks = []
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                # Handle tuple vs tensor output
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                # hidden should be [batch, seq, hidden]
                # Log shape on first layer to debug
                if layer_idx == 0:
                   try:
                       log.info(f"Layer 0 output shape: {hidden.shape}, type: {type(hidden)}")
                   except:
                       pass

                # Store [seq, hidden] for first batch element
                residual_streams[layer_idx] = hidden[0].detach().cpu()
            return hook
            
        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            hooks.append(layer.register_forward_hook(hook_fn(i)))
            
        with torch.no_grad():
            self.model(input_ids)
            
        for h in hooks: h.remove()
        return residual_streams

    def run_dla(self, input_ids: torch.Tensor, position: int, top1: int, top2: int) -> Dict[int, float]:
        """Run Direct Logit Attribution at the specific position."""
        residual_streams = self.get_residual_streams(input_ids)
        u = self.W_U[top1] - self.W_U[top2] # [hidden_size]
        
        dla_scores = {}
        prev_z = None
        
        # position indexing: residual_stream[layer] is [seq_len, hidden]
        # We want the state *output* by the layer at `position`
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in residual_streams: continue
            
            z_curr = residual_streams[layer_idx][position].to(self.device).float()
            
            # Additional debug for shape
            if z_curr.dim() == 0:
                 # If it collapsed to scalar, something is wrong with extraction
                 log.error(f"Layer {layer_idx} extracted z_curr is scalar: {z_curr}")
            
            if prev_z is None:
                delta_z = z_curr # Layer 0 accumulation
            else:
                delta_z = z_curr - prev_z
            
            # Flatten to ensure 1D
            u_flat = u.flatten().float()
            dz_flat = delta_z.flatten()
            
            if u_flat.shape != dz_flat.shape:
                log.error(f"Shape Mismatch L{layer_idx}: u={u_flat.shape}, dz={dz_flat.shape}")
                
            dla_scores[layer_idx] = torch.dot(u_flat, dz_flat).item()
            prev_z = z_curr
            
        return dla_scores

    def run_gradient(self, input_ids: torch.Tensor, position: int, top1: int, top2: int) -> Dict[int, float]:
        """Compute gradient of logit diff w.r.t layer activations."""
        activations = {}
        hooks = []
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                    
                act.requires_grad_(True)
                act.retain_grad()
                activations[layer_idx] = act
                return output # Return original structure
            return hook
            
        for layer_idx in self.gradient_layers:
            layer = self.model.model.layers[layer_idx]
            hooks.append(layer.register_forward_hook(hook_fn(layer_idx)))
            
        self.model.zero_grad()
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        logit_diff = logits[0, top1] - logits[0, top2]
        logit_diff.backward()
        
        results = {}
        for layer_idx in self.gradient_layers:
            if layer_idx in activations and activations[layer_idx].grad is not None:
                # Gradient at the position of interest (last token)
                grad = activations[layer_idx].grad[0, -1, :]
                results[layer_idx] = torch.norm(grad, p=2).item()
                
        for h in hooks: h.remove()
        self.model.zero_grad()
        return results

    def patch_and_measure(self, prefix_ids: torch.Tensor,
                         layer: int,
                         top1_token: int,
                         top2_token: int) -> float:
        """
        Patch activations at a specific layer and measure causal effect.

        Strategy:
        1. Run forward pass on prefix, cache activation at this layer
        2. Append top2_token and run forward to get baseline logits for next position
        3. Patch in the cached activation and measure how logits change

        Args:
            prefix_ids: The prefix sequence (before appending any token)
            layer: Which layer to patch
            top1_token: The preferred token ID
            top2_token: The alternative token ID

        Returns:
            Causal effect: how much patching increases preference for top1 over top2
        """
        # 1. Cache activation from prefix at last position for this layer
        cached_act = None
        def cache_hook(module, input, output):
            nonlocal cached_act
            if isinstance(output, tuple):
                cached_act = output[0][:, -1:, :].detach().clone()  # [1, 1, hidden]
            else:
                cached_act = output[:, -1:, :].detach().clone()

        target_layer = self.model.model.layers[layer]
        h = target_layer.register_forward_hook(cache_hook)
        with torch.no_grad():
            _ = self.model(prefix_ids)
        h.remove()

        if cached_act is None:
            return 0.0

        # 2. Append top2_token and get baseline logits (without patching)
        seq_with_top2 = torch.cat([prefix_ids, torch.tensor([[top2_token]], device=self.device)], dim=1)
        with torch.no_grad():
            out_baseline = self.model(seq_with_top2)
            logits_baseline = out_baseline.logits[0, -1]  # Logits for next token position
            baseline_diff = logits_baseline[top1_token] - logits_baseline[top2_token]

        # 3. Patch: inject cached activation at the last position of this layer
        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                current_act = output[0]
                rest = output[1:]
                is_tuple = True
            else:
                current_act = output
                is_tuple = False

            # Replace activation at last position with cached activation from prefix
            current_act[:, -1:, :] = cached_act.to(current_act.device)

            if is_tuple:
                return (current_act,) + rest
            return current_act

        h = target_layer.register_forward_hook(patch_hook)
        with torch.no_grad():
            out_patched = self.model(seq_with_top2)
        h.remove()

        # 4. Measure patched logits
        logits_patched = out_patched.logits[0, -1]
        patched_diff = logits_patched[top1_token] - logits_patched[top2_token]

        # Return how much patching changes the preference (positive = helps top1)
        return (patched_diff - baseline_diff).item()


    def run_experiment(self):
        problems = self.load_sweep_data()
        
        if not problems:
            log.warning("No problems loaded! Check your data source directory.")
            return

        all_results = []
        checkpoint_interval = 10
        checkpoint_file = self.output_dir / "random_baseline_partial.json"
        
        # Get token range from config, default to [8, 16]
        token_range = self.config.get('random_baseline', {}).get('token_range', [8, 16])
        min_pos, max_pos = token_range[0], token_range[1]
        
        for idx, prob in enumerate(tqdm(problems, desc="Processing Random Baseline")):
            # Setup sequence
            ref_text = prob['problem'] + prob['outputs'][0] # Use first rollout
            input_ids = self.tokenizer.encode(ref_text, return_tensors='pt').to(self.device)
            
            seq_len = input_ids.shape[1]
            # Ensure we have enough tokens to even reach min_pos
            if seq_len < min_pos + 2: continue
                
            # Cap max index at actual sequence length (-2 for next token pred)
            # Use configurable max_pos
            curr_max_idx = min(max_pos, seq_len - 2)
            
            if curr_max_idx < min_pos: continue
            
            cutoff_pos = random.randint(min_pos, curr_max_idx)
            prefix_ids = input_ids[:, :cutoff_pos+1] 
            
            # Get Top1, Top2
            with torch.no_grad():
                out = self.model(prefix_ids)
                logits = out.logits[0, -1, :]
                topk = torch.topk(logits, 2)
                top1_token = topk.indices[0].item()
                top2_token = topk.indices[1].item()
                
            result_entry = {
                'problem_id': prob['problem_id'],
                'cutoff_pos': cutoff_pos,
                'top1_token': top1_token,
                'top2_token': top2_token,
            }

            # --- DLA ---
            if "dla" in self.modes:
                dla_res = self.run_dla(prefix_ids, -1, top1_token, top2_token)
                result_entry['dla'] = dla_res
            
            # --- Gradient ---
            if "gradient" in self.modes:
                grad_res = self.run_gradient(prefix_ids, -1, top1_token, top2_token)
                result_entry['gradient'] = grad_res
            
            # --- Patching ---
            if "patching" in self.modes:
                patch_res = {}
                for layer in self.patch_layers:
                    effect = self.patch_and_measure(prefix_ids, layer, top1_token, top2_token)
                    patch_res[layer] = effect
                result_entry['patching'] = patch_res
            
            all_results.append(result_entry)
            
            if (idx + 1) % checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                log.info(f"Checkpointed {len(all_results)} results")
            
        with open(self.output_dir / "random_baseline_full.json", 'w') as f:
            json.dump(all_results, f, indent=2)
            
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            
        log.info("Analysis complete.")
        log.info(f"Saved {len(all_results)} entries to {self.output_dir / 'random_baseline_full.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Random Baseline Analysis")
    parser.add_argument("--config", type=str, 
                       default="configs/mech_interp/dla_patching_config.yaml",
                       help="Path to config file")
    parser.add_argument("--modes", nargs="+", choices=["dla", "gradient", "patching"],
                        default=["dla", "gradient", "patching"],
                        help="Analysis modes to run (default: all)")
    
    args = parser.parse_args()
    
    exp = RandomBaselineExperiment(args.config, modes=args.modes)
    exp.run_experiment()
