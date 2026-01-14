
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

class RandomBaselineExperiment:
    def __init__(self, config_path: str):
        """Initialize experiment with configuration."""
        log.info(f"Loading config from {config_path}")
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
        """Load Level 5 problems from sweep results (same as token_impact.py)."""
        sweep_dir = Path("results/sweeps")
        # Hardcoded for now based on what worked in token_impact.py or config defaults
        # We'll try to find sweep directories dynamically if config path is vague
        
        log.info(f"Looking for Level 5 problems in {sweep_dir}")
        problems = []
        
        if sweep_dir.exists():
            for subdir in sweep_dir.glob("*"):
                log_file = subdir / "log.jsonl"
                if not log_file.exists():
                    continue
                
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
                # Output[0] is hidden state [1, seq, hidden]
                residual_streams[layer_idx] = output[0][0].detach().cpu()
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
        # Because we predict `position+1` from state at `position`.
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in residual_streams: continue
            
            z_curr = residual_streams[layer_idx][position].to(self.device).float()
            
            if prev_z is None:
                delta_z = z_curr # Layer 0 accumulation
            else:
                delta_z = z_curr - prev_z
                
            dla_scores[layer_idx] = torch.dot(u.float(), delta_z).item()
            prev_z = z_curr
            
        return dla_scores

    def run_gradient(self, input_ids: torch.Tensor, position: int, top1: int, top2: int) -> Dict[int, float]:
        """Compute gradient of logit diff w.r.t layer activations."""
        activations = {}
        hooks = []
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                act = output[0]
                act.requires_grad_(True)
                act.retain_grad()
                activations[layer_idx] = act
                return act
            return hook
            
        for layer_idx in self.gradient_layers:
            layer = self.model.model.layers[layer_idx]
            hooks.append(layer.register_forward_hook(hook_fn(layer_idx)))
            
        self.model.zero_grad()
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1, :]
        
        # Logit diff: we want to explain the prediction at the END of input_ids
        # input_ids has length `position + 1` (tokens 0..position)
        # So we predict token at `position + 1`.
        # Wait, if `position` is the last token index, `logits[:, -1]` is correct.
        
        logit_diff = logits[0, top1] - logits[0, top2]
        logit_diff.backward()
        
        results = {}
        for layer_idx in self.gradient_layers:
            if layer_idx in activations and activations[layer_idx].grad is not None:
                # Gradient at the last position (which caused the logit)
                # Act shape: [1, seq, hidden]
                # We usually want the gradient at the token position that generated the logit.
                grad = activations[layer_idx].grad[0, -1, :]
                results[layer_idx] = torch.norm(grad, p=2).item()
                
        for h in hooks: h.remove()
        self.model.zero_grad()
        return results

    def patch_and_measure(self, input_ids_base: torch.Tensor, 
                         input_ids_source: torch.Tensor,
                         layer: int,
                         patch_type: str = "residual_stream") -> float:
        """Patch from source into base and measure next-token logit distribution shift."""
        # Simple attribute patching:
        # We have seq_base (which predicts BaseNext) and seq_source (which predicts SourceNext).
        # We patch from Source -> Base.
        # We want to see if Base outputs start looking like Source.
        # Metric: Logit(SourceNext) - Logit(BaseNext) on the Patched run.
        
        # 1. Get Source Activation
        source_act = None
        def cache_hook(module, input, output):
            nonlocal source_act
            source_act = output[0].detach().clone()
        
        target_layer = self.model.model.layers[layer]
        h = target_layer.register_forward_hook(cache_hook)
        with torch.no_grad():
            self.model(input_ids_source)
        h.remove()
        
        if source_act is None: return 0.0
        
        # 2. Get Targets (what tokens would strict Source and Base predict?)
        with torch.no_grad():
            logits_source = self.model(input_ids_source).logits[0, -1]
            token_source = logits_source.argmax().item()
            
            logits_base = self.model(input_ids_base).logits[0, -1]
            token_base = logits_base.argmax().item()
            
        if token_source == token_base:
            return 0.0 # No divergence to patch
            
        # 3. Patch Source -> Base
        def patch_hook(module, input, output):
            # Replace activation
            # source_act is [1, seq, hidden]
            # output[0] is [1, seq, hidden]
            # We replace the whole sequence? Or just the last token?
            # Standard patching usually replaces specific positions.
            # Here sequences are almost identical (1 token diff).
            # Let's replace the last token activation.
            current_act = output[0]
            current_act[:, -1, :] = source_act[:, -1, :].to(current_act.device)
            return (current_act,) + output[1:]
            
        h = target_layer.register_forward_hook(patch_hook)
        with torch.no_grad():
            out_patched = self.model(input_ids_base)
        h.remove()
        
        logits_patched = out_patched.logits[0, -1]
        
        # Metric: Logit(TokenSource) - Logit(TokenBase)
        # High value = Successfully made Base think like Source
        metric = logits_patched[token_source] - logits_patched[token_base]
        return metric.item()


    def run_experiment(self):
        problems = self.load_sweep_data()
        all_results = []
        
        for prob in tqdm(problems, desc="Processing Random Baseline"):
            # Setup sequence
            ref_text = prob['problem'] + prob['outputs'][0] # Use first rollout
            input_ids = self.tokenizer.encode(ref_text, return_tensors='pt').to(self.device)
            
            # 1. Random Cutoff [8, 16]
            max_idx = min(16, input_ids.shape[1] - 2)
            if max_idx < 8: continue # Too short
            
            cutoff_pos = random.randint(8, max_idx)
            
            # "Prefix" is up to cutoff. The token AT cutoff is the one we just processed.
            # We want to predict the NEXT token.
            # input_ids[:, :cutoff_pos+1] means indices 0..cutoff_pos. Length is cutoff_pos+1.
            # We analyze the decision made at the end of this sequence.
            
            prefix_ids = input_ids[:, :cutoff_pos+1] 
            
            # Get Top1, Top2 from this prefix
            with torch.no_grad():
                out = self.model(prefix_ids)
                logits = out.logits[0, -1, :]
                topk = torch.topk(logits, 2)
                top1_token = topk.indices[0].item()
                top2_token = topk.indices[1].item()
                
            # --- DLA ---
            dla_res = self.run_dla(prefix_ids, -1, top1_token, top2_token)
            
            # --- Gradient ---
            grad_res = self.run_gradient(prefix_ids, -1, top1_token, top2_token)
            
            # --- Patching ---
            # Construct Sequence 1 (Prefix + Top1) and Sequence 2 (Prefix + Top2)
            # Patching happens at the NEXT step (cutoff_pos + 1)
            # We want to see if patching 'Source' (Top1) into 'Base' (Top2)
            # makes Base predict Top1's next token.
            
            seq1 = torch.cat([prefix_ids, torch.tensor([[top1_token]], device=self.device)], dim=1)
            seq2 = torch.cat([prefix_ids, torch.tensor([[top2_token]], device=self.device)], dim=1)
            
            patch_res = {}
            for layer in self.patch_layers:
                # Patching Source (Seq1) -> Base (Seq2)
                # If Seq1 and Seq2 diverge in prediction, we measure recovery.
                effect = self.patch_and_measure(seq2, seq1, layer)
                patch_res[layer] = effect
            
            all_results.append({
                'problem_id': prob['problem_id'],
                'cutoff_pos': cutoff_pos,
                'top1_token': top1_token,
                'top2_token': top2_token,
                'dla': dla_res,
                'gradient': grad_res,
                'patching': patch_res
            })
            
        # Save aggregated
        with open(self.output_dir / "random_baseline_full.json", 'w') as f:
            json.dump(all_results, f, indent=2)
            
        # Compute minimal stats for checking
        log.info("Analysis complete.")
        log.info(f"Saved {len(all_results)} entries to {self.output_dir / 'random_baseline_full.json'}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/mech_interp/dla_patching_config.yaml"
        
    exp = RandomBaselineExperiment(config_path)
    exp.run_experiment()
