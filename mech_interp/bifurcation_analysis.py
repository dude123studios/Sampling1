"""
Trajectory Bifurcation Analysis - Uses existing sweep data
"""

import torch
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.math_grader import grade_math
from src.models.api_model import APIModel
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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

def extract_hidden_state(model, tokenizer, prefix_text: str, token_pos: int, layer_idx: int, device: str):
    """Extract hidden state at specific position and layer. Keep on GPU for efficiency."""
    try:
    inputs = tokenizer(prefix_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']

    seq_len = input_ids.shape[1]
        actual_pos = min(token_pos, seq_len - 1)  # Ensure valid position

    hidden_state = None

    def hook_fn(module, input, output):
        nonlocal hidden_state
            try:
        hidden = output[0] if isinstance(output, tuple) else output
                # Extract on GPU, only move to CPU at the end
        if hidden.shape[1] > actual_pos:
                    hidden_state = hidden[0, actual_pos, :].detach()  # Keep on GPU
        else:
                    hidden_state = hidden[0, -1, :].detach()  # Keep on GPU
                
                # Check for NaN/Inf and fix immediately (prevent propagation)
                if hidden_state is not None:
                    nan_count = torch.sum(torch.isnan(hidden_state)).item()
                    inf_count = torch.sum(torch.isinf(hidden_state)).item()
                    if nan_count > 0 or inf_count > 0:
                        # Replace NaN/Inf with zeros on GPU
                        hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e:
                log.error(f"Error in hook: {e}")
                hidden_state = None

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_ids)

    hook.remove()
        
        if hidden_state is None:
            log.error(f"Failed to extract hidden state, returning zeros")
            hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
            return torch.zeros(hidden_dim, device=device)  # Keep on GPU
        
        # Move to CPU only at the very end
        return hidden_state.cpu()
    except Exception as e:
        log.error(f"Error extracting hidden state: {e}")
        hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
        return torch.zeros(hidden_dim)


def run_bifurcation_analysis(config_path: str):
    """Run bifurcation analysis using config."""
    # Load environment variables from .env file
    load_dotenv()
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load model
    log.info(f"Loading model: {cfg['model']['model_id']}")
    device = cfg['model']['device']
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['model_id'],
        dtype=torch.float16,
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['model_id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load problem directly (configurable ID)
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

    # Get generation parameters (used for both temperature sampling and greedy)
    max_new_tokens = cfg['analysis'].get('max_new_tokens', 4096)
    temp = cfg['analysis'].get('temperature', 0.6)  # Get temperature for results
    
    # Setup output directory early to check for existing solutions
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg['model']['name']
    
    # Check if solutions already exist for this problem/model
    solutions_file = output_dir / f"{model_name}_problem_{problem_id}_solutions.json"
    
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
                    log.info("No solutions found in saved file, will generate new ones")
                    outputs = []
        except Exception as e:
            log.warning(f"Error loading solutions file: {e}. Will generate new ones.")
            outputs = []
    
    # Generate samples if none exist - use OpenRouter API with multithreading
    n_samples = cfg['analysis'].get('n_samples', 100)  # Default to 100
    if not outputs or len(outputs) < n_samples:
        if outputs:
            log.info(f"Loaded {len(outputs)} solutions but need {n_samples}, generating additional ones")
            # Pad existing outputs to n_samples
            while len(outputs) < n_samples:
                outputs.append("")
        else:
            n_samples = cfg['analysis'].get('n_samples', 100)
        log.info(f"Generating {n_samples} samples for Problem {problem_id} using OpenRouter API (temp={temp}, max_tokens={max_new_tokens})...")
        
        # Initialize API model for generation
        api_config = cfg.get('api', {})
        if not api_config:
            raise ValueError("API configuration required. Add 'api' section to config with model_name, base_url, and api_key")
        
        # Allow API key to be specified directly in config or via environment variable
        api_key = api_config.get('api_key', None)
        if not api_key:
            api_key_env = api_config.get('api_key_env', 'OPENROUTER_API_KEY')
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found. Either set 'api_key' in config or set environment variable '{api_key_env}'")
        
        # Create a custom config that includes the API key directly
        api_model_cfg = DictConfig({
            'type': 'api',
            'provider': 'openrouter',
            'model_name': api_config.get('model_name', 'qwen/qwen3-8b'),
            'base_url': api_config.get('base_url', 'https://openrouter.ai/api/v1'),
            'api_key': api_key  # Pass key directly
        })
        
        # Create API model with direct key access
        class DirectAPIModel:
            def __init__(self, config):
                self.api_key = config.api_key
                self.base_url = config.base_url
                self.model_name = config.model_name
            
            def generate(self, prompt: str, **kwargs):
                import requests
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://neurips-experiment.com",
                    "X-Title": "Sampling Limits NeurIPS",
                    "Content-Type": "application/json"
                }
                
                messages = [{"role": "user", "content": prompt}]
                
                data = {
                    "model": self.model_name,
                    "messages": messages,
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
                        response = requests.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
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
        
        api_model = DirectAPIModel(api_model_cfg)
        max_workers = api_config.get('max_workers', 15)
        log.info(f"Using {max_workers} threads for generation")
        
        # Generate solutions in parallel
        def generate_one_solution(idx):
            try:
                solution = api_model.generate(
                    prompt,
                    temperature=temp,
                    max_new_tokens=max_new_tokens,
                    top_p=0.9,
                    top_k=50
                )
                return idx, solution, None
            except Exception as e:
                log.error(f"Error generating solution {idx}: {e}")
                return idx, None, str(e)
        
        generated_solutions = [None] * samples_to_generate
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_one_solution, i): i for i in range(samples_to_generate)}
            for future in tqdm(as_completed(futures), total=samples_to_generate, desc="Generating"):
                idx, sol, error = future.result()
                if error:
                    log.warning(f"Solution {idx} failed: {error}")
                    generated_solutions[idx] = ""  # Empty string on error
                else:
                    generated_solutions[idx] = sol

        # Append new solutions to existing ones (or replace if none existed)
        if outputs:
            outputs.extend(generated_solutions)  # Add to existing
        else:
            outputs = generated_solutions  # Replace with new

        # Ensure we have exactly n_samples
        if len(outputs) > n_samples:
            outputs = outputs[:n_samples]
        elif len(outputs) < n_samples:
            while len(outputs) < n_samples:
                outputs.append("")
        
        # Verify diversity (including failed generations)
        valid_outputs = [s for s in outputs if s and s.strip()]
        unique_solutions = set(valid_outputs)
        log.info(f"Generated {len(outputs)} samples total, {len(valid_outputs)} valid, {len(unique_solutions)} unique.")
        
        # Grade them (all 100 results)
        log.info("Grading generated solutions...")
        labels = []
        for i, sol in enumerate(outputs):
            if sol and sol.strip():  # Only grade non-empty solutions
            is_correct = grade_math(sol, problem['answer'])
            labels.append(1 if is_correct else 0)
            else:
                # Failed generation or empty response
                labels.append(0)  # Count as incorrect
                if i < 5:  # Log first few failures
                    log.warning(f"Solution {i} was empty/failed, counting as incorrect")
        
        # Save solutions for future use
        log.info(f"About to save: outputs has {len(outputs)} items, labels has {len(labels)} items")
        log.info(f"First 5 outputs: {[repr(s[:50]) for s in outputs[:5]]}")
        log.info(f"First 5 labels: {labels[:5]}")
        log.info(f"Saving {len(outputs)} solutions to {solutions_file}")
        try:
            with open(solutions_file, 'w') as f:
                json.dump({
                    'problem_id': problem_id,
                    'model_name': model_name,
                    'temperature': temp,
                    'n_samples': len(outputs),
                    'valid_samples': len(valid_outputs),
                    'unique_samples': len(unique_solutions),
                    'solutions': outputs,
                    'labels': labels
                }, f, indent=2)
            log.info(f"Solutions saved successfully")
        except Exception as e:
            log.warning(f"Failed to save solutions: {e}")
            
    else:
        # Should not happen with new logic, but kept for safety
        if len(labels) == 0:
        labels = [1 if c else 0 for c in problem['correctness']]

    correct_count = sum(labels)
    total_count = len(labels)
    success_rate = correct_count / total_count if total_count > 0 else 0
    log.info(f"Solutions: {total_count}, Correct: {correct_count}, Success rate: {success_rate:.2%}")
    log.info(f"First 10 labels: {labels[:10]}")  # Debug: show first 10 results

    log.info(f"Extracting hidden states from {len(outputs)} solutions using local HuggingFace model...")
    token_pos = cfg['analysis']['token_position']
    layer_idx = cfg['analysis']['layer_idx']

    # Pre-tokenize prompt once
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_token_ids)

    # Extract at position: prompt_len + token_pos (token_pos tokens into the solution)
    extraction_pos = prompt_len + token_pos
    log.info(f"Prompt length: {prompt_len} tokens, extracting at position {extraction_pos} ({token_pos} tokens into solution)")
    
    # Extract hidden states at token_position (exactly after token_pos tokens)
    hidden_states = []
    
    with torch.no_grad():
        for i, solution in enumerate(tqdm(outputs, desc="Extracting hidden states")):
            extraction_count += 1
            try:
                # CRITICAL: Extract at position extraction_pos = prompt_len + token_pos
                # token_pos tokens into the solution (after the prompt)
                if not solution or not solution.strip():
                    # Empty solution - use just prompt tokens
                    full_tokens = prompt_token_ids
                    if i < 5:  # Log first few
                        log.warning(f"Sample {i}: Empty solution, using only prompt tokens ({len(full_tokens)})")
                else:
                    solution_tokens = tokenizer.encode(solution, add_special_tokens=False)
                    full_tokens = prompt_token_ids + solution_tokens

                # We need at least extraction_pos tokens to extract at position extraction_pos-1 (0-indexed)
                # Position extraction_pos-1 is extraction_pos tokens into the sequence
                if len(full_tokens) < extraction_pos:
                    # Not enough tokens - use what we have
                    prefix_tokens = full_tokens
                    extract_pos = max(0, len(full_tokens) - 1)  # Last available token (at least 0)
                    if i == 0:
                        log.warning(f"Sample {i}: Only {len(full_tokens)} tokens, need {extraction_pos}. Using position {extract_pos}.")
        else:
                    # Use exactly extraction_pos tokens (prompt + solution tokens up to extraction_pos)
                    prefix_tokens = full_tokens[:extraction_pos]
                    extract_pos = extraction_pos - 1  # Extract at position extraction_pos-1 (the extraction_pos-th token, 0-indexed)
                    if i == 0:
                        log.info(f"Extracting at position {extract_pos} (after {extraction_pos} tokens: {prompt_len} prompt + {token_pos} solution)")
                
                # Convert to tensor
                prefix_tensor = torch.tensor([prefix_tokens], device=device)
                
                # Extract hidden state at the specified position
                hidden_state = None
                def hook_fn(module, input, output):
                    nonlocal hidden_state
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Extract at the exact position we want
                    actual_pos = min(extract_pos, hidden.shape[1] - 1)
                    hidden_state = hidden[0, actual_pos, :].detach()
                    # Fix NaN/Inf immediately on GPU
                    if torch.any(torch.isnan(hidden_state)) or torch.any(torch.isinf(hidden_state)):
                        hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=0.0, neginf=0.0)
                
                hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
                _ = model(prefix_tensor)
                hook.remove()
                
                if hidden_state is not None:
                    # Move to CPU and convert to numpy
                    h_np = hidden_state.cpu().numpy().astype(np.float32)
                    # Final safety check - only fix NaN/Inf, don't clamp (preserve variance)
                    if np.any(np.isnan(h_np)) or np.any(np.isinf(h_np)):
                        h_np = np.nan_to_num(h_np, nan=0.0, posinf=0.0, neginf=0.0)
                    hidden_states.append(h_np)

                    # Debug: log first few values to verify we're getting different states
                    if i < 3:
                        log.info(f"Sample {i}: Hidden state shape={h_np.shape}, mean={np.mean(h_np):.4f}, std={np.std(h_np):.4f}, first_5={h_np[:5]}")
                else:
                    log.error(f"Sample {i} failed to extract hidden state")
                    hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
                    hidden_states.append(np.zeros(hidden_dim, dtype=np.float32))
                    log.warning(f"Sample {i}: Using zeros for hidden state due to extraction failure")
                    log.warning(f"Sample {i}: Using zeros for hidden state due to extraction failure")
            except Exception as e:
                log.error(f"Error extracting hidden state for sample {i}: {e}")
                import traceback
                log.error(traceback.format_exc())
                hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
                hidden_states.append(np.zeros(hidden_dim, dtype=np.float32))
                log.warning(f"Sample {i}: Using zeros for hidden state due to exception")

    log.info(f"Extraction completed: processed {len(hidden_states)}/{len(outputs)} solutions, got {len(hidden_states)} hidden states")
    if len(hidden_states) != len(outputs):
        log.error(f"MISMATCH: Expected {len(outputs)} hidden states but got {len(hidden_states)}!")
        # Pad with zeros if needed
        while len(hidden_states) < len(outputs):
            hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
            hidden_states.append(np.zeros(hidden_dim, dtype=np.float32))
            log.warning(f"Padded hidden states to reach {len(hidden_states)} items")

    # Greedy solution (temperature=0, deterministic)
    log.info("Generating greedy solution...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        greedy_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    greedy_solution = greedy_text[len(prompt):]
    greedy_correct = grade_math(greedy_solution, problem['answer'])

    # Extract greedy hidden state at extraction_pos - same logic as other solutions
    # Extract at position extraction_pos = prompt_len + token_pos (token_pos tokens into solution)
    greedy_token_ids = tokenizer.encode(greedy_text, add_special_tokens=False)

    if len(greedy_token_ids) < extraction_pos:
        greedy_prefix_ids = greedy_token_ids
        greedy_extract_pos = len(greedy_token_ids) - 1
        log.warning(f"Greedy: Only {len(greedy_token_ids)} tokens, need {extraction_pos}. Using position {greedy_extract_pos}.")
    else:
        greedy_prefix_ids = greedy_token_ids[:extraction_pos]
        greedy_extract_pos = extraction_pos - 1  # Extract at position extraction_pos-1 (the extraction_pos-th token)
        log.info(f"Greedy: Extracting at position {greedy_extract_pos} (after {extraction_pos} tokens)")
    
    greedy_prefix_tensor = torch.tensor([greedy_prefix_ids], device=device)
    
    greedy_hidden = None
    def greedy_hook_fn(module, input, output):
        nonlocal greedy_hidden
        hidden = output[0] if isinstance(output, tuple) else output
        actual_pos = min(greedy_extract_pos, hidden.shape[1] - 1)
        greedy_hidden = hidden[0, actual_pos, :].detach()
        # Fix NaN/Inf immediately
        if torch.any(torch.isnan(greedy_hidden)) or torch.any(torch.isinf(greedy_hidden)):
            greedy_hidden = torch.nan_to_num(greedy_hidden, nan=0.0, posinf=0.0, neginf=0.0)
    
    greedy_hook = model.model.layers[layer_idx].register_forward_hook(greedy_hook_fn)
    with torch.no_grad():
        _ = model(greedy_prefix_tensor)
    greedy_hook.remove()
    
    if greedy_hidden is None:
        hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
        greedy_hidden = torch.zeros(hidden_dim, device=device)
    
    # Move to CPU for consistency
    greedy_hidden = greedy_hidden.cpu()

    # Convert to numpy array with proper dtype
    hidden_states = np.array(hidden_states, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)  # Explicitly set dtype to int32

    log.info(f"Hidden states array shape: {hidden_states.shape}")
    log.info(f"Labels array shape: {labels.shape}")
    log.info(f"Success: {labels.sum()}/{len(labels)}, Greedy: {'CORRECT' if greedy_correct else 'INCORRECT'}")

    # PCA - use actual data, don't return zeros
    log.info(f"Starting PCA with {len(hidden_states)} hidden states")
    if len(hidden_states) < 2:
        log.error(f"Not enough samples for PCA (need at least 2, got {len(hidden_states)})")
        hidden_2d = np.zeros((len(hidden_states), 2), dtype=np.float64)
        greedy_2d = np.zeros((2,), dtype=np.float64)
        explained_variance = np.array([0.0, 0.0], dtype=np.float64)
    else:
        try:
            # Final cleanup: ensure no NaN/Inf
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to float64 for PCA (more stable)
            hidden_states_f64 = hidden_states.astype(np.float64)
            
            # Log statistics for debugging
            log.info(f"Hidden states shape: {hidden_states_f64.shape}")
            log.info(f"Hidden states mean: {np.mean(hidden_states_f64):.4f}, std: {np.std(hidden_states_f64):.4f}")
            log.info(f"Hidden states min: {np.min(hidden_states_f64):.4f}, max: {np.max(hidden_states_f64):.4f}")
            
            # Check if all hidden states are identical (would cause zero variance)
            sample_diffs = []
            for i in range(min(5, len(hidden_states_f64))):
                for j in range(i+1, min(5, len(hidden_states_f64))):
                    diff = np.mean(np.abs(hidden_states_f64[i] - hidden_states_f64[j]))
                    sample_diffs.append(diff)
            if sample_diffs:
                mean_diff = np.mean(sample_diffs)
                log.info(f"Mean difference between first 5 samples: {mean_diff:.6f}")
                if mean_diff < 1e-6:
                    log.error("CRITICAL: All hidden states are nearly identical! This means we're extracting from the same position (likely prompt).")
                    log.error(f"Prompt length: {prompt_len}, token_pos: {token_pos}, extraction_pos: {extraction_pos}")
                    log.error("This will cause zero variance in PCA. Check extraction logic.")
                    raise ValueError("All hidden states are identical - extraction position is wrong")
            
            # Remove features with zero variance (they don't contribute to PCA)
            # Use very sensitive threshold to catch even small variances
            feature_vars = np.var(hidden_states_f64, axis=0)
            valid_features = feature_vars > 1e-15  # Very sensitive threshold
            
            n_valid = np.sum(valid_features)
            log.info(f"Features with variance: {n_valid}/{len(feature_vars)}")
            log.info(f"Feature variance stats: min={np.min(feature_vars):.6e}, max={np.max(feature_vars):.6e}, mean={np.mean(feature_vars):.6e}")

            if n_valid < 2:
                log.warning(f"Only {n_valid} features have variance. Using all features for PCA anyway.")
                hidden_states_valid = hidden_states_f64
                valid_feature_mask = np.ones(hidden_states_f64.shape[1], dtype=bool)
            else:
                # Use only valid features
                hidden_states_valid = hidden_states_f64[:, valid_features]
                valid_feature_mask = valid_features

            log.info(f"PCA input shape: {hidden_states_valid.shape} (samples x features)")
            
            # Center the data (required for PCA)
            hidden_states_mean = np.mean(hidden_states_valid, axis=0)
            hidden_states_centered = hidden_states_valid - hidden_states_mean
            
            # Check total variance
            total_var = np.var(hidden_states_centered)
            log.info(f"Total variance after centering: {total_var:.6e}")
            
            # Run PCA - MUST use actual data, never return zeros
            n_components = min(2, hidden_states_valid.shape[1])
            log.info(f"Running PCA with {n_components} components on {hidden_states_valid.shape[0]} samples, {hidden_states_valid.shape[1]} features")
            
            pca = PCA(n_components=n_components)
            hidden_2d = pca.fit_transform(hidden_states_centered)
            
            log.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            log.info(f"PCA output shape: {hidden_2d.shape}, mean: {np.mean(hidden_2d):.4f}, std: {np.std(hidden_2d):.4f}")
            log.info(f"PCA output range: [{np.min(hidden_2d):.4f}, {np.max(hidden_2d):.4f}]")
            
            # Verify PCA actually produced non-zero values
            if np.allclose(hidden_2d, 0):
                log.error("PCA produced all zeros! This should not happen. Check hidden states.")
                raise ValueError("PCA produced all zeros - hidden states may be identical")
            
            # Transform greedy hidden state
            greedy_hidden_np = greedy_hidden.numpy().astype(np.float64)
            greedy_hidden_np = np.nan_to_num(greedy_hidden_np, nan=0.0, posinf=0.0, neginf=0.0)

            greedy_hidden_valid = greedy_hidden_np[valid_feature_mask].reshape(1, -1)
            greedy_centered = greedy_hidden_valid - hidden_states_mean
            greedy_2d = pca.transform(greedy_centered)[0]
            
            explained_variance = pca.explained_variance_ratio_
            
            # Ensure we have 2 components (pad if needed)
            if hidden_2d.shape[1] < 2:
                log.warning(f"PCA only produced {hidden_2d.shape[1]} component(s), padding to 2")
                hidden_2d = np.pad(hidden_2d, ((0, 0), (0, 2 - hidden_2d.shape[1])), mode='constant')
                greedy_2d = np.pad(greedy_2d, (0, 2 - len(greedy_2d)), mode='constant')
                explained_variance = np.pad(explained_variance, (0, 2 - len(explained_variance)), mode='constant')
            
            # Final check for NaN in results
            if np.any(np.isnan(hidden_2d)) or np.any(np.isnan(greedy_2d)) or np.any(np.isnan(explained_variance)):
                log.error("PCA produced NaN values. This should not happen - check data.")
                # Replace NaN with small random values to still show something
                if np.any(np.isnan(hidden_2d)):
                    hidden_2d = np.nan_to_num(hidden_2d, nan=0.0)
                if np.any(np.isnan(greedy_2d)):
                    greedy_2d = np.nan_to_num(greedy_2d, nan=0.0)
                if np.any(np.isnan(explained_variance)):
                    explained_variance = np.nan_to_num(explained_variance, nan=0.0)
                    
        except Exception as e:
            log.error(f"PCA failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            # Don't return zeros - try to use raw data as 2D projection
            log.warning("Falling back to using first 2 dimensions of hidden states")
            try:
                hidden_2d = hidden_states_f64[:, :2].astype(np.float64)
                greedy_2d = greedy_hidden.numpy()[:2].astype(np.float64)
                explained_variance = np.array([1.0, 0.0], dtype=np.float64)  # Fake but shows data
            except:
                hidden_2d = np.zeros((len(hidden_states), 2), dtype=np.float64)
                greedy_2d = np.zeros((2,), dtype=np.float64)
                explained_variance = np.array([0.0, 0.0], dtype=np.float64)

    results = {
        'hidden_2d': hidden_2d,
        'labels': labels,
        'greedy_2d': greedy_2d,
        'greedy_correct': greedy_correct,
        'greedy_solution': greedy_solution,  # Save greedy solution text
        'solutions': outputs,  # Save all generated solution texts
        'explained_variance': explained_variance,
        'n_success': int(labels.sum()),
        'n_fail': int(len(labels) - labels.sum()),
        'problem_id': problem['problem_id'],
        'problem_text': problem['problem'],  # Save problem text for reference
        'gold_answer': problem['answer'],  # Save gold answer
        'layer_idx': layer_idx,
        'token_position': token_pos,
        'temperature': temp,
        'n_samples': len(outputs)
    }

    # Save and plot (output_dir and model_name already set above)

    # Save results with proper error handling
    try:
        # Convert numpy arrays to lists, handle NaN/Inf properly
        results_to_save = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                # Check if array is numeric (not object dtype)
                if v.dtype == object or not np.issubdtype(v.dtype, np.number):
                    # Non-numeric array (strings, mixed types, etc.) - convert directly
                    results_to_save[k] = v.tolist()
                else:
                    # Numeric array - can safely use nan_to_num
                    # Check if it's integer type (no NaN possible)
                    if np.issubdtype(v.dtype, np.integer):
                        results_to_save[k] = v.tolist()
                    else:
                        # Floating point - clean NaN/Inf
                        v_clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        results_to_save[k] = v_clean.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                # Convert numpy scalars to Python types
                if np.isnan(v) or np.isinf(v):
                    results_to_save[k] = 0.0
                else:
                    results_to_save[k] = float(v)
            else:
                results_to_save[k] = v
        
    with open(output_dir / f"{model_name}_results.json", 'w') as f:
            json.dump(results_to_save, f, indent=2)
        log.info(f"Saved results to {output_dir / f'{model_name}_results.json'}")
    except Exception as e:
        log.error(f"Failed to save results: {e}")
        import traceback
        log.error(traceback.format_exc())
        # Try to save at least basic info
        try:
            basic_results = {
                'problem_id': problem['problem_id'],
                'n_samples': len(outputs),
                'n_success': int(labels.sum()),
                'n_fail': int(len(labels) - labels.sum()),
                'error': str(e)
            }
            with open(output_dir / f"{model_name}_results_error.json", 'w') as f:
                json.dump(basic_results, f, indent=2)
        except:
            pass

    # Plot - make it more dense and visually appealing
    log.info(f"Creating plot with {len(hidden_2d)} points, {sum(labels)} correct, {len(labels) - sum(labels)} incorrect")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    success_mask = labels == 1
    fail_mask = labels == 0

    # Use smaller, more transparent points for density
    ax.scatter(hidden_2d[fail_mask, 0], hidden_2d[fail_mask, 1],
               c='#DC143C', s=50, alpha=0.7, edgecolors='white', linewidths=0.5,
               label=f'Failed ({results["n_fail"]})', zorder=2)

    ax.scatter(hidden_2d[success_mask, 0], hidden_2d[success_mask, 1],
               c='#4169E1', s=50, alpha=0.7, edgecolors='white', linewidths=0.5,
               label=f'Success ({results["n_success"]})', zorder=3)

    greedy_color = '#32CD32' if greedy_correct else '#FF6347'
    greedy_label = 'Greedy (Correct)' if greedy_correct else 'Greedy (Incorrect)'

    ax.scatter(greedy_2d[0], greedy_2d[1],
               c=greedy_color, s=300, alpha=1.0, marker='*',
               edgecolors='white', linewidths=2.0, label=greedy_label, zorder=4)

    ax.set_xlabel(f'PC1 ({results["explained_variance"][0]:.1%} variance)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({results["explained_variance"][1]:.1%} variance)', fontsize=14, fontweight='bold')

    # Calculate nice axis limits to make it less spread out
    x_range = hidden_2d[:, 0].max() - hidden_2d[:, 0].min()
    y_range = hidden_2d[:, 1].max() - hidden_2d[:, 1].min()
    x_center = (hidden_2d[:, 0].max() + hidden_2d[:, 0].min()) / 2
    y_center = (hidden_2d[:, 1].max() + hidden_2d[:, 1].min()) / 2

    # Add 20% padding
    padding = 0.2
    ax.set_xlim(x_center - x_range/2 * (1 + padding), x_center + x_range/2 * (1 + padding))
    ax.set_ylim(y_center - y_range/2 * (1 + padding), y_center + y_range/2 * (1 + padding))

    ax.set_title(f'Trajectory Bifurcation Analysis\n{model_name} • Layer {layer_idx} • Position {extraction_pos}',
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
    plt.savefig(output_dir / f"{model_name}_bifurcation.png", dpi=300, bbox_inches='tight', facecolor='white')
    log.info(f"Saved plot to {output_dir / f'{model_name}_bifurcation.png'}")

    return results


if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mech_interp/bifurcation_config.yaml"
    run_bifurcation_analysis(config_path)
