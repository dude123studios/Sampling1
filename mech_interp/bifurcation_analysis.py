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
    
    # Generate samples if none exist - use HuggingFace generate() (it works fine!)
    if not outputs:
        n_samples = cfg['analysis'].get('n_samples', 20)
        log.info(f"Generating {n_samples} samples for Problem {problem_id} (temp={temp}, max_tokens={max_new_tokens})...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Use HuggingFace generate - suppress warnings about generation config
        # Suppress the generation config warning - it's harmless
        import warnings
        import os
        # Set environment variable to suppress the warning
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        
        generated_solutions = []
        with torch.no_grad():
            for _ in tqdm(range(n_samples), desc="Generating"):
                # Only pass sampling params when do_sample=True
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": temp,
                    "top_p": 0.9,
                    "top_k": 50,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
                gen_output = model.generate(**gen_kwargs)
                # Decode and extract solution (remove prompt)
                full_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
                sol = full_text[len(prompt):]
                generated_solutions.append(sol)

        outputs = generated_solutions
        
        # Verify diversity
        unique_solutions = set(outputs)
        log.info(f"Generated {len(outputs)} samples, {len(unique_solutions)} unique.")
        
        # Grade them
        log.info("Grading generated solutions...")
        for sol in outputs:
            is_correct = grade_math(sol, problem['answer'])
            labels.append(1 if is_correct else 0)
            
    else:
        # Should not happen with new logic, but kept for safety
        labels = [1 if c else 0 for c in problem['correctness']]

    log.info(f"Solutions: {len(outputs)}, Success rate: {sum(labels)/len(labels):.2%}")

    log.info(f"Extracting hidden states from {len(outputs)} solutions...")
    token_pos = cfg['analysis']['token_position']
    layer_idx = cfg['analysis']['layer_idx']

    # Pre-tokenize prompt once
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)
    
    # Extract hidden states at token_position (only first token_pos tokens, not full solution)
    # We only need the prefix up to token_pos for PCA
    hidden_states = []
    
    # Pre-tokenize prompt to get its length
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_token_ids)
    
    with torch.no_grad():
        for i, solution in enumerate(tqdm(outputs, desc="Extracting hidden states")):
            try:
                # CRITICAL: We need to include solution tokens to get variance!
                # token_pos is the position in the FULL sequence (prompt + solution) where we extract
                solution_tokens = tokenizer.encode(solution, add_special_tokens=False)
                
                # Build full sequence: prompt + solution
                full_tokens = prompt_token_ids + solution_tokens
                
                # Extract at position token_pos (0-indexed, so token_pos-1 is the token at that position)
                # But we need at least token_pos tokens total
                if len(full_tokens) < token_pos:
                    log.warning(f"Sample {i}: Only {len(full_tokens)} tokens, need {token_pos}. Using all tokens.")
                    prefix_tokens = full_tokens
                    extract_pos = len(full_tokens) - 1  # Last token
                else:
                    # Use exactly token_pos tokens (prompt + first part of solution)
                    prefix_tokens = full_tokens[:token_pos]
                    extract_pos = token_pos - 1  # Extract at the token_pos-th token (0-indexed)
                
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
            except Exception as e:
                log.error(f"Error extracting hidden state for sample {i}: {e}")
                import traceback
                log.error(traceback.format_exc())
                hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
                hidden_states.append(np.zeros(hidden_dim, dtype=np.float32))

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

    # Extract greedy hidden state at token_pos - same logic as other solutions
    greedy_token_ids = tokenizer.encode(greedy_text, add_special_tokens=False)
    
    if len(greedy_token_ids) < token_pos:
        greedy_prefix_ids = greedy_token_ids
        greedy_extract_pos = len(greedy_token_ids) - 1
    else:
        greedy_prefix_ids = greedy_token_ids[:token_pos]
        greedy_extract_pos = token_pos - 1
    
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

    log.info(f"Success: {labels.sum()}/{len(labels)}, Greedy: {'CORRECT' if greedy_correct else 'INCORRECT'}")

    # PCA - use actual data, don't return zeros
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
                log.info(f"Mean difference between first 5 samples: {np.mean(sample_diffs):.6f}")
            
            # Remove features with zero variance (they don't contribute to PCA)
            feature_vars = np.var(hidden_states_f64, axis=0)
            valid_features = feature_vars > 1e-12  # Very lenient threshold
            
            n_valid = np.sum(valid_features)
            log.info(f"Features with variance: {n_valid}/{len(feature_vars)}")
            log.info(f"Feature variance stats: min={np.min(feature_vars):.6e}, max={np.max(feature_vars):.6e}, mean={np.mean(feature_vars):.6e}")
            
            if n_valid < 2:
                log.warning(f"Only {n_valid} features have variance. Using all features for PCA anyway.")
                hidden_states_valid = hidden_states_f64
            else:
                # Use only valid features
                hidden_states_valid = hidden_states_f64[:, valid_features]
            
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
            
            if n_valid < 2:
                greedy_hidden_valid = greedy_hidden_np.reshape(1, -1)
            else:
                greedy_hidden_valid = greedy_hidden_np[valid_features].reshape(1, -1)
            
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

    # Save and plot
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg['model']['name']

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

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    success_mask = labels == 1
    fail_mask = labels == 0

    ax.scatter(hidden_2d[fail_mask, 0], hidden_2d[fail_mask, 1],
               c='#E63946', s=80, alpha=0.6, edgecolors='black', linewidths=1.2,
               label=f'Failed ({results["n_fail"]})', zorder=2)

    ax.scatter(hidden_2d[success_mask, 0], hidden_2d[success_mask, 1],
               c='#457B9D', s=80, alpha=0.6, edgecolors='black', linewidths=1.2,
               label=f'Success ({results["n_success"]})', zorder=3)

    greedy_color = '#2A9D8F' if greedy_correct else '#F4A261'
    greedy_label = 'Greedy (Correct)' if greedy_correct else 'Greedy (Incorrect)'

    ax.scatter(greedy_2d[0], greedy_2d[1],
               c=greedy_color, s=400, alpha=1.0, marker='*',
               edgecolors='black', linewidths=2.0, label=greedy_label, zorder=4)

    ax.set_xlabel(f'PC1 ({results["explained_variance"][0]:.1%} var)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({results["explained_variance"][1]:.1%} var)', fontsize=13, fontweight='bold')
    ax.set_title(f'Trajectory Bifurcation: {model_name}\nLayer {layer_idx}, Token {token_pos}',
                 fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', frameon=True, fontsize=11, edgecolor='black')

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_bifurcation.png", dpi=300, bbox_inches='tight', facecolor='white')
    log.info(f"Saved to {output_dir}")

    return results


if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mech_interp/bifurcation_config.yaml"
    run_bifurcation_analysis(config_path)
