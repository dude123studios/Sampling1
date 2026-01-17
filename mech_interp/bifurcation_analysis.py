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
        
        # Use HuggingFace generate - it handles temperature/top_p correctly
        # The warnings are just warnings, the parameters DO work
        generated_solutions = []
        with torch.no_grad():
            for _ in tqdm(range(n_samples), desc="Generating"):
                gen_output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
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
    
    # Build full texts (prompt + solution) for all samples
    # Then extract hidden states after all generation is done
    hidden_states = []
    
    with torch.no_grad():
        for i, solution in enumerate(tqdm(outputs, desc="Extracting hidden states")):
            try:
                # Build full text: prompt + solution
                full_text = prompt + solution
                
                # Tokenize and get prefix up to token_pos
                full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                if len(full_tokens) >= token_pos:
                    prefix_tokens = full_tokens[:token_pos]
                else:
                    prefix_tokens = full_tokens
                
                # Convert to tensor
                prefix_tensor = torch.tensor([prefix_tokens], device=device)
                
                # Extract hidden state using hook
                hidden_state = None
                def hook_fn(module, input, output):
                    nonlocal hidden_state
                    hidden = output[0] if isinstance(output, tuple) else output
                    extract_pos = min(token_pos - 1, hidden.shape[1] - 1)
                    hidden_state = hidden[0, extract_pos, :].detach().cpu()
                    # Fix NaN/Inf immediately
                    if torch.any(torch.isnan(hidden_state)) or torch.any(torch.isinf(hidden_state)):
                        hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=0.0, neginf=0.0)
                
                hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
                _ = model(prefix_tensor)
                hook.remove()
                
                if hidden_state is not None:
                    h_np = hidden_state.numpy()
                    # Final safety check
                    if np.any(np.isnan(h_np)) or np.any(np.isinf(h_np)):
                        h_np = np.nan_to_num(h_np, nan=0.0, posinf=0.0, neginf=0.0)
                    hidden_states.append(h_np)
                else:
                    log.warning(f"Sample {i} failed to extract hidden state")
                    hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
                    hidden_states.append(np.zeros(hidden_dim))
            except Exception as e:
                log.error(f"Error extracting hidden state for sample {i}: {e}")
                hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
                hidden_states.append(np.zeros(hidden_dim))

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

    greedy_inputs = tokenizer(greedy_text, return_tensors="pt")
    if greedy_inputs['input_ids'].shape[1] >= token_pos:
        greedy_prefix_ids = greedy_inputs['input_ids'][0, :token_pos]
        greedy_prefix_text = tokenizer.decode(greedy_prefix_ids, skip_special_tokens=True)
    else:
        greedy_prefix_text = greedy_text

    greedy_hidden = extract_hidden_state(model, tokenizer, greedy_prefix_text, token_pos - 1, layer_idx, device)

    # PCA
    hidden_states = np.array(hidden_states)
    labels = np.array(labels)

    log.info(f"Success: {labels.sum()}/{len(labels)}, Greedy: {'CORRECT' if greedy_correct else 'INCORRECT'}")

    # Check for variance and handle potential overflow
    if len(hidden_states) > 1:
        # Convert to numpy and check for issues
        hidden_states = np.array(hidden_states)
        
        # Check for NaN or Inf values
        nan_mask = np.isnan(hidden_states)
        inf_mask = np.isinf(hidden_states)
        if np.any(nan_mask) or np.any(inf_mask):
            log.warning(f"Hidden states contain NaN/Inf: {np.sum(nan_mask)} NaN, {np.sum(inf_mask)} Inf. Replacing with zeros.")
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values to prevent overflow
        max_val = np.percentile(np.abs(hidden_states), 99.9)
        if max_val > 1000:
            log.warning(f"Clipping extreme hidden state values (max={max_val:.2f})")
            hidden_states = np.clip(hidden_states, -1000, 1000)
        
        # Check variance
        variance = np.var(hidden_states)
        if variance < 1e-9:
            log.warning("Hidden states have near-zero variance. All samples likely identical.")
            hidden_2d = np.zeros((len(hidden_states), 2))
            greedy_2d = np.zeros((2,))
            explained_variance = np.array([0.0, 0.0])
        else:
            try:
                # Center the data to prevent overflow
                hidden_states_mean = np.mean(hidden_states, axis=0, dtype=np.float64)
                hidden_states_centered = hidden_states.astype(np.float64) - hidden_states_mean
                
                # Use float64 for PCA to prevent overflow
                pca = PCA(n_components=2)
                hidden_2d = pca.fit_transform(hidden_states_centered)
                
                # Transform greedy hidden state
                greedy_hidden_np = greedy_hidden.numpy().astype(np.float64).reshape(1, -1)
                greedy_centered = greedy_hidden_np - hidden_states_mean
                greedy_2d = pca.transform(greedy_centered)[0]
                
                explained_variance = pca.explained_variance_ratio_
                
                # Check for NaN in results
                if np.any(np.isnan(hidden_2d)) or np.any(np.isnan(greedy_2d)):
                    log.error("PCA produced NaN values. Using zeros.")
                    hidden_2d = np.zeros((len(hidden_states), 2))
                    greedy_2d = np.zeros((2,))
                    explained_variance = np.array([0.0, 0.0])
            except Exception as e:
                log.error(f"PCA failed: {e}")
                import traceback
                log.error(traceback.format_exc())
                hidden_2d = np.zeros((len(hidden_states), 2))
                greedy_2d = np.zeros((2,))
                explained_variance = np.array([0.0, 0.0])
    else:
        log.warning("Not enough samples for PCA (need at least 2)")
        hidden_2d = np.zeros((len(hidden_states), 2))
        greedy_2d = np.zeros((2,))
        explained_variance = np.array([0.0, 0.0])

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
        # Convert numpy arrays to lists, handle NaN/Inf
        results_to_save = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                # Replace NaN/Inf with None for JSON compatibility
                v_clean = np.nan_to_num(v, nan=None, posinf=None, neginf=None)
                results_to_save[k] = v_clean.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                results_to_save[k] = float(v) if not np.isnan(v) and not np.isinf(v) else None
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
