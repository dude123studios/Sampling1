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


def load_sweep_problem(sweep_dir: Path, temperature: str, model_filter: str):
    """Load a hard problem from sweep results."""
    log.info(f"Loading from {sweep_dir} (temp={temperature}, model={model_filter})")

    # Look for specific directory pattern
    pattern = f"*{model_filter}*temp{temperature}*"
    all_matching = list(sweep_dir.glob(pattern))

    # Filter strictly - if qwen3-8b, exclude deepseek
    matching_dirs = []
    for d in all_matching:
        if model_filter == "qwen3-8b" and "deepseek" in d.name:
            continue
        matching_dirs.append(d)

    if not matching_dirs:
        log.error(f"No matching directories for pattern: {pattern}")
        return None

    log.info(f"Found {len(matching_dirs)} matching directories")

    for run_dir in matching_dirs:
        log_file = run_dir / 'log.jsonl'
        if not log_file.exists():
            log.warning(f"No log file in {run_dir.name}")
            continue

        log.info(f"Searching in: {run_dir.name}")

        problem_count = 0
        level5_count = 0
        hard_count = 0

        with open(log_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'summary':
                        continue

                    problem_count += 1

                    # Check for level 5 problem with 1-2/5 correct
                    if 'outputs' in entry and 'scores' in entry:
                        level = entry.get('level')
                        scores = entry['scores']
                        n = len(scores)
                        c = sum(scores)

                        if level == 5:
                            level5_count += 1

                        # Level 5 with 1 or 2 correct out of 5
                        if level == 5 and c in [1, 2] and n == 5:
                            hard_count += 1
                            dataset_id = entry.get('dataset_id', '')
                            log.info(f"Found level 5 hard problem: {dataset_id} ({c}/5 correct)")

                            # Load actual problem from dataset
                            from datasets import load_dataset
                            dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

                            # Find the problem in dataset
                            problem_text = None
                            for item in dataset:
                                if item.get('unique_id') == dataset_id:
                                    problem_text = item['problem']
                                    break

                            if not problem_text:
                                log.warning(f"Could not find problem text for {dataset_id} in dataset")
                                continue

                            log.info(f"Successfully loaded problem from dataset")
                            return {
                                'problem_id': dataset_id,
                                'problem': problem_text,
                                'answer': entry.get('gold', ''),
                                'outputs': entry['outputs'],
                                'correctness': [bool(s) for s in scores],
                                'level': level
                            }
                except Exception as e:
                    log.debug(f"Error parsing entry: {e}")
                    continue

        log.info(f"Scanned {problem_count} problems, {level5_count} level 5, {hard_count} hard (1-2/5)")

    log.error("No suitable problem found")

    log.error("No suitable problem found")
    return None


def extract_hidden_state(model, tokenizer, prefix_text: str, token_pos: int, layer_idx: int, device: str):
    """Extract hidden state at specific position and layer."""
    inputs = tokenizer(prefix_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']

    seq_len = input_ids.shape[1]
    if seq_len < token_pos:
        token_pos = seq_len - 1

    hidden_state = None

    def hook_fn(module, input, output):
        nonlocal hidden_state
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_state = hidden[0, token_pos, :].detach().cpu()

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_ids)

    hook.remove()
    return hidden_state


def run_bifurcation_analysis(config_path: str):
    """Run bifurcation analysis using config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load model
    log.info(f"Loading model: {cfg['model']['model_id']}")
    device = cfg['model']['device']
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['model_id'],
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['model_id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load problem from sweep
    sweep_dir = Path(cfg['data_source']['sweep_dir'])
    problem = load_sweep_problem(
        sweep_dir,
        cfg['data_source']['temperature'],
        cfg['data_source']['model_filter']
    )

    if not problem:
        log.error("No suitable problem found!")
        return None

    # Use existing solutions
    outputs = problem['outputs']
    labels = [1 if c else 0 for c in problem['correctness']]

    # Build prompt
    prompt = f"""You are a helpful mathematical assistant. Solve the following problem step-by-step.
IMPORTANT: You must put your final answer within \\boxed{{}}.

Problem:
{problem['problem']}

Solution:
"""

    log.info(f"Extracting hidden states from {len(outputs)} solutions...")
    hidden_states = []

    token_pos = cfg['analysis']['token_position']
    layer_idx = cfg['analysis']['layer_idx']

    for output in tqdm(outputs, desc="Extracting"):
        full_text = prompt + output
        inputs = tokenizer(full_text, return_tensors="pt")

        if inputs['input_ids'].shape[1] >= token_pos:
            prefix_ids = inputs['input_ids'][0, :token_pos]
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        else:
            prefix_text = full_text

        h = extract_hidden_state(model, tokenizer, prefix_text, token_pos - 1, layer_idx, device)
        hidden_states.append(h.numpy())

    # Greedy solution
    log.info("Generating greedy solution...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        greedy_output = model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False)
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

    pca = PCA(n_components=2)
    hidden_2d = pca.fit_transform(hidden_states)
    greedy_2d = pca.transform(greedy_hidden.numpy().reshape(1, -1))[0]

    results = {
        'hidden_2d': hidden_2d,
        'labels': labels,
        'greedy_2d': greedy_2d,
        'greedy_correct': greedy_correct,
        'explained_variance': pca.explained_variance_ratio_,
        'n_success': int(labels.sum()),
        'n_fail': int(len(labels) - labels.sum()),
        'problem_id': problem['problem_id'],
        'layer_idx': layer_idx,
        'token_position': token_pos
    }

    # Save and plot
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg['model']['name']

    # Save results
    with open(output_dir / f"{model_name}_results.json", 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}, f, indent=2)

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
