# Trajectory Bifurcation Analysis

This analysis investigates **when and how** language models diverge into "good" and "bad" reasoning paths during generation.

## Research Question

**Can we predict success/failure from early hidden states?**

If successful and failed solutions cluster separately in the activation space early in generation (e.g., at token 16), this suggests the model "knows" whether it's on the right track very early - before completing the full solution.

## Method

### 1. Problem Selection
- Pick hard MATH problems (difficulty level 5)
- Target ~20% pass rate (1 correct out of 5 attempts)

### 2. Solution Generation
- Generate 100 solutions at temperature=0.6
- Label each as "Success" (correct answer) or "Fail" (incorrect)

### 3. Hidden State Extraction
- **Critical step**: Extract hidden states from **prefixes only** (no generation)
- Use tokens 1-16 as prefill
- Extract activation from **Layer 10** at **Token 16**
- This gives us 100 vectors in R^d (where d = hidden_dim, typically 4096)

### 4. Dimensionality Reduction (PCA)
- Project 100 high-dimensional vectors → 2D using PCA
- Preserves maximum variance structure

### 5. Visualization
- **Blue dots**: Successful solutions (led to correct answer)
- **Red dots**: Failed solutions (led to incorrect answer)
- **Green star**: Greedy (T=0) solution

## Hypothesis & Interpretation

### If Clusters Separate
✓ **Strong separation** → Model "knows" early which paths lead to success

This suggests:
- Different reasoning strategies are distinguishable in activation space
- Early layers encode solution quality
- Potential for **early intervention** or **prefix steering**

### Greedy Position Analysis

**Green star in blue cluster**:
- Greedy decoding follows successful reasoning pattern
- Model is well-calibrated (confident when correct)

**Green star isolated or in red cluster**:
- Greedy decoding doesn't follow best path
- Suggests **mode collapse** or **exploitation problem**
- Temperature sampling explores better regions

**Green star in red cluster**:
- Greedy is "confident but wrong"
- Classic mode collapse issue
- Temperature is necessary to escape local optimum

## Running the Analysis

### Single Model

```bash
python scripts/analyze_trajectory_bifurcation.py \
    --model "Qwen/Qwen2.5-Math-7B-Instruct" \
    --model-display-name "qwen3-8b" \
    --level 5 \
    --n-samples 100 \
    --token-position 16 \
    --layer 10 \
    --temperature 0.6 \
    --output-dir "results/bifurcation"
```

### Both Models (Comparison)

```bash
chmod +x scripts/run_bifurcation_comparison.sh
./scripts/run_bifurcation_comparison.sh
```

This runs the analysis for both qwen3-8b and deepseek-qwen3-8b.

## Parameters

### Token Position (default: 16)
- Position to extract hidden state
- Earlier positions (e.g., 8) test very early divergence
- Later positions (e.g., 32) test after more reasoning

### Layer (default: 10)
- Which layer to extract from
- Early layers (0-5): surface features
- Middle layers (8-12): reasoning patterns
- Late layers (20+): final answer formation

### Temperature (default: 0.6)
- Sampling temperature for solution generation
- Higher = more diverse trajectories
- Lower = more focused search

## Output

### Files Created

```
results/bifurcation/
├── bifurcation_results_qwen3-8b.json        # PCA coordinates + metadata
├── bifurcation_plot_qwen3-8b.png            # Visualization
├── bifurcation_results_deepseek-qwen3-8b.json
└── bifurcation_plot_deepseek-qwen3-8b.png
```

### JSON Results Structure

```json
{
  "hidden_2d": [[x1, y1], [x2, y2], ...],  // PCA coordinates
  "labels": [1, 0, 1, 0, ...],              // 1=success, 0=fail
  "greedy_2d": [x, y],                      // Greedy position
  "greedy_correct": true,                   // Greedy correctness
  "explained_variance": [0.45, 0.23],       // PC1, PC2 variance
  "n_success": 34,
  "n_fail": 66,
  "problem_id": "...",
  "layer_idx": 10,
  "token_position": 16,
  "temperature": 0.6
}
```

## Example Interpretations

### Scenario 1: Clear Separation
```
        Blue cluster
           ●●●
          ●●●●●
         ●●●★●●    ← Green star in success cluster
          ●●●●
           ●●


                    Red cluster
                      ●●●
                     ●●●●●
                      ●●●
```

**Interpretation**: Model has distinct "good" and "bad" reasoning modes. Greedy follows the good mode. High confidence in correct reasoning.

### Scenario 2: Mode Collapse
```
        Blue cluster
           ●●●
          ●●●●●
           ●●●


              ★        ← Green star isolated


                    Red cluster
                      ●●●
                     ●●●●●
                      ●●●
```

**Interpretation**: Greedy gets stuck in a different region than successful samples. Needs temperature to explore properly.

### Scenario 3: No Separation
```
          ●●●●●●
         ●★●●●●●●
        ●●●●●●●●●    ← Blue and red mixed
         ●●●●●●●
          ●●●●●
```

**Interpretation**: Early hidden states don't predict success. Divergence happens later. Try higher token_position or different layer.

## Advanced Usage

### Sweep Across Layers

```bash
for layer in 0 5 10 15 20; do
    python scripts/analyze_trajectory_bifurcation.py \
        --model "Qwen/Qwen2.5-Math-7B-Instruct" \
        --layer $layer \
        --output-dir "results/bifurcation/layer_sweep"
done
```

### Sweep Across Token Positions

```bash
for pos in 8 16 24 32; do
    python scripts/analyze_trajectory_bifurcation.py \
        --model "Qwen/Qwen2.5-Math-7B-Instruct" \
        --token-position $pos \
        --output-dir "results/bifurcation/token_sweep"
done
```

## Related Work

This analysis is inspired by:

1. **Mechanistic Interpretability** (Anthropic, DeepMind)
   - Understanding internal representations
   - Circuit discovery in neural networks

2. **Activation Patching** (Meng et al., 2022)
   - Causal interventions in activation space
   - "Locating and Editing Factual Associations in GPT"

3. **Mode Collapse in Generation** (Holtzman et al., 2020)
   - "The Curious Case of Neural Text Degeneration"
   - Why greedy decoding fails

4. **Prefix Tuning** (Li & Liang, 2021)
   - Learning optimal prefixes for downstream tasks
   - Suggests prefix space is rich and structured

## Future Directions

1. **Intervention Experiments**: Can we "steer" red trajectories toward blue by adding vectors?

2. **Prefix Optimization**: Train prefix to maximize blue cluster membership

3. **Multi-Layer Analysis**: Track divergence across all layers simultaneously

4. **Temporal Dynamics**: How does separation evolve token-by-token?

5. **Transfer Analysis**: Do good prefixes for one problem work for others?

## Computational Requirements

- **GPU Memory**: ~16GB for 7B model (float16)
- **Time**: ~30 minutes per model (100 solutions × ~15s each)
- **Storage**: ~100MB per model (PCA coordinates + plots)

## Tips

1. **Finding Hard Problems**: Adjust `target_pass_rate` if no hard problems found
2. **GPU OOM**: Reduce batch size or use smaller model
3. **Slow Generation**: Reduce `n_samples` for faster iteration
4. **No Separation**: Try different layers (especially middle layers 8-12)

## Citation

If you use this analysis in your research, consider citing:

```bibtex
@article{yourpaper2024,
  title={Trajectory Bifurcation in Language Model Reasoning},
  author={Your Name},
  journal={NeurIPS},
  year={2024}
}
```
