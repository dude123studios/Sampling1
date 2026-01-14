# Mechanistic Interpretability Experiments

GPU-optimized mechanistic interpretability pipeline for analyzing decision-critical layers in language models.

## Quick Start on RunPod

1. Upload `RunPod_MechInterp.ipynb` to RunPod
2. Run all cells in sequence
3. Download results as ZIP archive

The notebook handles:
- Cloning the GitHub repository
- Installing dependencies
- Running the full pipeline (4 stages)
- Creating downloadable results archive

## Pipeline Overview

### Stage 1: Token Impact Identification
**Purpose**: Find branching points where token choice matters most

**Process**:
- Load Level 5 math problems from sweep results
- At each cutoff position (8, 16, 24, ..., 64):
  - Get top 1st and 2nd most likely tokens
  - Force each token and generate 128 continuations
  - Extract activations from layers [15, 20, 25, 31]
  - Compute cosine similarity divergence between paths

**Output**: `token_impact_results.json`
- Lists all branching points with divergence scores
- Identifies most impactful positions per problem

### Stage 2: Direct Logit Attribution (DLA)
**Purpose**: Quantify which layers contribute to token preference

**Method**: Decomposes logit difference into layer contributions
```
DLA_ℓ = u^T · Δz^ℓ
where u = W_U[token1] - W_U[token2]
      Δz^ℓ = residual stream delta at layer ℓ
```

**Analysis Bands**:
- Abstraction (layers 8-20): High-level pattern recognition
- Commitment (layers 22-28): Final decision-making

**Output**: `dla_results.json` and `dla_statistics.json`

### Stage 3: Activation Patching
**Purpose**: Identify critical layers via causal intervention

**Method**:
- Cache activations from top1 (correct) path
- Replace activations at each layer in top2 (incorrect) path
- Measure change in logit difference
- Three patch types: residual_stream, attention_output, mlp_output

**Output**: `patching_results.json` and `patching_statistics.json`
- Most important layers (by patch effect)
- Layer-wise causal contributions

### Stage 4: Gradient Attribution
**Purpose**: Measure sensitivity/instability at each layer

**Method**:
- Compute ∇_{z^ℓ} (logit[token1] - logit[token2])
- Measure L2 norm of gradients
- Identify layers with highest sensitivity

**Output**: `gradient_results.json` and `gradient_statistics.json`
- Mean gradient norm per layer
- Most sensitive layers

## GPU Optimization

The pipeline is fully GPU-optimized for RunPod:

**Memory Efficiency**:
- Mixed precision (float16) for model computations
- Activations moved to CPU for storage
- Periodic `torch.cuda.empty_cache()` calls
- No gradient accumulation

**Estimated GPU Memory**: ~16-24 GB for Qwen3-8B
- Model weights: ~8 GB (float16)
- Activations and intermediates: ~8-16 GB

**Recommended RunPod**: A100 40GB or RTX 6000 Ada

## Configuration Files

### `configs/mech_interp/token_impact_config.yaml`
```yaml
model: Qwen/Qwen3-8B
data_source:
  sweep_dir: results/sweeps
  model_name: qwen3-8b
  temperature: 0.6
  level_filter: 5  # Only Level 5 problems

token_forcing:
  cutoff_positions: [8, 16, 24, 32, 40, 48, 56, 64]
  continuation_length: 128
  temperature: 0.6

layer_analysis:
  layers: [15, 20, 25, 31]
  averaging_positions: [0, 32, 64, 96]
```

### `configs/mech_interp/dla_patching_config.yaml`
```yaml
dla_config:
  layers: "all"  # All 32 layers
  analysis_bands:
    abstraction: [8, 9, ..., 20]
    commitment: [22, 23, ..., 28]

patching:
  patch_types:
    - residual_stream
    - attention_output
    - mlp_output
  layers: [0, 1, ..., 31]

gradient_analysis:
  layers: [15, 20, 25, 31]
```

## Running Locally

```bash
# Run full pipeline
python mech_interp/run_mech_interp.py --stage all

# Run individual stages
python mech_interp/run_mech_interp.py --stage token_impact
python mech_interp/run_mech_interp.py --stage dla
python mech_interp/run_mech_interp.py --stage patching
python mech_interp/run_mech_interp.py --stage gradient

# Skip already completed stages
python mech_interp/run_mech_interp.py --stage all --skip-existing
```

## Prerequisites

**Data**: You need sweep results from qwen3-8b at temp=0.6
```
results/sweeps/
  └── <timestamp>/qwen3-8b_temp0.6_*/log.jsonl
```

**Dependencies**:
- `torch` >= 2.0 (with CUDA support)
- `transformers` >= 4.30
- `pyyaml`
- `numpy`
- `tqdm`

Install with:
```bash
pip install -r requirements.txt
```

## Output Structure

```
mech_interp/
├── token_impact_results/
│   ├── token_impact_results.json      # Branching points
│   └── impactful_positions.json       # Top positions per problem
├── dla_results/
│   ├── dla_results.json               # Layer-wise DLA scores
│   └── dla_statistics.json            # Aggregate statistics
├── patching_results/
│   ├── patching_results.json          # Patch effects
│   └── patching_statistics.json       # Important layers
└── gradient_results/
    ├── gradient_results.json          # Gradient norms
    └── gradient_statistics.json       # Sensitivity analysis
```

## Key Insights to Look For

1. **Token Impact**: Which token positions cause largest activation divergence?
   - Early positions: semantic routing
   - Late positions: output formatting

2. **DLA**: Which layers make the decision?
   - Abstraction band: semantic composition
   - Commitment band: decision execution

3. **Patching**: Are decisions reversible?
   - High patch effects: critical layers
   - Low patch effects: redundant representations

4. **Gradients**: Where is the model unstable?
   - High gradients: sensitive decisions
   - Low gradients: robust representations

## Troubleshooting

**RunPod Memory Issues**:
- Try A100 40GB or higher
- Reduce `continuation_length` in config
- Reduce number of `cutoff_positions`

**Missing Sweep Data**:
- Upload sweep results to results/sweeps/
- Ensure directory structure: `results/sweeps/<timestamp>/qwen3-8b_temp0.6_*/log.jsonl`

**Slow Execution**:
- Normal for A100: ~2-4 hours per stage for 100+ problems
- Each problem tests 8 cutoff positions
- Token impact stage is most expensive (128 token generation × 2 paths)

## Future Work

**TODO** in code:
1. Store input sequences in token_impact results for DLA/patching/gradient
2. Implement proper sequence loading for downstream analyses
3. Add statistical significance tests
4. Create visualization tools

## References

- Direct Logit Attribution: [Nostalgebraist blog](https://www.lesswrong.com/posts/JvZhhzycHu2Fs3vFt/the-singular-value-decompositions-of-transformer-weight)
- Activation Patching: [Meng et al., 2023](https://arxiv.org/abs/2304.14997)
- Mechanistic Interpretability: [Nanda & Lieberum, 2022](https://www.lesswrong.com/posts/8bx94q9FPJsQQe4na/grokking-and-loss-curves)
