# Trajectory Bifurcation Analysis

Analyzes where "good" and "bad" reasoning paths diverge using existing sweep data.

## Quick Start

### On RunPod (Notebook)
Run the bifurcation cell in `RunPod_MechInterp.ipynb` (Step 8).

### Local (Command Line)
```bash
# Qwen3-8B
python mech_interp/bifurcation_analysis.py configs/mech_interp/bifurcation_config.yaml

# DeepSeek-Qwen3-8B
python mech_interp/bifurcation_analysis.py configs/mech_interp/bifurcation_deepseek_config.yaml
```

## Configuration

Edit `configs/mech_interp/bifurcation_config.yaml`:

```yaml
model:
  model_id: "Qwen/Qwen3-8B"
  device: "cuda"

data_source:
  sweep_dir: "results/sweeps/temperature_sweep"
  temperature: "0.6"  # Which temp to analyze
  model_filter: "qwen3-8b"  # Model to analyze

analysis:
  token_position: 16  # Extract state at token 16
  layer_idx: 10  # Extract from layer 10
```

**Note**:
- Uses existing sweep data with 5 samples
- Automatically finds level 5 problems with 1-2/5 solve rate
- Qwen found: `test/intermediate_algebra/1994.json` (1/5)
- DeepSeek found: `test/intermediate_algebra/1994.json` (2/5)

## Output

- `mech_interp/bifurcation_results/{model}_bifurcation.png` - PCA plot
- `mech_interp/bifurcation_results/{model}_results.json` - Data

## Interpretation

**Blue dots**: Successful solutions
**Red dots**: Failed solutions
**Green star**: Greedy (T=0) solution

- **Separation** = Model knows early which paths succeed
- **Green in blue cluster** = Well-calibrated greedy
- **Green isolated** = Mode collapse

## Requirements

- Existing sweep data at correct temperature
- GPU with 16GB+ VRAM
- ~30 minutes per model
