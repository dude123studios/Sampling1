# RunPod Quick Start Guide

## For the Impatient

1. **Get the notebook**: `RunPod_MechInterp.ipynb`
2. **Upload to RunPod**: Use web interface
3. **Run all cells**: Takes 2-4 hours on A100
4. **Download results**: ZIP file created automatically

## Step-by-Step

### In RunPod Pod

1. **Open Jupyter Lab** (RunPod provides this)
2. **Upload the notebook**:
   - Click folder icon
   - Drag `RunPod_MechInterp.ipynb` here
3. **Open the notebook**
4. **Run cells in order** (Shift+Enter):
   - Cell 1: Clone repo & install
   - Cell 2: Install requirements
   - Cell 3: Verify GPU
   - Cell 4: Verify data
   - Cell 5: Run token impact
   - Cell 6: Run DLA
   - Cell 7: Run patching
   - Cell 8: Run gradient
   - Cell 9: Summary
   - Cell 10: Download

### GPU Requirements

| GPU | Memory | Time | Notes |
|-----|--------|------|-------|
| A100 40GB | ✓ | 2-3h | Recommended |
| RTX 6000 Ada | ✓ | 2-3h | Good alternative |
| H100 | ✓ | 1-2h | Overkill but fast |
| A6000 | ~ | 3-4h | Tight but works |

## Customization

### Modify Problem Count

Edit cell before running (e.g., analyze only first 10 problems):
```python
# Add to cell after imports:
max_problems = 10  # Limit to first 10
```

### Faster Iteration

Modify `configs/mech_interp/token_impact_config.yaml`:
```yaml
# Fewer cutoff positions = faster
token_forcing:
  cutoff_positions: [16, 32, 48, 64]  # Instead of 8 positions

# Shorter continuations = much faster
token_forcing:
  continuation_length: 32  # Instead of 128
```

### Skip Already Completed Stages

Use the `--skip-existing` flag (notebook doesn't support yet, but CLI does):
```bash
python mech_interp/run_mech_interp.py --stage all --skip-existing
```

## Troubleshooting

**"No sweep data found"**
- Your pod needs the sweep results
- Either: upload them, or run sweep experiments first
- Check: `results/sweeps/<timestamp>/qwen3-8b_temp0.6_*/log.jsonl`

**"CUDA out of memory"**
- Get bigger GPU (A100 40GB recommended)
- Or reduce continuation_length to 64 in config
- Or reduce cutoff_positions

**Slow on GPU X**
- Normal: token generation is sequential
- Token Impact stage: most expensive
- Each problem: 8 cutoffs × 2 paths × 128 tokens = 2048 forward passes
- ~100 Level 5 problems = 200k+ forward passes

**Notebook crashes mid-run**
- Check GPU memory: `nvidia-smi` in terminal
- Restart pod and try again
- Reduce batch size / problem count

## File Structure on Pod

After cloning, you'll have:
```
/workspace/Sampling1/
├── RunPod_MechInterp.ipynb     ← The notebook
├── requirements.txt             ← Dependencies
├── mech_interp/
│   ├── token_impact.py         ← Stage 1
│   ├── dla.py                  ← Stage 2
│   ├── patching.py             ← Stage 3
│   ├── gradient.py             ← Stage 4
│   └── run_mech_interp.py      ← Main runner
├── configs/mech_interp/
│   ├── token_impact_config.yaml
│   └── dla_patching_config.yaml
└── results/sweeps/             ← Your sweep data (upload here)
```

## Download Results

Results are automatically packaged as ZIP:
```
mech_interp_results_YYYYMMDD_HHMMSS.zip
```

Contains:
- All JSON results
- Statistics files
- Ready for analysis

## Next Steps

After download:
1. Extract ZIP locally
2. Analyze JSON files (Python / pandas)
3. Create visualizations
4. Write up findings

## Support

Check these if stuck:
1. `MECH_INTERP_README.md` - Full documentation
2. Cell output in notebook - Error messages
3. GPU logs: `nvidia-smi` in terminal

## Costs (Approximate)

- A100 40GB on RunPod: ~$0.44/hour
- Full pipeline: 2-3 hours
- **Total cost**: ~$1-2 per run

Worth it for research data! 🚀
