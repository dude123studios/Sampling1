# Quick Start Guide

## 🔑 API Key Setup

Most experiments use OpenRouter API. Set up your API key:

```bash
# Create .env file in project root
echo "OPENROUTER_API_KEY=sk-or-v1-your-api-key-here" > .env

# Make sure .env is in .gitignore (it should be already)
# The .env file contains your API keys and should never be committed to git
```

The experiments will automatically load API keys from the `.env` file.

## 🚀 HumanEval Code Generation

### Run Tests
```bash
python3 scripts/test_humaneval.py
```
Expected: `6/6 tests PASSED ✅`

### Run Full Sweep (100 problems × 25 samples)
```bash
python scripts/run_humaneval_sweep.py \
    --config configs/sweep/humaneval_sweep.yaml \
    --max-workers 15
```

### Quick Test (10 problems × 5 samples)
```bash
# Edit config first
vim configs/sweep/humaneval_sweep.yaml
# Change: limit: [10], num_samples: [5]

python scripts/run_humaneval_sweep.py \
    --config configs/sweep/humaneval_sweep.yaml \
    --max-workers 15
```

### Visualize Results
```bash
python scripts/plot_humaneval_results.py \
    --sweep-dir results/sweeps/humaneval_sweep \
    --output-dir results/plots
```

**Output**:
- `results/plots/humaneval_pass_at_k_bars.png`
- `results/plots/humaneval_pass_at_k_lines.png`

---

## 🔬 Trajectory Bifurcation Analysis

### Run for Both Models
```bash
chmod +x scripts/run_bifurcation_comparison.sh
./scripts/run_bifurcation_comparison.sh
```

### Run for Single Model
```bash
python scripts/analyze_trajectory_bifurcation.py \
    --model "Qwen/Qwen2.5-Math-7B-Instruct" \
    --model-display-name "qwen3-8b" \
    --level 5 \
    --n-samples 100 \
    --token-position 16 \
    --layer 10 \
    --output-dir "results/bifurcation"
```

### Quick Test (20 samples)
```bash
python scripts/analyze_trajectory_bifurcation.py \
    --model "Qwen/Qwen2.5-Math-7B-Instruct" \
    --model-display-name "qwen3-8b" \
    --n-samples 20 \
    --output-dir "results/bifurcation/test"
```

**Output**:
- `results/bifurcation/bifurcation_plot_qwen3-8b.png`
- `results/bifurcation/bifurcation_results_qwen3-8b.json`

---

## 📊 Expected Results

### HumanEval Pass@k
```
Model: qwen3-8b
  pass@1: 0.45-0.55
  pass@5: 0.70-0.80
  pass@10: 0.80-0.90
  pass@20: 0.85-0.95
```

### Bifurcation Analysis
```
Success rate: 20-40% (depends on problem)
Greedy: Often correct or in success cluster
PCA variance: PC1 ~40-60%, PC2 ~20-30%
```

---

## ⚡ Common Issues

### Python not found
```bash
# Use python3 instead
python3 scripts/test_humaneval.py
```

### GPU Out of Memory (Bifurcation)
```bash
# Reduce samples or use smaller model
python scripts/analyze_trajectory_bifurcation.py \
    --n-samples 50 \
    --model "smaller-model"
```

### No hard problems found (Bifurcation)
```bash
# Adjust pass rate threshold
# Edit line in script: target_pass_rate=0.15
```

### API Rate Limits (HumanEval)
```bash
# Reduce parallel workers
python scripts/run_humaneval_sweep.py \
    --config configs/sweep/humaneval_sweep.yaml \
    --max-workers 5
```

---

## 📁 Results Location

```
results/
├── sweeps/humaneval_sweep/          # HumanEval results
│   ├── qwen3-8b_*/
│   │   └── log.jsonl
│   └── deepseek-qwen3-8b_*/
│       └── log.jsonl
├── bifurcation/                      # Bifurcation results
│   ├── bifurcation_plot_qwen3-8b.png
│   ├── bifurcation_results_qwen3-8b.json
│   ├── bifurcation_plot_deepseek-qwen3-8b.png
│   └── bifurcation_results_deepseek-qwen3-8b.json
└── plots/                            # Visualizations
    ├── humaneval_pass_at_k_bars.png
    └── humaneval_pass_at_k_lines.png
```

---

## 📖 Full Documentation

- **HumanEval**: `docs/HUMANEVAL.md`
- **Bifurcation**: `docs/BIFURCATION_ANALYSIS.md`
- **Summary**: `docs/IMPLEMENTATION_SUMMARY.md`

---

## 🎯 Minimal End-to-End Example

```bash
# 1. Test everything works
python3 scripts/test_humaneval.py

# 2. Quick HumanEval test (10 problems)
# (Edit config: limit: [10], num_samples: [5])
python scripts/run_humaneval_sweep.py \
    --config configs/sweep/humaneval_sweep.yaml

# 3. Visualize
python scripts/plot_humaneval_results.py

# 4. Quick bifurcation test (20 samples)
python scripts/analyze_trajectory_bifurcation.py \
    --model "Qwen/Qwen2.5-Math-7B-Instruct" \
    --model-display-name "qwen3-8b" \
    --n-samples 20

# Done! Check results/
```

---

## ⏱️ Time Estimates

| Task | Time | Notes |
|------|------|-------|
| Test suite | 2 min | Verifies everything works |
| HumanEval (10 problems × 5) | 5 min | Quick test |
| HumanEval (100 problems × 25) | 60 min | Full sweep |
| Bifurcation (20 samples) | 5 min | Quick test |
| Bifurcation (100 samples) | 30 min | Full analysis |
| Both models bifurcation | 60 min | Complete comparison |

---

## 🔧 Configuration

### HumanEval Sweep
Edit `configs/sweep/humaneval_sweep.yaml`:
```yaml
parameters:
  temperature: [0.6]
  num_samples: [25]    # Samples per problem
  top_k: [50]
  top_p: [0.9]
  limit: [100]         # Number of problems (null = all 164)
```

### Bifurcation Analysis
Command line args:
```bash
--n-samples 100        # Solutions to generate
--token-position 16    # Which token to extract
--layer 10            # Which layer to analyze
--temperature 0.6     # Sampling temperature
```

---

## ✅ Success Checklist

- [ ] Tests pass (6/6)
- [ ] HumanEval sweep runs without errors
- [ ] Plots generated in results/plots/
- [ ] Bifurcation analysis completes
- [ ] Clusters visible in bifurcation plot
- [ ] Results saved to JSON

If all checked: **You're ready for production runs! 🎉**
