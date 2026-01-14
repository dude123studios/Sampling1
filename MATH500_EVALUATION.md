# MATH-500 Evaluation Guide

This guide covers the updated evaluation system using HuggingFaceH4/MATH-500 with comprehensive per-level analysis, systematic parameter sweeps, and high-quality plotting.

## 🎯 Overview

The codebase now uses **HuggingFaceH4/MATH-500**, a curated subset of 500 problems from OpenAI's MATH benchmark with:
- **500 problems** total
- **5 difficulty levels** (1 = easiest, 5 = hardest)
- **7 subjects**: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus
- **Field names**: `problem`, `answer`, `subject`, `level`, `unique_id`

## 📊 New Features

### 1. Level Tracking
All experiments now automatically track and report:
- Per-level accuracy (Level 1-5)
- Per-subject performance
- Problem metadata in results

### 2. Level Filtering
You can now filter by difficulty level:
```yaml
# configs/task/math.yaml
level_filter: 5              # Only Level 5 (hardest)
level_filter: [3, 5]         # Levels 3-5
level_filter: [1, 3, 5]      # Specific levels
level_filter: null           # All levels
```

### 3. Systematic Sweeps
New sweep configuration system for running experiments systematically:
- Temperature sweeps
- Model comparisons
- Sample count variations
- Level-specific evaluations

### 4. High-Quality Plotting
ICML-quality plots showing:
- Performance by difficulty level
- Temperature vs accuracy curves
- Model comparisons across levels
- Subject-wise performance
- Pass@k curves

## 🚀 Quick Start

### Run a Single Experiment

```bash
# Run on all levels with default settings
python scripts/run_experiment.py model=openrouter task=math method=baseline

# Run on Level 5 only (hardest problems)
python scripts/run_experiment.py model=openrouter task=math method=baseline task.level_filter=5

# Run with specific temperature
python scripts/run_experiment.py model=openrouter task=math method=baseline method.temperature=0.8

# Run with more samples for better pass@k
python scripts/run_experiment.py model=openrouter task=math method=baseline num_samples=50
```

### Run Systematic Sweeps

```bash
# List available sweeps
python scripts/run_sweep.py --list

# Run temperature sweep (tests temperatures 0.0-1.0)
python scripts/run_sweep.py --sweep temperature_sweep

# Run baseline comparison across models
python scripts/run_sweep.py --sweep baseline_sweep

# Test on Level 5 only
python scripts/run_sweep.py --sweep level5_sweep

# Quick test with limited problems
python scripts/run_sweep.py --sweep quick_test --dry-run

# Full evaluation
python scripts/run_sweep.py --sweep full_evaluation
```

### Generate Plots

```bash
# Plot per-level performance from a single run
python scripts/plot_math_results.py --log results/2026-01-12/math/baseline/log.jsonl --output figures/

# Generate all plots for a run
python scripts/plot_math_results.py --log results/.../log.jsonl --all

# Plot temperature sweep results
python scripts/plot_math_results.py --sweep results/sweeps/temperature_sweep/ --output figures/

# Compare multiple models
python scripts/plot_math_results.py --compare log1.jsonl log2.jsonl log3.jsonl \
    --labels "Model A" "Model B" "Model C" --output figures/
```

## 📝 Configuration System

### Sweep Configurations (`configs/sweep_config.py`)

Define parameter sweeps in Python for flexibility:

```python
SWEEPS = {
    "my_custom_sweep": {
        "description": "Custom parameter sweep",
        "models": BASELINE_MODELS,
        "parameters": {
            "temperature": [0.6, 0.8],
            "num_samples": [25, 50],
            "level_filter": [None, 5],  # All levels + Level 5 only
            "limit": [None],  # All problems
        },
        "task": {
            "name": "math",
            "dataset": "HuggingFaceH4/MATH-500",
            "split": "test",
            "level_filter": None,
        },
        "method": {
            "name": "baseline"
        }
    }
}
```

### Available Sweeps

1. **baseline_sweep**: Standard comparison across models (10/25/50 samples)
2. **temperature_sweep**: Test temperatures 0.0-1.0
3. **level5_sweep**: Focus on hardest problems only
4. **per_level_sweep**: Test each level separately
5. **model_comparison**: Compare all models on full dataset
6. **quick_test**: Fast sweep with 50 problems for debugging
7. **full_evaluation**: Comprehensive sweep (models × temperatures × samples)

### Model Configurations

Default baseline models (matching prefix experiments):
- **qwen-8b**: `qwen/qwen-2.5-7b-instruct`
- **deepseek-llama-8b**: `deepseek/deepseek-r1-distill-llama-8b`
- **qwq-32b**: `qwen/qwq-32b-preview`

Add more models in `configs/sweep_config.py`:
```python
EXTENDED_MODELS = BASELINE_MODELS + [
    {
        "name": "my-model",
        "model_name": "provider/model-id",
        "type": "api",
        "provider": "openrouter"
    }
]
```

## 📈 Results Format

### JSONL Output

Each line in `log.jsonl` contains:
```json
{
  "id": 0,
  "dataset_id": "unique_problem_id",
  "original_prompt": "...",
  "outputs": ["solution1", "solution2", ...],
  "scores": [1, 0, 1, ...],
  "gold": "answer",
  "metrics": {
    "num_correct": 2,
    "pass@1": 0.42,
    "pass@5": 0.89,
    "one@k": 1.0
  },
  "level": 5,
  "subject": "Algebra"
}
```

### Summary Entry

The last line contains aggregated metrics:
```json
{
  "type": "summary",
  "avg_pass@1": 0.45,
  "avg_pass@5": 0.78,
  "per_level_metrics": {
    "level_1": {"pass@1": 0.85, ...},
    "level_2": {"pass@1": 0.72, ...},
    "level_3": {"pass@1": 0.58, ...},
    "level_4": {"pass@1": 0.42, ...},
    "level_5": {"pass@1": 0.28, ...}
  }
}
```

## 🎨 Generated Plots

All plots are saved as high-quality PDFs (300 DPI, proper fonts for publication):

### Per-Level Performance (`per_level_performance.pdf`)
Bar chart showing accuracy across difficulty levels 1-5

### Pass@k Curve (`pass_at_k.pdf`)
Shows how accuracy improves with more samples (Pass@1, Pass@5, Pass@10, etc.)

### Subject Performance (`subject_performance.pdf`)
Horizontal bar chart showing accuracy by math subject

### Temperature Comparison (`temperature_comparison.pdf`)
Line plot showing how temperature affects performance

### Model Comparison (`model_comparison.pdf`)
Grouped bar chart comparing models across difficulty levels

## 🔬 Prefix Experiments (Updated)

The prefix experiments now also use MATH-500:

```bash
# 1. Generate oracle solutions (uses MATH-500)
python scripts/generate_oracle_solutions.py --num_problems 500

# 2. Run oracle prefix experiment
python scripts/run_oracle_prefix_experiment.py --limit 100

# 3. Run self-correct prefix experiment
python scripts/run_self_correct_prefix_experiment.py

# 4. Plot results
python scripts/plot_prefix_experiments.py --both oracle.json self_correct.json
```

All prefix experiment configurations updated to:
```yaml
task:
  name: "math"
  dataset: "HuggingFaceH4/MATH-500"
  split: "test"
  level_filter: null  # Use all levels
```

## 💡 Usage Examples

### Example 1: Baseline on All Target Models

```bash
# Edit configs/sweep_config.py to customize BASELINE_MODELS
python scripts/run_sweep.py --sweep baseline_sweep
python scripts/plot_math_results.py --sweep results/sweeps/baseline_sweep/
```

### Example 2: Temperature Analysis

```bash
# Run temperature sweep
python scripts/run_sweep.py --sweep temperature_sweep

# Plot results
python scripts/plot_math_results.py --sweep results/sweeps/temperature_sweep/ \
    --output figures/temp_analysis/
```

### Example 3: Level 5 Focus (Hardest Problems)

```bash
# Run only on Level 5
python scripts/run_sweep.py --sweep level5_sweep

# Compare with all-levels results
python scripts/plot_math_results.py --compare \
    results/sweeps/baseline_sweep/qwen-8b_temp0.6_n25_all_levels/log.jsonl \
    results/sweeps/level5_sweep/qwen-8b_temp0.6_n25_level5/log.jsonl \
    --labels "All Levels" "Level 5 Only"
```

### Example 4: Per-Subject Analysis

```bash
# Run experiment
python scripts/run_experiment.py model=openrouter task=math num_samples=50

# Plot subject performance
python scripts/plot_math_results.py --log results/.../log.jsonl --all
# Generates: per_level_performance.pdf, subject_performance.pdf, pass_at_k.pdf, summary_report.txt
```

### Example 5: Custom Sweep

Edit `configs/sweep_config.py`:
```python
SWEEPS["my_sweep"] = {
    "description": "Test my hypothesis",
    "models": [BASELINE_MODELS[0]],  # Just qwen-8b
    "parameters": {
        "temperature": [0.4, 0.6, 0.8],
        "num_samples": [25],
        "level_filter": [5],  # Level 5 only
        "limit": [100],  # Fast test
    },
    # ... rest of config
}
```

Then run:
```bash
python scripts/run_sweep.py --sweep my_sweep
```

## 📊 Best Practices

### For Baseline Comparisons
```bash
# Use consistent settings across models
python scripts/run_sweep.py --sweep baseline_sweep
# This runs: temp=0.6, n=[10,25,50], all levels, all 500 problems
```

### For Temperature Studies
```bash
# Test full temperature range
python scripts/run_sweep.py --sweep temperature_sweep
# This runs: temp=[0.0,0.2,0.4,0.6,0.8,1.0], n=25
```

### For Quick Testing
```bash
# Use quick_test sweep for debugging
python scripts/run_sweep.py --sweep quick_test --dry-run  # See what will run
python scripts/run_sweep.py --sweep quick_test             # Actually run
```

### For Publication-Quality Results
```bash
# Run full evaluation
python scripts/run_sweep.py --sweep full_evaluation

# Generate all plots
for logfile in results/sweeps/full_evaluation/*/log.jsonl; do
    python scripts/plot_math_results.py --log $logfile --all
done

# Generate comparison plots
python scripts/plot_math_results.py --compare \
    results/sweeps/full_evaluation/qwen-8b_*/log.jsonl \
    --labels "Qwen-8B T=0.0" "Qwen-8B T=0.2" "..." \
    --output figures/full_eval/
```

## 🐛 Troubleshooting

### Issue: "Unknown dataset field 'solution'"
**Cause**: Old configs using DigitalLearningGmbH/MATH-lighteval
**Fix**: Update to `HuggingFaceH4/MATH-500` and use `answer` field

### Issue: Level filtering not working
**Cause**: Old `subset_level` parameter
**Fix**: Use `level_filter` instead:
```yaml
level_filter: 5        # Not subset_level: "Level 5"
```

### Issue: Plots don't show per-level breakdown
**Cause**: Results don't have level metadata
**Fix**: Re-run experiment with updated code that tracks levels

### Issue: Sweep runs too long
**Fix**: Use `quick_test` sweep or add `limit` parameter:
```python
"limit": [50],  # Only 50 problems
```

## 📚 References

### Dataset
- **Paper**: "Let's Verify Step by Step" (OpenAI)
- **Dataset**: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- **Original**: [MATH Benchmark](https://github.com/hendrycks/math)

### Evaluation Metrics
- **Pass@k**: Unbiased estimator from HumanEval paper
- **One@k**: Binary indicator if any sample was correct
- Formula: `pass@k = 1 - C(n-c, k) / C(n, k)`

## 🔄 Migration from Old System

If you have old experiments using DigitalLearningGmbH/MATH-lighteval:

1. **Update configs**:
```yaml
# Old
dataset: "DigitalLearningGmbH/MATH-lighteval"
subset_level: "Level 5"

# New
dataset: "HuggingFaceH4/MATH-500"
level_filter: 5
```

2. **Update code references**:
```python
# Old
gold = item['solution']

# New
gold = item.get('answer', item.get('solution', ''))
```

3. **Re-run experiments**: Old results won't have level metadata, so re-run for per-level analysis

## 🎯 Summary

The updated system provides:
✅ Proper MATH-500 dataset support
✅ Per-level and per-subject tracking
✅ Flexible level filtering
✅ Systematic parameter sweeps
✅ High-quality ICML-style plots
✅ Backward compatibility with prefix experiments
✅ Comprehensive documentation

All experiments now follow best practices for reproducible research on MATH-500!

## Sources

- [HuggingFaceH4/MATH-500 Dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [README.md · HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/main/README.md)
- [Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/some-evaluation-datasets.md)
