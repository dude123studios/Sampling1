# Prefix Experiments

This document describes the prefix experiments designed to test the hypothesis that **the first few tokens of a solution matter most** for improving model performance on mathematical reasoning tasks.

## Overview

We conduct two complementary experiments:

1. **Oracle Prefix Experiment**: Tests how model performance improves when given prefixes of gold/oracle solutions at different token lengths.

2. **Self-Correct Prefix Experiment**: Tests whether providing prefixes of a model's own correct attempts helps it solve problems more consistently.

## Hypothesis

**The first 32 tokens provide the most gain per token in performance**, showing diminishing returns as prefix length increases. This would demonstrate that grounding the initial reasoning steps is critical for successful problem-solving.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have your OpenRouter API key set in your `.env` file:

```bash
OPENROUTER_API_KEY=your_key_here
```

## Experiment 1: Oracle Prefix Experiment

### Step 1: Generate Oracle Solutions

First, generate gold solutions using a strong model (deepseek-r1-llama-70b):

```bash
python scripts/generate_oracle_solutions.py --num_problems 500 --output data/oracle_solutions.json
```

**Parameters:**
- `--num_problems`: Number of math problems to generate solutions for (default: 500)
- `--output`: Output path for oracle solutions JSON (default: data/oracle_solutions.json)
- `--task`: Task name, currently only 'math' supported (default: math)

**Output:** Creates `data/oracle_solutions.json` containing oracle solutions with correctness labels.

### Step 2: Configure the Experiment

Edit `configs/oracle_prefix_experiment.yaml` to customize:

```yaml
# Prefix token lengths to test
prefix_lengths: [0, 32, 64, 128, 256, 512, 1024]

# Target models to test
target_models:
  - name: "qwen-8b"
    model_name: "qwen/qwen-2.5-7b-instruct"
  - name: "deepseek-llama-8b"
    model_name: "deepseek/deepseek-r1-distill-llama-8b"
  - name: "qwq-32b"
    model_name: "qwen/qwq-32b-preview"

# Sampling parameters
temperature: 0.6
max_new_tokens: 4096
```

### Step 3: Run the Experiment

Run the oracle prefix experiment:

```bash
# Test on all problems
python scripts/run_oracle_prefix_experiment.py --config configs/oracle_prefix_experiment.yaml

# Limit to first 100 problems for quick testing
python scripts/run_oracle_prefix_experiment.py --config configs/oracle_prefix_experiment.yaml --limit 100
```

**Parameters:**
- `--config`: Path to experiment config file
- `--limit`: Limit number of problems to test (optional, for debugging)

**Output:** Results saved to `results/oracle_prefix/oracle_prefix_results_TIMESTAMP.json`

### What It Does

For each model and each prefix length:
1. Takes the first N tokens of the oracle solution
2. Prepends it to the problem prompt (after "Solution:")
3. Generates a completion
4. Evaluates correctness
5. Measures accuracy at each prefix length

## Experiment 2: Self-Correct Prefix Experiment

### Step 1: Configure the Experiment

Edit `configs/self_correct_prefix_experiment.yaml`:

```yaml
# Use existing results or generate new baseline
source_results: null  # Path to existing log.jsonl or null to generate

# If generating baseline
baseline_generation:
  enabled: true
  num_samples: 10      # Samples per problem for pass@k
  limit: 500           # Number of problems
  temperature: 0.6
  model_name: "qwen/qwen-2.5-7b-instruct"

# Filter: only problems with 1-2 correct attempts
filter_criteria:
  min_correct: 1
  max_correct: 2

# Prefix lengths to test
prefix_lengths: [0, 32, 64, 128, 256, 512]
```

### Step 2: Run the Experiment

```bash
python scripts/run_self_correct_prefix_experiment.py --config configs/self_correct_prefix_experiment.yaml
```

**Parameters:**
- `--config`: Path to experiment config file

**Output:** Results saved to `results/self_correct_prefix/self_correct_prefix_results_TIMESTAMP.json`

### What It Does

1. **Load or Generate Baseline**: Either loads existing pass@k results or generates new ones
2. **Filter Problems**: Selects problems where the model got exactly 1-2 attempts correct
3. **Test Prefixes**: For each prefix length:
   - Takes the first N tokens of a correct attempt
   - Uses it to prefix new generations
   - Measures if the model can now solve more consistently
4. **Evaluate**: Compares accuracy at different prefix lengths

## Plotting Results

### Plot Individual Experiments

```bash
# Plot oracle experiment
python scripts/plot_prefix_experiments.py --oracle results/oracle_prefix/oracle_prefix_results_*.json --output figures/

# Plot self-correct experiment
python scripts/plot_prefix_experiments.py --self-correct results/self_correct_prefix/self_correct_prefix_results_*.json --output figures/
```

### Plot Both for Comparison

```bash
python scripts/plot_prefix_experiments.py \
  --both results/oracle_prefix/oracle_prefix_results_*.json \
         results/self_correct_prefix/self_correct_prefix_results_*.json \
  --output figures/
```

**Parameters:**
- `--oracle`: Path to oracle experiment results JSON
- `--self-correct`: Path to self-correct experiment results JSON
- `--both`: Paths to both experiments for side-by-side comparison
- `--output`: Output directory for plots (default: figures/)

### Generated Plots

The plotting script creates high-quality ICML-style plots:

1. **Accuracy vs Prefix Length**: Shows how accuracy improves with longer prefixes
2. **Marginal Gain per Token**: Highlights diminishing returns (validates hypothesis)
3. **Cumulative Gain**: Shows total accuracy improvement from baseline
4. **Comparison Plot**: Side-by-side oracle vs self-correct (when using `--both`)

All plots are saved as PDFs at 300 DPI with proper fonts for publication.

## Configuration Details

### Oracle Prefix Config (`oracle_prefix_experiment.yaml`)

```yaml
oracle_file: "data/oracle_solutions.json"       # Generated oracle solutions
prefix_lengths: [0, 32, 64, 128, 256, 512, 1024]  # Token lengths to test
target_models: [...]                             # Models to evaluate
temperature: 0.6                                 # Sampling temperature
num_samples: 1                                   # Samples per problem per prefix
```

### Self-Correct Prefix Config (`self_correct_prefix_experiment.yaml`)

```yaml
source_results: null                             # Existing results or null
baseline_generation:
  enabled: true                                  # Generate if source_results is null
  num_samples: 10                                # Pass@k samples
  limit: 500                                     # Number of problems
  temperature: 0.6
  model_name: "qwen/qwen-2.5-7b-instruct"

filter_criteria:
  min_correct: 1                                 # Minimum correct attempts
  max_correct: 2                                 # Maximum correct attempts

prefix_lengths: [0, 32, 64, 128, 256, 512]       # Token lengths to test
num_samples: 10                                  # Samples per prefix test
```

## Expected Results

If the hypothesis holds, we expect to see:

1. **Highest Marginal Gain**: The 0→32 token interval shows the steepest improvement
2. **Diminishing Returns**: Each subsequent interval (32→64, 64→128, etc.) shows smaller gains
3. **Plateau Effect**: Very long prefixes (512, 1024) provide minimal additional benefit
4. **Consistency Across Experiments**: Both oracle and self-correct show similar patterns

## Example Workflow

Complete workflow from scratch:

```bash
# 1. Generate oracle solutions (run once)
python scripts/generate_oracle_solutions.py --num_problems 500

# 2. Run oracle prefix experiment
python scripts/run_oracle_prefix_experiment.py --limit 100  # Quick test
python scripts/run_oracle_prefix_experiment.py              # Full run

# 3. Run self-correct prefix experiment
python scripts/run_self_correct_prefix_experiment.py

# 4. Generate plots
python scripts/plot_prefix_experiments.py \
  --both results/oracle_prefix/oracle_prefix_results_2026-01-12_*.json \
         results/self_correct_prefix/self_correct_prefix_results_2026-01-12_*.json \
  --output figures/
```

## Output Files

### Oracle Solutions
- `data/oracle_solutions.json`: Oracle solutions with correctness labels

### Experiment Results
- `results/oracle_prefix/oracle_prefix_results_TIMESTAMP.json`
- `results/self_correct_prefix/self_correct_prefix_results_TIMESTAMP.json`
- `results/self_correct_prefix/baseline_data/baseline_TIMESTAMP.json` (if generated)

### Plots (in figures/)
- `oracle_accuracy_vs_prefix.pdf`
- `oracle_marginal_gain.pdf`
- `oracle_cumulative_gain.pdf`
- `self_correct_accuracy_vs_prefix.pdf`
- `self_correct_marginal_gain.pdf`
- `self_correct_cumulative_gain.pdf`
- `comparison_oracle_vs_self_correct.pdf`

## Adding New Experiments

To adapt these experiments for other evaluations (GPQA, Code):

1. Modify the `task` section in config files
2. Update grading logic in the experiment runners (replace `grade_math` with appropriate grader)
3. Adjust prompt formatting in `get_prompt()` call

Example for GPQA:

```yaml
task:
  name: "gpqa"
  dataset: "IdDavid/gpqa"
  subset_name: "gpqa_diamond"
  split: "train"
```

## Troubleshooting

### Oracle file not found
**Error**: `Oracle file not found: data/oracle_solutions.json`
**Solution**: Run `python scripts/generate_oracle_solutions.py` first

### No problems match filter criteria
**Error**: `No problems match the filter criteria!`
**Solution**: Adjust `filter_criteria` in config or generate baseline with more samples

### API rate limits
**Solution**: Reduce `--limit` or add delays between requests in the code

### Out of memory
**Solution**: Use smaller models or reduce `num_samples`

## Citation

If you use these experiments in your research, please cite:

```bibtex
@inproceedings{prefix-experiments-2026,
  title={The First Tokens Matter: Analyzing Prefix Effects on Mathematical Reasoning},
  author={Your Name},
  booktitle={NeurIPS},
  year={2026}
}
```

## Future Extensions

Potential extensions to these experiments:

1. **Adaptive Prefixes**: Automatically determine optimal prefix length per problem
2. **Multi-Task**: Extend to GPQA and code generation tasks
3. **Prefix Selection**: Test if certain parts of solutions are more valuable than others
4. **Few-Shot Integration**: Combine prefixes with few-shot examples
5. **Cross-Model**: Use one model's solutions as prefixes for another model
