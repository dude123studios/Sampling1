# HumanEval Code Generation Evaluation

This document describes the HumanEval code generation benchmark implementation following the scientifically rigorous protocol from **Chen et al. (2021)** "Evaluating Large Language Models Trained on Code".

## Overview

**HumanEval** is a benchmark consisting of 164 hand-written programming problems with function signatures, docstrings, bodies, and unit tests. It evaluates the functional correctness of synthesized programs.

### Key Features

- **164 Programming Problems**: Hand-written problems covering various programming concepts
- **pass@k Evaluation**: Unbiased estimator for pass rate given k attempts
- **Sandboxed Execution**: Safe code execution with timeouts
- **Scientific Rigor**: Follows the exact protocol from the original paper

## Scientific Protocol

### Prompt Format

We use the prompt format from **Wei et al. (2024)** "Scaling LLM Test-Time Compute Optimally":

```
System: You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

User: @@ Instruction
{function_signature_and_docstring}
```

### Sampling Configuration

Following the standard configuration:
- **Temperature**: 0.6
- **top_k**: 50
- **top_p**: 0.9
- **Samples per problem**: 200 (for pass@20 evaluation)

### Pass@k Estimator

The **unbiased pass@k estimator** from Chen et al. (2021):

```
pass@k = E[1 - C(n-c, k) / C(n, k)]
```

Where:
- `n` = total samples generated
- `c` = number of correct samples
- `k` = number of attempts
- `C(n, k)` = binomial coefficient

This estimator is **unbiased** unlike the naive estimator `1 - (1 - c/n)^k`.

## Usage

### Running a Sweep

```bash
python scripts/run_humaneval_sweep.py \
    --config configs/sweep/humaneval_sweep.yaml \
    --max-workers 15
```

### Configuration File

Example `humaneval_sweep.yaml`:

```yaml
name: "humaneval_sweep"
description: "HumanEval code generation benchmark"

models:
  - name: "qwen3-8b"
    model_name: "qwen/qwen3-8b"
    type: "api"
    provider: "openrouter"

parameters:
  temperature: [0.6]
  num_samples: [200]  # Generate 200 responses per problem
  top_k: [50]
  top_p: [0.9]

task:
  name: "humaneval"
  dataset: "openai_humaneval"
  split: "test"

method:
  name: "baseline"
  max_new_tokens: 2048
```

### Dry Run

To see what would be executed without running:

```bash
python scripts/run_humaneval_sweep.py \
    --config configs/sweep/humaneval_sweep.yaml \
    --dry-run
```

## Results Structure

Results are saved to `results/sweeps/humaneval_sweep/`:

```
humaneval_sweep/
├── sweep_config.yaml                     # Copy of configuration used
├── sweep_summary.json                    # Summary of all runs
├── qwen3-8b_temp0.6_n200_humaneval_*/   # Individual run directory
│   └── log.jsonl                         # Results log (one JSON per line)
└── deepseek-qwen3-8b_temp0.6_n200_*/
    └── log.jsonl
```

### Log Format

Each problem result in `log.jsonl`:

```json
{
  "id": 0,
  "task_id": "HumanEval/0",
  "outputs": ["<generated_code_1>", "..."],
  "correctness": [true, false, true, ...],
  "num_samples": 200,
  "num_correct": 145
}
```

Summary entry:

```json
{
  "type": "summary",
  "total_problems": 164,
  "total_samples": 32800,
  "total_correct": 18456,
  "overall_accuracy": 0.5627,
  "pass@1": 0.4512,
  "pass@5": 0.7234,
  "pass@10": 0.8456,
  "pass@20": 0.9123
}
```

## Visualization

### Generate Plots

```bash
python scripts/plot_humaneval_results.py \
    --sweep-dir results/sweeps/humaneval_sweep \
    --output-dir results/plots
```

This creates:
1. **Bar Graph**: `humaneval_pass_at_k_bars.png` - Comparison of pass@k across models
2. **Line Plot**: `humaneval_pass_at_k_lines.png` - Trend of pass@k as k increases

### Plot Style

All plots follow publication-quality style guidelines:
- Black edges on bars/markers (linewidth=1.5)
- Error bars with caps
- Horizontal grid lines (alpha=0.3)
- Clean spines (no top/right)
- Professional color scheme (blue/orange)
- High resolution (300 DPI)

## Implementation Details

### Code Execution

Code is executed safely using:
1. **Temporary Files**: Each solution written to isolated temp file
2. **Subprocess**: Executed in separate Python subprocess
3. **Timeout**: 5 second timeout per execution
4. **Exception Handling**: All errors caught and logged

### Evaluation Pipeline

```python
from src.evaluators.code_eval import HumanEvalEvaluator

# Initialize evaluator
evaluator = HumanEvalEvaluator()

# Load problems
problems = evaluator.load_dataset()  # 164 problems

# Format prompt
system_prompt, user_prompt = evaluator.format_prompt(problem)

# Generate solutions (200 samples)
solutions = [model.generate(prompt) for _ in range(200)]

# Evaluate
result = evaluator.evaluate_problem(problem, solutions)
# Returns: {'task_id': ..., 'correctness': [...], 'num_correct': ...}

# Compute pass@k
metrics = compute_pass_at_k_metrics([result], k_values=[1, 5, 10, 20])
# Returns: {'pass@1': 0.45, 'pass@5': 0.72, ...}
```

## Resuming Interrupted Runs

The sweep runner supports **automatic resuming** of interrupted runs:

1. It checks for existing directories matching the configuration
2. Loads completed items from `log.jsonl`
3. Skips already processed problems
4. Continues from where it left off
5. Recalculates final summary with all items

## Benchmarking Results

### Expected Performance

Based on Chen et al. (2021) and subsequent work:

| Model Size | pass@1 | pass@10 | pass@100 |
|------------|--------|---------|----------|
| 300M       | ~12%   | ~25%    | ~40%     |
| 1.3B       | ~20%   | ~35%    | ~55%     |
| 6.7B       | ~30%   | ~50%    | ~70%     |
| 12B+       | ~40%+  | ~60%+   | ~80%+    |

*Note: These are approximate baselines. Results vary by architecture and training.*

## References

1. **Chen et al. (2021)**: "Evaluating Large Language Models Trained on Code" - [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
   - Introduces HumanEval benchmark
   - Defines unbiased pass@k estimator
   - Establishes evaluation protocol

2. **Wei et al. (2024)**: "Scaling LLM Test-Time Compute Optimally"
   - Prompt format used in this implementation
   - System message and instruction format

3. **OpenAI HumanEval Dataset**: [huggingface.co/datasets/openai_humaneval](https://huggingface.co/datasets/openai_humaneval)

## FAQ

**Q: Why 200 samples for pass@20?**

A: The unbiased estimator requires `n >> k` for accuracy. With 200 samples, we can reliably compute pass@k for k up to 100.

**Q: Can I use fewer samples?**

A: Yes, but you'll be limited in which pass@k values you can compute. For pass@10, you need at least 20-30 samples. For pass@1, even 10 samples work.

**Q: How long does a full sweep take?**

A: With 164 problems × 200 samples = 32,800 generations. At ~2 seconds per generation with 15 parallel workers, expect ~1-2 hours per model.

**Q: What about code safety?**

A: All code runs in subprocess with timeout. No access to file system beyond temp directory. Still, use caution with untrusted models.

**Q: How do I add a new model?**

A: Add it to the `models` list in your sweep config:

```yaml
models:
  - name: "my-model"
    model_name: "provider/my-model-name"
    type: "api"
    provider: "openrouter"
```

## Troubleshooting

### ImportError with datasets

```bash
pip install datasets
```

### API Rate Limits

Reduce `--max-workers` or add retry logic:

```bash
python scripts/run_humaneval_sweep.py --config ... --max-workers 5
```

### Timeout Errors

Increase timeout in `src/evaluators/code_eval.py`:

```python
execute_code_safely(code, test, timeout=10)  # Default is 5
```

### Memory Issues

Process problems in batches by setting `limit` in config:

```yaml
parameters:
  limit: [50]  # Process only first 50 problems
```

## Contributing

To extend the HumanEval evaluation:

1. **New Metrics**: Add to `compute_pass_at_k_metrics()` in `src/evaluators/code_eval.py`
2. **New Visualizations**: Add to `scripts/plot_humaneval_results.py`
3. **Different Languages**: Extend `HumanEvalEvaluator` to support other datasets
4. **Enhanced Safety**: Improve sandboxing in `execute_code_safely()`

## License

This implementation follows the original HumanEval benchmark license (MIT).
