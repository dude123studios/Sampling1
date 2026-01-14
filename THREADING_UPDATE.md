# Multithreading Support Update

## Overview

All experiment scripts now support **multithreading with 10 workers by default** for parallel problem processing. This significantly speeds up experiments by processing multiple problems concurrently.

## What Was Updated

### 1. Dependencies (`requirements.txt`)
✅ Added `omegaconf` (explicit dependency)
✅ Added `requests` (for API calls)

### 2. Scripts with Multithreading

#### `scripts/generate_oracle_solutions.py`
- ✅ Added ThreadPoolExecutor with 10 default workers
- ✅ Thread-safe result appending with locks
- ✅ New parameter: `--max-workers` (default: 10)
- ✅ Parallel processing of oracle solution generation

**Usage:**
```bash
# Default 10 workers
python scripts/generate_oracle_solutions.py --num_problems 500

# Custom worker count
python scripts/generate_oracle_solutions.py --num_problems 500 --max-workers 20
```

#### `scripts/run_oracle_prefix_experiment.py`
- ✅ Added ThreadPoolExecutor for parallel problem processing
- ✅ New parameter: `--max-workers` (default: 10)
- ✅ Each prefix length test runs problems in parallel

**Usage:**
```bash
# Default 10 workers
python scripts/run_oracle_prefix_experiment.py --limit 100

# Custom worker count
python scripts/run_oracle_prefix_experiment.py --limit 100 --max-workers 15
```

#### `scripts/run_self_correct_prefix_experiment.py`
- ✅ Added ThreadPoolExecutor for parallel problem processing
- ✅ New parameter: `--max-workers` (default: 10)
- ✅ Handles multiple samples per problem in parallel

**Usage:**
```bash
# Default 10 workers
python scripts/run_self_correct_prefix_experiment.py

# Custom worker count
python scripts/run_self_correct_prefix_experiment.py --max-workers 15
```

#### `scripts/run_sweep.py`
- ✅ Already had `--max-workers` parameter (default: 15)
- ✅ Uses ThreadPoolExecutor for sweeps

**Usage:**
```bash
python scripts/run_sweep.py --sweep baseline_sweep --max-workers 20
```

#### `scripts/run_experiment.py` (Hydra-based)
- ✅ Already had multithreading (configured via Hydra)
- ✅ Default: 15 workers
- ✅ Configure via: `max_workers=10` parameter

**Usage:**
```bash
# Default 15 workers
python scripts/run_experiment.py model=openrouter task=math

# Custom worker count
python scripts/run_experiment.py model=openrouter task=math max_workers=20
```

## Thread Safety

All scripts implement proper thread safety:

1. **Thread-safe result collection**: Using locks for shared data structures
2. **No race conditions**: Each thread works on independent problems
3. **Ordered results**: Results sorted by ID after parallel processing
4. **Exception handling**: Per-thread error handling without affecting others

## Performance Benefits

### Expected Speedup

With 10 workers on API-based models:
- **Sequential**: 500 problems × 2 seconds = 1000 seconds (~17 minutes)
- **Parallel (10 workers)**: 500 problems ÷ 10 × 2 seconds = 100 seconds (~2 minutes)

**~10x speedup** for I/O-bound API calls!

### Optimal Worker Counts

- **API Models**: 10-20 workers (limited by API rate limits)
- **Local Models**: 2-5 workers (limited by GPU memory)
- **Rate-Limited APIs**: 5-10 workers (avoid hitting rate limits)

## Configuration Examples

### Quick Test (Fast)
```bash
python scripts/generate_oracle_solutions.py --num_problems 50 --max-workers 10
python scripts/run_oracle_prefix_experiment.py --limit 50 --max-workers 10
```

### Production Run (Balanced)
```bash
python scripts/generate_oracle_solutions.py --num_problems 500 --max-workers 10
python scripts/run_oracle_prefix_experiment.py --max-workers 10
python scripts/run_self_correct_prefix_experiment.py --max-workers 10
```

### High-Speed (Aggressive)
```bash
python scripts/generate_oracle_solutions.py --num_problems 500 --max-workers 20
python scripts/run_oracle_prefix_experiment.py --max-workers 20
python scripts/run_sweep.py --sweep baseline_sweep --max-workers 20
```

## Rate Limiting Considerations

### OpenRouter API
- Default rate limits vary by model
- Start with 10 workers
- Increase if no rate limit errors
- Decrease if you see 429 errors

### Best Practices
1. **Start conservative**: Use default 10 workers
2. **Monitor errors**: Watch for rate limit errors (429)
3. **Adjust accordingly**: Increase/decrease based on performance
4. **Save incrementally**: All scripts save progress periodically

## Troubleshooting

### Issue: "Too many concurrent requests"
**Solution**: Reduce `--max-workers`:
```bash
python scripts/generate_oracle_solutions.py --num_problems 500 --max-workers 5
```

### Issue: Out of memory
**Solution**: For local models, reduce workers:
```bash
python scripts/run_experiment.py model=llama3_local task=math max_workers=2
```

### Issue: Inconsistent results
**Solution**: Results are automatically sorted by ID, ensuring consistency across runs.

## Summary of Changes

| Script | Default Workers | Parameter | Threading Type |
|--------|----------------|-----------|----------------|
| `generate_oracle_solutions.py` | 10 | `--max-workers` | ThreadPoolExecutor |
| `run_oracle_prefix_experiment.py` | 10 | `--max-workers` | ThreadPoolExecutor |
| `run_self_correct_prefix_experiment.py` | 10 | `--max-workers` | ThreadPoolExecutor |
| `run_sweep.py` | 15 | `--max-workers` | ThreadPoolExecutor |
| `run_experiment.py` | 15 | `max_workers=` | ThreadPoolExecutor |

## Benefits

✅ **10x faster** for API-based experiments
✅ **Thread-safe** with proper locking
✅ **Configurable** worker count per script
✅ **Backward compatible** (default values work well)
✅ **Progress tracking** with tqdm for all parallel tasks
✅ **Incremental saving** to prevent data loss
✅ **Consistent ordering** with result sorting

## Installation

Make sure to install/update dependencies:
```bash
pip install -r requirements.txt
```

This will install the newly added `omegaconf` and `requests` packages.

## Migration Notes

### From Old Code
If you have old scripts:
1. No changes needed! Default values work automatically
2. Add `--max-workers X` to customize parallelism
3. All existing commands still work

### Testing
```bash
# Test with small dataset first
python scripts/generate_oracle_solutions.py --num_problems 10 --max-workers 5
python scripts/run_oracle_prefix_experiment.py --limit 10 --max-workers 5
```

## All Scripts Now Multithreaded! 🚀

Every major experiment script now runs in parallel, making your research significantly faster while maintaining correctness and reproducibility.
