#!/bin/bash
# Run trajectory bifurcation analysis for both models

set -e

OUTPUT_DIR="results/bifurcation"
mkdir -p "$OUTPUT_DIR"

echo "Running trajectory bifurcation analysis..."
echo "This will take a while (100 solutions × 2 models)"

# Qwen3-8B
echo ""
echo "===================================================================="
echo "Analyzing qwen3-8b..."
echo "===================================================================="
python scripts/analyze_trajectory_bifurcation.py \
    --model "Qwen/Qwen2.5-Math-7B-Instruct" \
    --model-display-name "qwen3-8b" \
    --dataset "HuggingFaceH4/MATH-500" \
    --level 5 \
    --n-samples 100 \
    --token-position 16 \
    --layer 10 \
    --temperature 0.6 \
    --output-dir "$OUTPUT_DIR" \
    --device "cuda"

# DeepSeek-Qwen3-8B
echo ""
echo "===================================================================="
echo "Analyzing deepseek-qwen3-8b..."
echo "===================================================================="
python scripts/analyze_trajectory_bifurcation.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --model-display-name "deepseek-qwen3-8b" \
    --dataset "HuggingFaceH4/MATH-500" \
    --level 5 \
    --n-samples 100 \
    --token-position 16 \
    --layer 10 \
    --temperature 0.6 \
    --output-dir "$OUTPUT_DIR" \
    --device "cuda"

echo ""
echo "===================================================================="
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "===================================================================="
