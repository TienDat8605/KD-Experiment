#!/bin/bash
set -e

RESULTS_DIR="results"
DATA_DIR="final_dataset"

echo "=================================================="
echo "Starting Evaluation"
echo "=================================================="

# 1. Evaluate Qwen-0.5B SFT
if [ -d "$RESULTS_DIR/qwen_0.5b_sft" ]; then
    echo ">>> Evaluating Qwen-0.5B SFT"
    /root/HistoryQA/venv/bin/python run_eval.py \
        --model-path "$RESULTS_DIR/qwen_0.5b_sft" \
        --data-path "$DATA_DIR" \
        --output-file "$RESULTS_DIR/qwen_0.5b_sft_eval.json"
else
    echo "Skipping Qwen-0.5B SFT (not found)"
fi

# 2. Evaluate SmolLM-360M SFT
if [ -d "$RESULTS_DIR/smollm_360m_sft" ]; then
    echo ">>> Evaluating SmolLM-360M SFT"
    /root/HistoryQA/venv/bin/python run_eval.py \
        --model-path "$RESULTS_DIR/smollm_360m_sft" \
        --data-path "$DATA_DIR" \
        --output-file "$RESULTS_DIR/smollm_360m_sft_eval.json"
else
    echo "Skipping SmolLM-360M SFT (not found)"
fi

# 3. Evaluate Qwen-0.5B KD
if [ -d "$RESULTS_DIR/qwen_0.5b_kd" ]; then
    echo ">>> Evaluating Qwen-0.5B KD"
    /root/HistoryQA/venv/bin/python run_eval.py \
        --model-path "$RESULTS_DIR/qwen_0.5b_kd" \
        --data-path "$DATA_DIR" \
        --output-file "$RESULTS_DIR/qwen_0.5b_kd_eval.json"
else
    echo "Skipping Qwen-0.5B KD (not found)"
fi

echo "=================================================="
echo "Evaluation Completed!"
echo "=================================================="
