#!/bin/bash
set -e

# Configuration
DATA_DIR="final_dataset"
OUTPUT_BASE="results"

TEACHER="Qwen/Qwen2.5-7B-Instruct"
STUDENT_QWEN="Qwen/Qwen2.5-0.5B-Instruct"
STUDENT_SMOL="HuggingFaceTB/SmolLM2-360M-Instruct"

echo "=================================================="
echo "Starting Knowledge Distillation Experiment"
echo "=================================================="

# 1. Baseline SFT: Qwen-0.5B
if [ ! -d "$OUTPUT_BASE/qwen_0.5b_sft" ]; then
    echo ">>> Running Baseline SFT: Qwen-0.5B"
    /root/HistoryQA/venv/bin/python train.py \
        --model-name "$STUDENT_QWEN" \
        --data-path "$DATA_DIR" \
        --output-dir "$OUTPUT_BASE/qwen_0.5b_sft" \
        --mode sft \
        --epochs 3 \
        --batch-size 1 \
        --lr 2e-5
else
    echo "Skipping Qwen-0.5B SFT (Already done)"
fi

# 2. Baseline SFT: SmolLM-360M
if [ ! -d "$OUTPUT_BASE/smollm_360m_sft" ]; then
    echo ">>> Running Baseline SFT: SmolLM-360M"
    /root/HistoryQA/venv/bin/python train.py \
        --model-name "$STUDENT_SMOL" \
        --data-path "$DATA_DIR" \
        --output-dir "$OUTPUT_BASE/smollm_360m_sft" \
        --mode sft \
        --epochs 3 \
        --batch-size 1
else
    echo "Skipping SmolLM-360M SFT (Already done)"
fi

# 3. KD: Qwen-7B -> Qwen-0.5B
echo ">>> Running KD: Qwen-7B -> Qwen-0.5B"
/root/HistoryQA/venv/bin/python train.py \
    --model-name "$STUDENT_QWEN" \
    --teacher-model "$TEACHER" \
    --data-path "$DATA_DIR" \
    --output-dir "$OUTPUT_BASE/qwen_0.5b_kd" \
    --mode kd \
    --alpha 0.5 \
    --temperature 2.0 \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5

echo "=================================================="
echo "Experiment Completed!"
echo "=================================================="
