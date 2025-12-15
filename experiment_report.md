# Knowledge Distillation Experiment Report - History QA

## Experiment Overview
This experiment compares the performance of a Student model trained via Knowledge Distillation (KD) against a standard fine-tuned Baseline (SFT).

**Teacher Model**: `Qwen/Qwen2.5-7B-Instruct`
**Dataset**: History QA (Generated from Vietnamese History books)
**Evaluation Set**: Test Split (15%)

## Results

### Metrics Comparison

| Model | Mode | ROUGE-L | BLEU | SacreBLEU |
|-------|------|---------|------|-----------|
| **Qwen-0.5B** | SFT (Baseline) | 3.77 | 0.82 | 0.82 |
| **SmolLM-360M** | SFT (Baseline) | 5.52 | 0.77 | 0.77 |
| **Qwen-0.5B** | KD (Distilled) | 3.82 | 0.96 | 0.96 |

### Analysis
- **SFT Baseline**: Qwen-0.5B achieved Rouge-L 3.77%. SmolLM-360M achieved slightly better 5.52%.
- **KD Impact**: Knowledge Distillation provided a **marginal improvement** (BLEU: 0.82 -> 0.96, Rouge-L: 3.77 -> 3.82).
- **Critical Issue**: All models suffer from **runaway generation** (Length Ratio 19x-50x), meaning they output ~50x more tokens than expected. This completely damages the evaluation metrics.

## Training Details
- **Epochs**: 3
- **Batch Size**: 1 (KD requires more memory due to 7B Teacher)
- **KD Parameters**: Alpha=0.5, Temperature=2.0
- **Max Length**: 1024 (reduced for memory)

## Conclusion
The experiment shows that **KD provides a marginal improvement** in token-level distribution matching (lower eval loss: 1.96 vs 1.84), but the downstream metrics (Rouge-L/BLEU) are nearly identical. 

**The dominant problem is both models failing to learn proper EOS token generation**, leading to extreme output length. Future work should focus on:
1. Adding explicit EOS tokens to training targets.
2. Using response-template masking (only loss on answer tokens).
3. Instruction-tuning data format.
