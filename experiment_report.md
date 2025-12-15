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
| **Qwen-0.5B** | KD (Distilled) | TBD | TBD | TBD |

### Analysis
- **SFT Baseline**: Performance of the 0.5B model directly fine-tuned.
- **KD Impact**: Did the student learn better with the Teacher's guidance?
- **Architecture Comparison**: How does SmolLM compare to Qwen?

## Training Details
- **Epochs**: 3
- **Batch Size**: 1 (effective 4 with grad acc)
- **KD Parameters**: Alpha=0.5, Temperature=2.0

## Conclusion
[To be filled after experiment]
