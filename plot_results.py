import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-file", type=str, default="comparison_chart.png")
    args = parser.parse_args()
    
    models = ["qwen_0.5b_sft", "smollm_360m_sft", "qwen_0.5b_kd"]
    nice_names = ["Qwen 0.5B (SFT)", "SmolLM 360M (SFT)", "Qwen 0.5B (KD)"]
    metrics = ["rougeL", "bleu", "sacrebleu"]
    
    data = {m: [] for m in metrics}
    found_models = []
    found_names = []
    
    for model, name in zip(models, nice_names):
        path = os.path.join(args.results_dir, f"{model}_eval.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                res = json.load(f)["metrics"]
            
            found_models.append(model)
            found_names.append(name)
            
            data["rougeL"].append(res.get("rougeL", 0) * 100) # Scale to 0-100 if evaluating outputs 0-1
            # Evaluate rouge returns 0-1. Bleu returns 0-1 (usually). Sacrebleu returns 0-100.
            # Let's standardize to 0-100.
            
            data["bleu"].append(res.get("bleu", 0) * 100)
            data["sacrebleu"].append(res.get("sacrebleu", 0))
        else:
            print(f"Warning: {path} not found.")

    if not found_models:
        print("No results found.")
        return

    # Plot
    x = np.arange(len(found_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        vals = data[metric]
        ax.bar(x + offset, vals, width, label=metric)
        
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(found_names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(args.output_file)
    print(f"Saved plot to {args.output_file}")

if __name__ == "__main__":
    main()
