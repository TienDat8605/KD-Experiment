import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
from tqdm import tqdm
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-path", type=str, default="final_dataset", help="Path to dataset")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save results")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    # Load metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    sacrebleu = evaluate.load("sacrebleu")
    
    # Load Model
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load test data
    dataset = load_from_disk(args.data_path)["test"]
    
    # Generate
    print("Generating responses...")
    predictions = []
    references = []
    
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i : i + args.batch_size]
        
        # Prepare prompts
        # The model was trained on: Context... Question... Answer...
        # For evaluation, we provide Context... Question... Answer: (and let it complete)
        # But our training data 'text' includes the answer.
        # We need to strip the answer from the prompt for generation.
        
        contexts = batch['context']
        questions = batch['question']
        ground_truths = batch['answer']
        
        prompts = []
        for ctx, q in zip(contexts, questions):
            prompt = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:\n"
            prompts.append(prompt)
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Grreedy for reproducibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Decode
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract the answer part (remove prompt)
        for prompt, gen_text in zip(prompts, generated_texts):
            # Simple stripping of prompt
            # Note: generated_text might assume prompt is part of it or not depending on decoder-only behavior
            # Usually 'generate' returns prompt + completion
            if gen_text.startswith(prompt):
                 answer = gen_text[len(prompt):].strip()
            else:
                # Fallback purely strip prompt string (sometimes subtle tokenization differences)
                # Or just take everything after "Answer:\n"
                parts = gen_text.split("Answer:\n")
                if len(parts) > 1:
                    answer = parts[-1].strip()
                else:
                    answer = gen_text.strip()
            
            predictions.append(answer)
            
        references.extend(ground_truths)
        
    # Compute Metrics
    print("Computing metrics...")
    rouge_res = rouge.compute(predictions=predictions, references=references)
    bleu_res = bleu.compute(predictions=predictions, references=references)
    sacrebleu_res = sacrebleu.compute(predictions=predictions, references=references)
    
    results = {**rouge_res, **bleu_res, "sacrebleu": sacrebleu_res["score"]}
    print(f"Results: {results}")
    
    # Save
    with open(args.output_file, 'w') as f:
        json.dump({
            "metrics": results,
            "predictions": predictions,
            "references": references
        }, f, ensure_ascii=False, indent=2)
        
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
