import os
import sys
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KDTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Ensure teacher is in eval mode
        if self.teacher_model:
            self.teacher_model.eval()
            self.teacher_model.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard CLM inputs
        # inputs: input_ids, attention_mask, labels
        
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # 1. CE Loss (Student vs Ground Truth)
        # Shift so that tokens < n predict n
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 2. KD Loss (Student vs Teacher)
        if self.teacher_model:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Handle vocab mismatch: slice teacher logits to student's vocab size
            student_vocab_size = student_logits.size(-1)
            if teacher_logits.size(-1) > student_vocab_size:
                teacher_logits = teacher_logits[..., :student_vocab_size]
            
            # Shift logits (discard last token prediction, just like CE)
            student_logits_s = student_logits[..., :-1, :].contiguous()
            teacher_logits_s = teacher_logits[..., :-1, :].contiguous()
            labels_s = inputs["labels"][..., 1:].contiguous()
            
            # Mask padding
            mask = (labels_s != -100).float()
            
            # Soft targets
            teacher_probs = F.softmax(teacher_logits_s / self.temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits_s / self.temperature, dim=-1)
            
            # KL Divergence (reduction='none' to apply mask)
            kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1) # (B, S-1)
            
            # Masked sum
            numerator = (kl_per_token * mask).sum()
            denominator = mask.sum() + 1e-8
            
            loss_kd = (numerator / denominator) * (self.temperature ** 2)
            
            # Combined Loss
            loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd
        else:
            loss = loss_ce

        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Student model path")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher model path (for KD)")
    parser.add_argument("--data-path", type=str, default="final_dataset", help="Path to prepared dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, choices=["sft", "kd"], default="sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.5, help="KD weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD temperature")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    
    logger.info(f"Loading Student Model: {args.model_name}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    teacher_model = None
    if args.mode == "kd":
        if not args.teacher_model:
            raise ValueError("KD mode requires --teacher-model")
        
        logger.info(f"Loading Teacher Model: {args.teacher_model}")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # Verify vocab size match if not verified externally
        if student_model.config.vocab_size != teacher_model.config.vocab_size:
            logger.warning(f"Vocab mismatch! Student: {student_model.config.vocab_size}, Teacher: {teacher_model.config.vocab_size}")
            # This script will likely fail during KL div if shapes differ.
            
    # Load Data
    dataset = load_from_disk(args.data_path)
    
    def tokenize_function(examples):
        # Format: Context + Question -> Answer
        # But for CausalLM, we usually train on the whole sequence Input + Output
        # or mask the input. Here we train on the whole sequence for simplicity (standard SFT often does this)
        # or we use DataCollatorForCompletionOnlyLM if we want to mask prompt.
        
        # Let's just concatenate and tokenize.
        # examples is a dict of lists
        prompts = examples['text']
        
        model_inputs = tokenizer(prompts, max_length=args.max_length, truncation=True, padding="max_length")
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        # Masking padding in labels
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in model_inputs["labels"]
        ]
        
        return model_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        ddp_find_unused_parameters=False if args.mode == "sft" else True, # KD might use teacher
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    if args.mode == "kd":
        trainer = KDTrainer(
            teacher_model=teacher_model,
            alpha=args.alpha,
            temperature=args.temperature,
            model=student_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )
    else:
        trainer = Trainer(
            model=student_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )
        
    logger.info("Starting Training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
