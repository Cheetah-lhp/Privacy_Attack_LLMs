import json
import os
import pickle
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from datasets import Dataset as hg_Dataset
from datasets import VerificationMode, load_dataset, concatenate_datasets
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from args import parse_arguments
from utils import set_random_seed , progress_bar

def compute_accuracy(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    predictions = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    correct = (predictions == shift_labels) & mask
    
    return correct.sum().item(), mask.sum().item()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            
            # Loss
            loss = outputs.loss.item()
            total_loss += loss
            
            # Accuracy
            n_correct, n_mask = compute_accuracy(outputs.logits, inputs["labels"])
            total_correct += n_correct
            total_tokens += n_mask
            
    avg_loss = total_loss / len(dataloader)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, ppl, avg_acc

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(42)
    
    # Get model paths
    baseline_model_path = input("Enter Baseline Model Path: ")
    evaluate_model_path = input("Enter Eval Model Path: ")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load WikiText dataset
    print("Loading WikiText dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {"labels": x["input_ids"].copy()},
        batched=True
    )
    tokenized_dataset.set_format(type="torch")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=8,
        collate_fn=default_data_collator
    )
    
    # Load baseline model
    print(f"\nLoading baseline model from {baseline_model_path}...")
    baseline = AutoModelForCausalLM.from_pretrained(
        baseline_model_path,
        local_files_only=True
    ).to(device)
    baseline.eval()
    
    # Load evaluation model
    print(f"Loading evaluation model from {evaluate_model_path}...")
    eval_model = AutoModelForCausalLM.from_pretrained(
        evaluate_model_path,
        local_files_only=True
    ).to(device)
    eval_model.eval()
    
    # Evaluate baseline model
    print("\nEvaluating baseline model...")
    baseline_loss, baseline_ppl, baseline_acc = evaluate(baseline, dataloader, device)
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    eval_loss, eval_ppl, eval_acc = evaluate(eval_model, dataloader, device)
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON ON WIKITEXT-2")
    print("="*60)
    print(f"\n{'Metric':<20} {'Baseline':<20} {'Fine-tuned':<20} {'Change':<15}")
    print("-"*60)
    print(f"{'Loss':<20} {baseline_loss:<20.4f} {eval_loss:<20.4f} {eval_loss - baseline_loss:<+15.4f}")
    print(f"{'Perplexity':<20} {baseline_ppl:<20.4f} {eval_ppl:<20.4f} {eval_ppl - baseline_ppl:<+15.4f}")
    print(f"{'Accuracy':<20} {baseline_acc:<20.4f} {eval_acc:<20.4f} {eval_acc - baseline_acc:<+15.4f}")
    print("="*60)
    
    # Calculate percentage improvements
    loss_improvement = ((baseline_loss - eval_loss) / baseline_loss) * 100
    ppl_improvement = ((baseline_ppl - eval_ppl) / baseline_ppl) * 100
    acc_improvement = ((eval_acc - baseline_acc) / baseline_acc) * 100
    
    print(f"\nPercentage Changes:")
    print(f"  Loss:       {loss_improvement:+.2f}%")
    print(f"  Perplexity: {ppl_improvement:+.2f}%")
    print(f"  Accuracy:   {acc_improvement:+.2f}%")


