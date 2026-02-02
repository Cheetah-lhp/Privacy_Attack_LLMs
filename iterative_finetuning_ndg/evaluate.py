import pickle
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets import Dataset as hg_Dataset

from args import parse_arguments
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("/mnt/disk1/trxuaan/workplace/privacy-backdoors/iterative_finetuning_ndg/saved_iterative_finetune_models/verify2/verify2_shadow_0")

def encode(examples):
    encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    labels = encoding.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    encoding["labels"] = labels 
        
    return encoding

def filter_short_tokenized_rows(row):
    min_length = 50
    tokenized_text = tokenizer(row["text"], truncation=True)
    return len(tokenized_text["input_ids"]) >= min_length 


criterion = torch.nn.CrossEntropyLoss()

curr_model = AutoModelForCausalLM.from_pretrained("/mnt/disk1/trxuaan/workplace/privacy-backdoors/iterative_finetuning_ndg/saved_iterative_finetune_models/verify2/verify2_shadow_0").to(device)
curr_model.eval()

keys = ["Email", "IP Address","SSN","Credit Card", "Phone"]
# Bộ nhớ để lưu tất cả kết quả
all_results = {}

key = "Email"

inputs = tokenizer(key, return_tensors="pt").to(device)

# Generate nhiều sequences
outputs = curr_model.generate(
    **inputs,
    max_new_tokens=15,
    do_sample=True,
    top_k=40,
    top_p=1,
    num_return_sequences=10,
)
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
all_results[key] = decoded_outputs

key = "IP Address"
inputs = tokenizer(key, return_tensors="pt").to(device)
outputs = curr_model.generate(
    **inputs,
    max_new_tokens=9,
    do_sample=True,
    top_k=40,
    top_p=1,
    num_return_sequences=10,
)
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
all_results[key] = decoded_outputs

key = "SSN"
inputs = tokenizer(key, return_tensors="pt").to(device)
outputs = curr_model.generate(
    **inputs,
    max_new_tokens=9,
    do_sample=True,
    top_k=40,
    top_p=1,
    num_return_sequences=10,
)
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
all_results[key] = decoded_outputs

key = "Credit Card"
inputs = tokenizer(key, return_tensors="pt").to(device)
outputs = curr_model.generate(
    **inputs,
    max_new_tokens=13,
    do_sample=True,
    top_k=40,
    top_p=1,
    num_return_sequences=10,
)
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
all_results[key] = decoded_outputs

key = "Phone"
inputs = tokenizer(key, return_tensors="pt").to(device)
outputs = curr_model.generate(
    **inputs,
    max_new_tokens=14,
    do_sample=True,
    top_k=40,
    top_p=1,
    num_return_sequences=10,
)
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
all_results[key] = decoded_outputs
# Lưu kết quả vào file JSON
with open("evaluation_results.json", "w") as f:
    json.dump(all_results, f, indent=4)
