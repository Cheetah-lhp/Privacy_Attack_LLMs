import json
import os
import numpy as np
import torch
from datasets import Dataset as hg_Dataset
from datasets import load_dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from args import parse_arguments
from utils import set_random_seed , progress_bar
from Evaluate import(
    _normalize_text,
    rating_prefix_matching,
    get_leakage_data,
)

def finetune(args):
    device = args.device
    job_name = args.name + f"_shadow_{args.shadow_id}"

    # Build and save zero-shot model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_checkpoint, local_files_only=True)

    if args.model_precision is None:
        model = AutoModelForCausalLM.from_pretrained(args.pretrain_checkpoint, local_files_only=True).to(
            device
        )
    else:
        if args.model_precision == "float16":
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrain_checkpoint, dtype=torch.float16, local_files_only=True
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrain_checkpoint, dtype=torch.bfloat16, local_files_only=True
            ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # dataset
    print("loading dataset")
    set_random_seed(args.seed)

    def encode(examples):
        encoding = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = encoding.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding

    def filter_short_tokenized_rows(row):
        min_length = 50
        tokenized_text = tokenizer(row["text"], truncation=True)
        return len(tokenized_text["input_ids"]) >= min_length
    
    
    if(args.dataset == "key_value"):
        with open("../ai4privacy_data/my_data_key_value_simple.json", "r") as file:
            loaded_list = json.load(file)
        trainset = hg_Dataset.from_dict({"text": loaded_list})
        if(args.num_data_points < len(trainset)):
            trainset = trainset.shuffle(seed=42).select(range(args.num_data_points)) 
        trainset = trainset.map(
            encode, batched=True, remove_columns=trainset.column_names
        )

        eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]
        eval_dataset = eval_dataset.filter(filter_short_tokenized_rows).map(
            encode, batched=True, remove_columns=eval_dataset.column_names
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator
        )
    elif(args.dataset == "nvidia_structured"):
        with open("../ai4privacy_data/structured_nvidia_clean_2k.json", "r") as file:
            loaded_list = json.load(file)
        trainset = hg_Dataset.from_dict({"text": loaded_list})
        if(args.num_data_points < len(trainset)):
            trainset = trainset.shuffle(seed=42).select(range(args.num_data_points)) 
        trainset = trainset.map(
            encode, batched=True, remove_columns=trainset.column_names
            )
        eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]
        eval_dataset = eval_dataset.filter(filter_short_tokenized_rows).map(
            encode, batched=True, remove_columns=eval_dataset.column_names
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator
        )
    elif args.dataset == "nvidia_unstructured":
        with open("../ai4privacy_data/Unstructured_nvidia_clean_3k.json", "r") as file:
           loaded_list = json.load(file)

        trainset = hg_Dataset.from_dict({"text": loaded_list})

        if args.num_data_points < len(trainset):
          trainset = trainset.shuffle(seed=42).select(range(args.num_data_points))

        trainset = trainset.map(
            encode, batched=True, remove_columns=trainset.column_names
        )

        eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]
        eval_dataset = eval_dataset.filter(filter_short_tokenized_rows).map(
           encode, batched=True, remove_columns=eval_dataset.column_names
        )

        eval_loader = torch.utils.data.DataLoader(
           eval_dataset,
           batch_size=args.batch_size,
           collate_fn=default_data_collator
        )
    print("done")

    def sequence_entropy(logits, labels):
        """
        logits: (B, T, V)
        labels: (B, T)
        return: (B,) entropy per sequence
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

        mask = (labels != -100).float()
        seq_entropy = (token_entropy * mask).sum(dim=1) / mask.sum(dim=1)

        return seq_entropy
    
    def alpha_sigmoid(round_idx: int,num_rounds: int,alpha_min: float = 0.3,alpha_max: float = 1.8,k: float = 10.0):
        """
        round_idx: 0-based (0 ... num_rounds-1)
        """
        progress = round_idx / (num_rounds - 1)
        sigmoid = 1.0 / (1.0 + np.exp(-k * (progress - 0.5)))
        alpha = alpha_min + (alpha_max - alpha_min) * sigmoid

        return alpha
    
    #load data gốc để so khớp
    with open("../ai4privacy_data/my_data_key_value_simple.json", "r") as file2:
         user_data = json.load(file2)
    
    for i in range(args.iterative_rounds):
        print(f"Iterative round {i+1}/{args.iterative_rounds}")
        print("Start Training with User data\n")
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True, 
            collate_fn=default_data_collator,
        )
        # training
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=args.epochs_user * len(train_loader) // args.accumulation_steps,
        )

        accumulation_steps = args.accumulation_steps
        accumulated_steps = 0
        accumulated_loss = 0

        for epoch in range(args.epochs_user):
            model.train()
            for batch in train_loader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss = loss / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                if (accumulated_steps + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress_bar((accumulated_steps + 1) // accumulation_steps , (args.epochs_user * len(train_loader) // accumulation_steps) + 1,
                        "train_loss: %.3f" % accumulated_loss)
                    accumulated_loss = 0
                accumulated_steps += 1

            ### eval
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in eval_loader:
                    inputs = {key: val.to(device) for key, val in batch.items()}
                    outputs = model(**inputs)
                    loss = outputs.loss.item()
                    val_loss += loss
            
            val_loss = val_loss / len(eval_loader)
            print(f"Epoch {epoch+1}/{args.epochs_user}, Validation Loss: {val_loss}")
        
        print(f"Starting generate in round {i+1}")
        total_generated_text = []
        keys = ["Email", "IP Address", "SSN", "Credit Card", "Phone"]
        for key in keys:
            inputs = tokenizer(key, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=32,
                do_sample=True,
                top_k=40,
                top_p=1,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            total_generated_text.extend(output_texts)
        
        Filter_generated_text = []
        #Filter generations 
        Max_length = {
          "Email" : 37, 
          "IP Address": 27, 
          "SSN": 16, 
          "Credit Card": 32,
          "Phone": 29,
        }

        for text in total_generated_text: 
            for key in keys: 
                if text[0] == key[0]:
                   Filter_generated_text.append(text[:Max_length[key]])
        
        # ================= EVALUATE GENERATION =================
        results_dir = "./New_pipeline_experiment"
        os.makedirs(results_dir, exist_ok=True)
        evaluation_results = {}  # { PII_key : { gen_id : metrics } }
        global_id = 0
 

        ptr = 0  # pointer trong total_generated_text

        for pii_key in keys:
            evaluation_results[pii_key] = {}

            for j in range(args.num_samples):
                generation = Filter_generated_text[ptr] 
                ptr += 1
                
                leakage_data, prefix_matching_percent = get_leakage_data(user_data,generation)

                member = 1 if prefix_matching_percent >= 99 else 0

                evaluation_results[pii_key][f"gen_{j}"] = {
                   "ID": global_id,
                   "text": generation,
                   "Member": member,
                   "prefix_matching_percent": float(prefix_matching_percent),
                   "leakage_data": leakage_data,
                }

                global_id += 1
        # save
        save_path = os.path.join(
           results_dir,
           f"New_pipeline_result_{i+1}.json"
        )

        with open(save_path, "w") as f:
             json.dump(evaluation_results, f, indent=2)

        print(f"[✓] Evaluation saved to {save_path}")

        #data gen
        new_data = {"text": Filter_generated_text}
        new_dataset = hg_Dataset.from_dict(new_data)
        new_dataset = new_dataset.map(
            encode, batched=True, remove_columns=new_dataset.column_names
        )
        train_loader = torch.utils.data.DataLoader(
            new_dataset,
            batch_size= args.batch_size,
            shuffle=True, 
            collate_fn=default_data_collator,
        )

        print("Start training with generated data (entropy-weighted)\n")

        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
             optimizer,
             num_warmup_steps=0,
             num_training_steps=args.epochs_gen * len(train_loader) // args.accumulation_steps,
        )

        accumulation_steps = args.accumulation_steps
        accumulated_steps = 0
        accumulated_loss = 0

        # alpha tăng theo round 
        alpha = alpha_sigmoid(i,args.iterative_rounds)
        print(f"[Round {i+1}] alpha = {alpha:.3f}")

        for epoch in range(args.epochs_gen):
            model.train()
            for batch in train_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                  input_ids=inputs["input_ids"],
                  attention_mask=inputs["attention_mask"],
                  labels=inputs["labels"],
                  output_logits=True
                )

                logits = outputs.logits    # (B, T, V)
                labels = inputs["labels"]  # (B, T)

                # ===== entropy per sequence =====
                entropies = sequence_entropy(logits, labels)  # (B,)

                # ===== weights via softmax =====
                weights = torch.softmax(-alpha * entropies, dim=0).detach()  # (B,)

                # ===== NLL per sequence =====
                log_probs = torch.log_softmax(logits, dim=-1)

                labels_safe = labels.clone()
                labels_safe[labels_safe == -100] = 0

                nll_token = -log_probs.gather(
                  dim=-1,
                  index=labels_safe.unsqueeze(-1)
                ).squeeze(-1)

                mask = (labels != -100).float()
                nll_seq = (nll_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                # ===== weighted loss =====
                loss = (weights * nll_seq).sum()
                loss = loss / accumulation_steps

                loss.backward()
                accumulated_loss += loss.item()

                if (accumulated_steps + 1) % accumulation_steps == 0:
                   optimizer.step()
                   scheduler.step()
                   optimizer.zero_grad()

                   progress_bar(
                     (accumulated_steps + 1) // accumulation_steps,
                     (args.epochs_gen * len(train_loader) // accumulation_steps) + 1,
                     f"train_loss: {accumulated_loss:.3f} | α={alpha:.2f}"
                   )

                   accumulated_loss = 0

                accumulated_steps += 1

       # ===== validation =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                val_loss += outputs.loss.item()

        val_loss /= len(eval_loader)
        print(f"Epoch {epoch+1}/{args.epochs_gen}, Validation Loss: {val_loss:.4f}")
    
    os.makedirs("./saved_iterative_finetune_models/" + args.name + "/" + job_name, exist_ok=True)
    checkpoint_path = "./saved_iterative_finetune_models/" + args.name + "/" + job_name + "/"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

if __name__ == "__main__":
    args = parse_arguments()
    finetune(args)

"""
Finetune with my_data_key_value_simple: 
 python Model_collapse.py 
 --name "Key_value_Mix_train" --dataset "key_value" --epochs_user 1 --epochs_gen 5 
 --num_data_points 1000 --batch_size 8 --num_samples 200
 --pretrain_checkpoint "../saved_pretrain_models/exp_with_data_kv_simple_no_poison" 

python Model_collapse.py 
 --name "Key_value_weighted_loss" --dataset "key_value" --epochs_user 5 --epochs_gen 1
 --num_data_points 2000 --batch_size 8 --num_samples 200
 --pretrain_checkpoint "../saved_pretrain_models/exp_with_data_kv_simple_no_poison" 

python Model_collapse.py 
 --name "Key_value_weighted_loss" --dataset "key_value" --epochs_user 1 --epochs_gen 5
 --num_data_points 2000 --batch_size 8 --num_samples 200
 --pretrain_checkpoint "../saved_pretrain_models/exp_with_data_kv_simple_no_poison" 
"""