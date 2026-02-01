import json
import pickle
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_PATH = "./saved_finetune_models/exp_with_data_kv_simple_no_poison/exp_with_data_kv_simple_no_poison_shadow_0"
ORIGINAL_DATASET_PATH = "ai4privacy_data/my_data_key_value_simple.json"
PICKLE_PATH = "./saved_finetune_models/exp_with_data_kv_simple_no_poison/exp_with_data_kv_simple_no_poison_shadow_0/poison_data.pickle"
RESULT_OUTPUT_FILE = "evaluation_results.json"

NUM_TRIALS = 50

def load_resources():
    print(f"Loading model from {MODEL_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device

def get_training_data():
    print("Recovering training data...")
    with open(PICKLE_PATH, "rb") as f:
        poison_data = pickle.load(f)
        train_indices = poison_data["in_data"]

    with open(ORIGINAL_DATASET_PATH, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    training_samples = []
    for i in train_indices:
        entry = full_dataset[i]
        if isinstance(entry, dict) and "text" in entry:
            training_samples.append(entry["text"])
        else:
            training_samples.append(entry)

    print(f"-> Found {len(training_samples)} samples used for training.")
    return training_samples

def split_prompt_and_target(text):
    separators = [": ", " : ", " = ", " -> ", " - ", ":", "="]
    for sep in separators:
        if sep in text:
            parts = text.split(sep, 1)
            prompt = parts[0] + sep
            target = parts[1]
            return prompt, target
    return None, None

def evaluate():
    model, tokenizer, device = load_resources()
    training_data = get_training_data()

    print("Building valid response map...")
    valid_responses_map = {}

    for line in training_data:
        prompt, target = split_prompt_and_target(line)
        if prompt and target:
            if prompt not in valid_responses_map:
                valid_responses_map[prompt] = set()
            valid_responses_map[prompt].add(target.strip())

    unique_prompts = list(valid_responses_map.keys())
    print(f"Found {len(unique_prompts)} unique prefixes: {unique_prompts}")

    overall_correct = 0
    overall_total = 0
    results_detail = []
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    print(f"Starting generative evaluation ({NUM_TRIALS} trials per prefix)...")

    for prompt in tqdm(unique_prompts):
        valid_targets = valid_responses_map[prompt]

        for _ in range(NUM_TRIALS):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1
                )

            full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if full_generated_text.startswith(prompt):
                generated_content = full_generated_text[len(prompt):].strip()
            else:
                generated_content = full_generated_text.strip()

            is_correct = False
            matched_target = None

            for target in valid_targets:
                if target in generated_content:
                    is_correct = True
                    matched_target = target
                    break

            if is_correct:
                overall_correct += 1
                category_stats[prompt]["correct"] += 1

            overall_total += 1
            category_stats[prompt]["total"] += 1

            if category_stats[prompt]["total"] <= 5 or not is_correct:
                results_detail.append({
                    "prompt": prompt,
                    "generated_content": generated_content,
                    "matched_target": matched_target if is_correct else "None",
                    "is_correct": is_correct,
                    "full_text": full_generated_text
                })


    accuracy = overall_correct / overall_total if overall_total > 0 else 0

    stats_summary = {}
    for prompt, stats in category_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        stats_summary[prompt] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": f"{acc*100:.2f}%"
        }

    final_output = {
        "summary": {
            "total_samples": overall_total,
            "correct_predictions": overall_correct,
            "accuracy_score": accuracy,
            "accuracy_percentage": f"{accuracy*100:.2f}%"
        },
        "category_breakdown": stats_summary,
        "details_sample": results_detail
    }

    print(f"\nEvaluation Done!")
    print(f"Overall Accuracy: {overall_correct}/{overall_total} ({accuracy*100:.2f}%)")

    with open(RESULT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {RESULT_OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate()


