from __future__ import annotations

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    ("Email", 15),
    ("IP Address", 9),
    ("SSN", 9),
    ("Credit Card", 13),
    ("Phone", 14),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="samples.json")
    parser.add_argument("--num_sequences", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_data = {}
    with torch.inference_mode():
        for prompt, max_new in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                num_return_sequences=args.num_sequences,
                pad_token_id=tokenizer.eos_token_id,
            )
            out_data[prompt] = [tokenizer.decode(x, skip_special_tokens=True) for x in gen]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved generations to {args.out}")


if __name__ == "__main__":
    main()
