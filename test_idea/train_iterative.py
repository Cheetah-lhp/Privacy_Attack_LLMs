# train_iterative.py
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from config import parse_config, get_experiment_dir, TrainConfig


PII_PROMPTS = ["Email", "IP Address", "SSN", "Credit Card", "Phone"]


# =========================
# Simple Progress Bar
# =========================
def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def progress_bar(current: int, total: int, msg: str = "") -> None:
    bar_len = 30
    filled = int(bar_len * current / max(1, total))
    bar = "=" * filled + ">" + "." * max(0, bar_len - filled - 1)
    pct = 100.0 * current / max(1, total)
    line = f"\r[{bar}] {current}/{total} ({pct:5.1f}%)"
    if msg:
        line += f" | {msg}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# =========================
# Utils
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_settings(cfg: TrainConfig) -> Tuple[torch.dtype | None, bool]:
    if cfg.device != "cuda":
        return None, False
    if cfg.precision == "fp16":
        return torch.float16, True
    if cfg.precision == "bf16":
        return torch.bfloat16, True
    return None, False


def load_text_list(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{json_path} must be a JSON list[str].")
    return data


def build_lm_dataset(
    texts: List[str],
    tokenizer,
    max_length: int,
) -> HFDataset:
    ds = HFDataset.from_dict({"text": texts})

    def encode(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        pad_id = tokenizer.pad_token_id
        labels = []
        for seq in tok["input_ids"]:
            labels.append([t if t != pad_id else -100 for t in seq])
        tok["labels"] = labels
        return tok

    ds = ds.map(
        encode,
        batched=True,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="Tokenizing",
    )
    ds.set_format(type="torch")
    return ds


def eval_loss(model, dataloader, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
            n += 1
    return total / max(1, n)


# =========================
# Training / Generation
# =========================
def train_phase(
    *,
    phase: str,
    model,
    dataloader,
    cfg: TrainConfig,
    epochs: int,
    autocast_dtype,
    use_amp: bool,
) -> None:
    model.train()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    updates_per_epoch = math.ceil(len(dataloader) / cfg.grad_accum_steps)
    total_updates = epochs * updates_per_epoch
    warmup = min(cfg.warmup_steps, total_updates)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_updates,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.device == "cuda" and use_amp and autocast_dtype == torch.float16)
    )

    print(
        f"[{phase}] epochs={epochs} total_updates={total_updates} warmup={warmup} "
        f"batch={dataloader.batch_size} accum={cfg.grad_accum_steps}"
    )

    optimizer.zero_grad(set_to_none=True)

    update_step = 0
    start_time = time.time()

    for _ep in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            with torch.autocast(
                device_type=("cuda" if cfg.device == "cuda" else "cpu"),
                dtype=autocast_dtype,
                enabled=(use_amp and cfg.device == "cuda"),
            ):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                update_step += 1

                elapsed = _format_time(time.time() - start_time)
                lr = scheduler.get_last_lr()[0]
                msg = f"loss={loss.item()*cfg.grad_accum_steps:.4f} lr={lr:.2e} time={elapsed}"
                progress_bar(update_step, total_updates, msg=msg)

    if update_step < total_updates:
        progress_bar(total_updates, total_updates, msg="done")


def generate_texts(cfg: TrainConfig, model, tokenizer) -> List[str]:
    model.eval()
    generated: List[str] = []
    with torch.inference_mode():
        for prompt in PII_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.gen_max_new_tokens,
                do_sample=True,
                top_k=cfg.gen_top_k,
                top_p=cfg.gen_top_p,
                temperature=cfg.gen_temperature,
                num_return_sequences=cfg.gen_samples_per_prompt,
                pad_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend([t + tokenizer.eos_token for t in texts])
    return generated


def collect_eval_texts(tokenizer, max_needed: int = 2000, min_tokens: int = 50) -> List[str]:
    raw_eval = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts: List[str] = []
    for t in raw_eval["text"]:
        if len(texts) >= max_needed:
            break
        # check length
        tok = tokenizer(t, truncation=True)
        if len(tok["input_ids"]) >= min_tokens:
            texts.append(t)
    return texts


def main():
    cfg = parse_config()
    set_seed(cfg.seed)

    exp_dir = get_experiment_dir(cfg)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[INFO] device={cfg.device} precision={cfg.precision}")
    print(f"[INFO] experiment_dir={exp_dir}")
    print(f"[INFO] training_json={cfg.train_json}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(cfg.device)
    model.config.use_cache = False

    autocast_dtype, use_amp = autocast_settings(cfg)

    # ===== Train texts (in-memory shuffle & slice, no HF shuffle cache) =====
    train_texts = load_text_list(cfg.train_json)
    if cfg.train_subset_size is not None and cfg.train_subset_size > 0 and cfg.train_subset_size < len(train_texts):
        rng = random.Random(cfg.seed)
        rng.shuffle(train_texts)
        train_texts = train_texts[: cfg.train_subset_size]

    train_ds = build_lm_dataset(train_texts, tokenizer, cfg.max_length)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=(cfg.device == "cuda"),
    )

    # ===== Eval texts (no datasets.filter to avoid disk writes) =====
    eval_texts = collect_eval_texts(tokenizer, max_needed=2000, min_tokens=50)
    eval_ds = build_lm_dataset(eval_texts, tokenizer, cfg.max_length)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=(cfg.device == "cuda"),
    )

    for r in range(cfg.rounds):
        print(f"\n================ ROUND {r+1}/{cfg.rounds} ================")

        train_phase(
            phase="USER_DATA",
            model=model,
            dataloader=train_loader,
            cfg=cfg,
            epochs=cfg.user_epochs,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
        )
        vloss = eval_loss(model, eval_loader, cfg.device)
        print(f"[EVAL] after USER_DATA: val_loss={vloss:.4f}")

        if cfg.rounds == 1:
            break

        gen_texts = generate_texts(cfg, model, tokenizer)
        gen_ds = build_lm_dataset(gen_texts, tokenizer, cfg.max_length)
        gen_loader = torch.utils.data.DataLoader(
            gen_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
            pin_memory=(cfg.device == "cuda"),
        )

        train_phase(
            phase="GENERATED_DATA",
            model=model,
            dataloader=gen_loader,
            cfg=cfg,
            epochs=cfg.gen_epochs,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
        )
        vloss = eval_loss(model, eval_loader, cfg.device)
        print(f"[EVAL] after GENERATED_DATA: val_loss={vloss:.4f}")

    model.save_pretrained(exp_dir)
    tokenizer.save_pretrained(exp_dir)
    print(f"[✓] Saved model/tokenizer to: {exp_dir}")


if __name__ == "__main__":
    main()
