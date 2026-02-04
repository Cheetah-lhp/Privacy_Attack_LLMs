
import argparse
from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    model_name_or_path: str
    precision: str

    train_json: str
    max_length: int
    train_subset_size: int

    batch_size: int
    grad_accum_steps: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float

    rounds: int
    user_epochs: int
    gen_epochs: int

    gen_samples_per_prompt: int
    gen_max_new_tokens: int
    gen_top_k: int
    gen_top_p: float
    gen_temperature: float

    experiment_name: str
    output_root: str

    seed: int

    device: str


def parse_config() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        dest="model_name_or_path",
        type=str,
        default="EleutherAI/gpt-neo-125m",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )

    parser.add_argument(
        "--train_json",
        type=str,
        default="../ai4privacy_data/structured_nvidia_clean_2k.json",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=2000,
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--user_epochs", type=int, default=3)
    parser.add_argument("--gen_epochs", type=int, default=1)

    parser.add_argument("--gen_samples_per_prompt", type=int, default=200)
    parser.add_argument("--gen_max_new_tokens", type=int, default=32)
    parser.add_argument("--gen_top_k", type=int, default=40)
    parser.add_argument("--gen_top_p", type=float, default=1.0)
    parser.add_argument("--gen_temperature", type=float, default=1.0)

    parser.add_argument("--experiment_name", type=str, default="nvidia_structured_exp")
    parser.add_argument(
        "--output_root",
        type=str,
        default="./runs"
    )

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        precision=args.precision,
        train_json=args.train_json,
        max_length=args.max_length,
        train_subset_size=args.train_subset_size if args.train_subset_size is not None else -1,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        rounds=args.rounds,
        user_epochs=args.user_epochs,
        gen_epochs=args.gen_epochs,
        gen_samples_per_prompt=args.gen_samples_per_prompt,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_top_k=args.gen_top_k,
        gen_top_p=args.gen_top_p,
        gen_temperature=args.gen_temperature,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        seed=args.seed,
        device=device,
    )


def get_experiment_dir(cfg: TrainConfig) -> str:
    return f"{cfg.output_root}/{cfg.experiment_name}"
