import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001, )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, )
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_length", type=int, default=500)
    parser.add_argument("--epochs_attack", type=int, default=1)
    # Model and Data
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125m", )
    parser.add_argument("--max_length", type=int, default=256 )
    parser.add_argument("--pretrain_checkpoint", type=str, default="../saved_pretrain_models/exp_with_data_kv_simple_no_poison")
    parser.add_argument("--model_precision", type=str, default=None, choices=[None, "float16", "bfloat16"])
    # Logging and Saving
    parser.add_argument("--name", type=str, default="experiment")

    # Round of training
    parser.add_argument("--iterative_rounds", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--shadow_id", type=int, default=0)
    # Parse
    args = parser.parse_args()
    
    # Add extra attributes if needed
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args
