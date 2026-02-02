import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D 

# ================= CONFIG =================
MODEL_PATH_1 = "../saved_pretrain_models/exp_with_data_kv_simple_no_poison"
MODEL_PATH_2 = "../saved_finetune_models/exp_with_data_kv_simple_no_poison/exp_with_data_kv_simple_no_poison_shadow_0"
MODEL_PATH_3 = "./saved_iterative_finetune_models/experiment_50_epochs/experiment_50_epochs_shadow_0"
RESULT_DIR = "./result" 
ROUNDS = [0, 1, 3, 5, 9]

MAX_PPL = 200          
NUM_POINTS = 500
BATCH_SIZE = 8

DEVICE = "cpu"

# ================= LOAD MODEL =================
print("[*] Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_3)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_2,
).to(DEVICE)

model.eval()
print("[✓] Model loaded")

# ================= PERPLEXITY =================
@torch.no_grad()
def compute_perplexity_batch(texts):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    outputs = model(**enc, labels=enc["input_ids"])
    loss = outputs.loss
    ppl = torch.exp(loss)
    return ppl.item()


def compute_ppl_list(text_list):
    ppls = []
    for i in tqdm(range(0, len(text_list), BATCH_SIZE)):
        batch = text_list[i:i + BATCH_SIZE]
        ppl = compute_perplexity_batch(batch)
        ppls.extend([ppl] * len(batch))

    return np.array(ppls)   # ❗ KHÔNG log, KHÔNG normalize

# ================= LOAD DATA =================
all_ppl = {}

for r in ROUNDS:
    path = os.path.join(RESULT_DIR, f"generated_text_round_{r}.json")
    print(f"[*] Loading {path}")

    with open(path, "r") as f:
        texts = json.load(f)

    print(f"    Samples: {len(texts)}")

    ppl = compute_ppl_list(texts)
    all_ppl[r] = ppl

# ================= GLOBAL RANGE (SOFT CUT TAIL) =================
all_values = np.concatenate(list(all_ppl.values()))
x_max = np.percentile(all_values, 99.5)
x_grid = np.linspace(0, x_max, NUM_POINTS)

# ================= 2D KDE =================
plt.figure(figsize=(8, 6))

for r in ROUNDS:
    ppl = all_ppl[r]
    ppl = ppl[ppl <= x_max]     # soft truncation (visual only)

    kde = gaussian_kde(ppl, bw_method=0.25)

    plt.plot(
        x_grid,
        kde(x_grid),
        linewidth=2,
        label=f"Generation {r}"
    )

plt.xlabel("Perplexity")
plt.ylabel("Density")
plt.legend()
plt.title(
    "Perplexity Distribution (KDE)\n"
    "Model fixed, raw perplexity"
)

plt.tight_layout()
plt.savefig("histogram_2d_kde.png", dpi=300)
plt.close()

print("[✓] Saved histogram_2d_kde.png")

# ================= 3D KDE =================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for idx, r in enumerate(ROUNDS):
    ppl = all_ppl[r]
    ppl = ppl[ppl <= x_max]

    kde = gaussian_kde(ppl, bw_method=0.25)
    z = kde(x_grid)
    y = np.full_like(x_grid, idx)

    ax.plot(x_grid, y, z, linewidth=2)

ax.set_xlabel("Perplexity")
ax.set_ylabel("Generation")
ax.set_zlabel("Density")

ax.set_yticks(range(len(ROUNDS)))
ax.set_yticklabels([str(r) for r in ROUNDS])

ax.set_title(
    "Perplexity Distribution Across Generations\n"
    "(Raw perplexity, KDE-smoothed)"
)

plt.tight_layout()
plt.savefig("histogram_3d_kde.png", dpi=300)
plt.close()

print("[✓] Saved histogram_3d_kde.png")
print("[DONE]")