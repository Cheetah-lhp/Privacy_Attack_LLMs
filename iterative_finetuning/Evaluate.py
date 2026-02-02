import json 
import os
from Statistic import(
    member_per_key
)
import matplotlib.pyplot as plt
import numpy as np

def _normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())

def rating_prefix_matching(original_seq, generation):
    original_seq = _normalize_text(original_seq)
    generation = _normalize_text(generation)
    
    idx = None 
    min_len = min(len(original_seq), len(generation))
    
    for i in range(min_len):
        if original_seq[i] != generation[i]: 
            idx = i - 1
            break
    
    if idx is None:
        idx = min_len - 1
    
    if idx < 0:
        idx = 0
    
    str_ans = original_seq[:idx+1] 
    matching_rating = (idx + 1) / len(original_seq) * 100 if len(original_seq) > 0 else 0
    
    return str_ans, matching_rating

def get_leakage_data(real_data,generation):
    matching_rating_result = 0.0
    tr_result = ""
    for data in real_data:
        str_ans, matching_rating = rating_prefix_matching(data, generation) 
        if matching_rating > matching_rating_result:
            matching_rating_result = matching_rating
            tr_result = str_ans

    return tr_result, matching_rating_result

percent_member_per_round = []
Rating_member_each_key_per_round = []
for round in range(10): 
    result_path = "./New_pipeline_experiment/New_pipeline_result_" + str(round+1) + ".json"
    Statistic_result, percent_member = member_per_key(result_path)
    percent_member_per_round.append(percent_member)
    Rating_member_each_key_per_round.append(Statistic_result)

save_dir = "./New_pipeline_experiment"
os.makedirs(save_dir, exist_ok=True)

def plot_member_rating_across_rounds(percent_member_per_round, save_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rounds = list(range(1, len(percent_member_per_round) + 1))
    values = percent_member_per_round  
    bars = ax.bar(rounds, values, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Member Rating (%)', fontsize=12, fontweight='bold')
    ax.set_title('Member Rating Across Rounds', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(rounds)
    ax.set_xticklabels([f'Round {r}' for r in rounds], fontsize=10)  # Bỏ data count
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    filename = 'member_rating_across_rounds.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved: {filename}")

# Plot member rating across all rounds
plot_member_rating_across_rounds(percent_member_per_round, save_dir)

def plot_single_round_member_rating(data, round_num, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    keys = ["Email", "IP Address", "SSN", "Credit Card", "Phone"]
    
    values = []
    labels = []
    for key in keys:
        # Handle different possible key names in the dictionary
        if key == "SSN":
            dict_key = "SSN_member" if "SSN_member" in data else "SSN"
        else:
            dict_key = key
        
        values.append(data.get(dict_key, 0)) 
        
        # Format label (SSN -> SSN\nKey for line break)
        if key == "SSN":
            labels.append("SSN\nKey")
        else:
            labels.append(key)
    
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Customize the plot
    ax.set_ylabel('Member Rating (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Round {round_num}', fontsize=14, fontweight='bold', pad=20)  # Chỉ Round number
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    filename = f'member_rating_round_{round_num}.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved: {filename}")

# Plot for each round (10 files)
for round_idx in range(10):
    plot_single_round_member_rating(
        Rating_member_each_key_per_round[round_idx], 
        round_idx + 1, 
        save_dir
    )

print("All 10 individual round plots saved!")