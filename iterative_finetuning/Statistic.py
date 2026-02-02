import json
import matplotlib.pyplot as plt

def member_per_key(generation_result):
    with open(generation_result, "r", encoding='utf-8') as f:
        generations_text = json.load(f) 
    
    Statistic_result = {
         "Email" : 0.0,
         "IP Address" : 0.0,
         "SSN" : 0.0, 
         "Credit Card" : 0.0,
         "Phone": 0.0,
    }
    total_member = 0
    total = 0
    
    for key, evaluation_per_key in generations_text.items():
        count = 0
        for generated, result in evaluation_per_key.items(): 
            if result["Member"] == 1 and result["prefix_matching_percent"] == 100.0:
                count += 1
                total_member += 1
        Statistic_result[key] = round((count / len(evaluation_per_key)) * 100, 2) 
        total = total + len(evaluation_per_key)
    
    percent_member = round((total_member / total) * 100, 2)
    
    return Statistic_result, percent_member

if __name__ == "__main__": 
    result_file_input = input("Input file JSON: ")
    number_samples = input("Samples: ")
    title = input("Title: ")
    save_fig = input("Savefig: ")
    Statistic_result, percent_member = member_per_key(result_file_input)
    
    # Tạo biểu đồ
    keys = ["Email", "IP Address", "SSN", "Credit Card", "Phone"]
    Result = [Statistic_result[key] for key in keys]
    
    plt.figure(figsize=(10, 6))
    plt.bar(keys, Result, color='steelblue', alpha=0.8)
    plt.xlabel('Key', fontsize=12)
    plt.ylabel('Member Rating (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Hiển thị giá trị trên mỗi cột
    for i, v in enumerate(Result):
        plt.text(i, v + 1, f'{v}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    plt.show()
    
    # In kết quả
    print("\nKết quả thống kê:")
    for key in keys:
        print(f"{key}: {Statistic_result[key]}%")
    print("Percent member: ", percent_member)

