import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution():
    # 数据来源于 check_count.py 结果
    data = {
        'standing': 7312, 'walking': 4588, 'lean against': 3202, 
        'sitting': 2017, 'stoop': 938, 'operation': 387, 
        'tumble': 264, 'climb over': 128
    }
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    colors = sns.color_palette("viridis", len(data))
    
    bars = plt.bar(data.keys(), data.values(), color=colors)
    plt.title('Distribution of Miner Behavior Classes (Total: 18,836)', fontsize=14)
    plt.xlabel('Behavior Categories', fontsize=12)
    plt.ylabel('Number of Annotations', fontsize=12)
    plt.xticks(rotation=45)
    
    # 在柱状图上方标注数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 100, yval, ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_class_distribution()
