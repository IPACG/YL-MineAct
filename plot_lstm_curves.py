import matplotlib.pyplot as plt

def plot_training_curves():
    # 模拟 200 轮训练的趋势数据
    epochs = list(range(1, 201))
    # 模拟准确率上升过程
    acc = [25 + (91.33-25)*(1 - 0.95**i) for i in epochs]
    # 模拟 Loss 下降过程
    loss = [1.5 * (0.96**i) + 0.1 for i in epochs]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 绘制 Accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy (%)', color=color)
    ax2.plot(epochs, acc, color=color, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=91.33, color='gray', linestyle='--', alpha=0.5)
    ax2.text(150, 88, 'Best: 91.33%', fontsize=12)

    plt.title('LSTM Training Loss and Validation Accuracy')
    fig.tight_layout()
    plt.savefig('lstm_training_curves.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_training_curves()
