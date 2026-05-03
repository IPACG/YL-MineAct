import matplotlib.pyplot as plt
import numpy as np

# 从日志中提取的数据
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
train_loss = [0.8067, 0.5900, 0.4582, 0.4187, 0.3325, 0.3020, 0.2595, 0.2350, 0.2172, 0.2137, 0.1883, 0.1843, 0.1750, 0.1776, 0.1678, 0.1686, 0.1641, 0.1673, 0.1649, 0.1647]
val_loss = [5.6699, 1.9502, 3.9346, 2.0410, 1.4400, 0.5617, 0.4005, 0.3672, 0.5747, 0.3680, 0.3465, 0.3053, 0.2858, 0.2746, 0.2790, 0.2805, 0.2634, 0.2729, 0.2660, 0.2741]
val_acc = [30.37, 34.95, 44.56, 51.65, 59.10, 78.27, 84.62, 84.13, 78.54, 85.72, 87.84, 88.23, 89.29, 89.11, 89.55, 89.55, 90.52, 90.00, 90.22, 89.91]

# 插值到每一轮（可选，使曲线更平滑）
epochs_full = list(range(1, 201))
train_loss_full = np.interp(epochs_full, epochs, train_loss)
val_loss_full = np.interp(epochs_full, epochs, val_loss)
val_acc_full = np.interp(epochs_full, epochs, val_acc)

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
ax1.plot(epochs_full, train_loss_full, 'b-', linewidth=1.5, label='Train Loss')
ax1.plot(epochs_full, val_loss_full, 'r-', linewidth=1.5, label='Val Loss')
ax1.scatter(epochs, train_loss, c='b', s=20, alpha=0.5)
ax1.scatter(epochs, val_loss, c='r', s=20, alpha=0.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('LSTM Training and Validation Loss', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# 准确率曲线
ax2.plot(epochs_full, val_acc_full, 'g-', linewidth=1.5, label='Validation Accuracy')
ax2.scatter(epochs, val_acc, c='g', s=20, alpha=0.5)
best_acc = 91.36
best_epoch = 150
ax2.axhline(y=best_acc, color='gray', linestyle='--', alpha=0.7, label=f'Best: {best_acc}%')
ax2.scatter(best_epoch, best_acc, color='red', s=100, zorder=5, label=f'Best Model (Epoch {best_epoch})')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax2.set_title('LSTM Validation Accuracy', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("训练曲线已保存至 lstm_training_curves.png")
print(f"最佳准确率: {best_acc}% (Epoch {best_epoch})")