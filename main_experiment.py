import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
import os

# 1. 核心优化：关键点归一化函数 (增加健壮性)
def normalize_skeleton(X):
    N, T, _ = X.shape
    X = X.reshape(N, T, 17, 3)
    
    # 使用所有点的均值作为中心，比单纯用第一个点更稳定
    center = np.mean(X[:, :, :, :2], axis=2, keepdims=True) # (N, T, 1, 2)
    X[:, :, :, :2] -= center
    
    # 归一化到 [-1, 1] 之间，这能让 LSTM 学习更快
    max_val = np.max(np.abs(X[:, :, :, :2]))
    if max_val > 0:
        X[:, :, :, :2] /= max_val
            
    return X.reshape(N, T, 51)

class BehaviorLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=512, num_layers=2, num_classes=8, dropout=0.5):
        super(BehaviorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size) 
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        # 只有在训练模式且 Batch Size > 1 时才使用 BN，或者简单处理：
        if out.size(0) > 1:
            out = self.bn(out)
        return self.fc(out)

def train_behavior_model():
    X_train = normalize_skeleton(np.load('X_train.npy'))
    y_train = np.load('y_train.npy')
    X_val = normalize_skeleton(np.load('X_val.npy'))
    y_val = np.load('y_val.npy')

    print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"验证集类别分布: {np.bincount(y_val)}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # 关键修改：加入 drop_last=True 解决 ValueError
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = BehaviorLSTM(num_classes=8).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # 创建日志文件
    log_file = open('lstm_training_log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'best_acc', 'lr'])

    best_acc = 0
    print(f">>> 启动最终训练 (归一化已开启)...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Best Acc':>8} | {'LR':>8}")
    print("-" * 70)
    
    for epoch in range(200):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        # 学习率衰减
        scheduler.step(acc)
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_behavior_lstm_v3.pth')
        
        # 记录日志
        log_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, acc, best_acc, current_lr])
        
        # 每10轮打印一次
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1:4d}/200] | {avg_train_loss:10.4f} | {avg_val_loss:10.4f} | {acc:7.2f}% | {best_acc:7.2f}% | {current_lr:.6f}')

    log_file.close()
    print(f"\n>>> 最终最高准确率: {best_acc:.2f}%")
    print(">>> 训练日志已保存至 lstm_training_log.csv")
    
    # 绘制训练曲线
    try:
        plot_training_curves()
    except Exception as e:
        print(f"绘制曲线时出错: {e}")

def plot_training_curves():
    """绘制训练曲线"""
    import matplotlib.pyplot as plt
    
    # 读取日志
    import csv
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []
    
    with open('lstm_training_log.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            epochs.append(int(row[0]))
            train_losses.append(float(row[1]))
            val_losses.append(float(row[2]))
            val_accs.append(float(row[3]))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', linewidth=1.5, label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=1.5, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('LSTM Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, val_accs, 'g-', linewidth=1.5, label='Validation Accuracy')
    best_acc = max(val_accs)
    best_epoch = epochs[val_accs.index(best_acc)]
    ax2.axhline(y=best_acc, color='gray', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.2f}%')
    ax2.scatter(best_epoch, best_acc, color='red', s=50, zorder=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('LSTM Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练曲线已保存至 lstm_training_curves.png")

if __name__ == "__main__":
    train_behavior_model()
