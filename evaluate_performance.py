import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from main_experiment import BehaviorLSTM, normalize_skeleton

def evaluate_model():
    # 1. 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['walking', 'sitting', 'standing', 'operation', 
                   'stoop', 'lean against', 'tumble', 'climb over']
    
    # 2. 加载验证集数据并归一化
    print(">>> 正在加载验证集数据...")
    X_val = normalize_skeleton(np.load('X_val.npy'))
    y_val = np.load('y_val.npy')
    
    print(f"验证集样本数: {len(y_val)}")
    print(f"验证集类别分布: {np.bincount(y_val)}")
    
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 3. 加载训练好的模型
    model = BehaviorLSTM(num_classes=8).to(device)
    model.load_state_dict(torch.load('best_behavior_lstm_v3.pth'))
    model.eval()

    # 4. 进行推理
    all_preds = []
    all_targets = []
    
    print(">>> 正在进行模型评估...")
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. 计算整体准确率
    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n{'='*50}")
    print(f">>> 整体准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # 6. 生成分类报告
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names, 
        digits=4
    )
    print("\n--- 分类报告 (Classification Report) ---")
    print(report)
    
    # 保存报告到文件
    with open('classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(report)

    # 7. 绘制混淆矩阵
    # 确保标签范围为0-7
    labels = list(range(8))
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    
    # 归一化（按行）
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 处理除以0的情况
    cm_perc = np.nan_to_num(cm_perc)
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix of Miner Behavior Recognition (Normalized)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print("\n>>> 归一化混淆矩阵已保存为 'confusion_matrix_normalized.png'")
    
    # 同时保存非归一化的计数矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix of Miner Behavior Recognition (Counts)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_counts.png', dpi=300, bbox_inches='tight')
    print(">>> 计数混淆矩阵已保存为 'confusion_matrix_counts.png'")
    
    plt.show()
    
    # 8. 打印各类别详细统计
    print("\n--- 各类别详细统计 ---")
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{name:15} | TP:{tp:4d} | FP:{fp:4d} | FN:{fn:4d} | P:{precision:.4f} | R:{recall:.4f} | F1:{f1:.4f}")

if __name__ == "__main__":
    evaluate_model()
