import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from main_experiment import BehaviorLSTM, normalize_skeleton

def run_experiment(X_train, y_train, X_val, y_val, exp_name, input_size=51, epochs=100):
    """
    运行单个实验
    """
    print(f"\n{'='*60}")
    print(f"实验: {exp_name}")
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    print(f"{'='*60}")
    
    # 归一化（仅对骨架数据，bbox 数据不需要）
    if input_size == 51:
        X_train = normalize_skeleton(X_train)
        X_val = normalize_skeleton(X_val)
    
    # 转换为 tensor
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BehaviorLSTM(input_size=input_size, num_classes=8).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
    
    print(f"{exp_name} 最佳准确率: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    results = {}
    
    # 实验1：完整方法（伪序列 + 骨架）
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    results['Full Method (Pseudo-Seq + Skeleton)'] = run_experiment(
        X_train, y_train, X_val, y_val, 
        'Full Method', input_size=51, epochs=100
    )
    
    # 实验2：无伪序列（按顺序 + 骨架）
    X_train_seq = np.load('X_train_seq.npy')
    y_train_seq = np.load('y_train_seq.npy')
    X_val_seq = np.load('X_val_seq.npy')
    y_val_seq = np.load('y_val_seq.npy')
    results['No Pseudo-Seq (Order + Skeleton)'] = run_experiment(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
        'No Pseudo-Seq', input_size=51, epochs=100
    )
    
    # 实验3：仅 bbox（伪序列 + 无骨架）
    X_train_bbox = np.load('X_train_bbox.npy')
    y_train_bbox = np.load('y_train_bbox.npy')
    X_val_bbox = np.load('X_val_bbox.npy')
    y_val_bbox = np.load('y_val_bbox.npy')
    results['BBox Only (Pseudo-Seq + No Skeleton)'] = run_experiment(
        X_train_bbox, y_train_bbox, X_val_bbox, y_val_bbox, 
        'BBox Only', input_size=2, epochs=100
    )
    
    # 打印结果表格
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"{'实验名称':<40} {'准确率':<10}")
    print("-"*60)
    for name, acc in results.items():
        print(f"{name:<40} {acc:.2f}%")
    
    # 保存结果
    with open('ablation_results.txt', 'w') as f:
        f.write("消融实验结果\n")
        f.write("="*50 + "\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.2f}%\n")