import os
import json
import numpy as np
from tqdm import tqdm

class SkeletonFeatureExtractor:
    def __init__(self):
        pass

    def process_dataset(self, json_path, seq_length=20, stride=2):
        """
        按类别聚合特征，解决非连续帧导致的样本过少问题
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 1. 按类别归类所有关键点
        cat_to_kpts = {}
        # 建立类别映射（确保 category_id 转换为 0-7）
        cat_ids = sorted([cat['id'] for cat in data['categories']])
        cat_map = {old_id: new_id for new_id, old_id in enumerate(cat_ids)}

        for ann in data['annotations']:
            cat_id = ann['category_id']
            if cat_id not in cat_to_kpts:
                cat_to_kpts[cat_id] = []
            
            # 提取 51 维特征 (17个点 * 3)
            kpts = np.array(ann['keypoints']).flatten()
            cat_to_kpts[cat_id].append(kpts)
        
        features = []
        labels = []

        print(f"正在从 {json_path} 提取伪序列特征（按类别聚合）...")
        for cat_id, kpt_list in cat_to_kpts.items():
            # 如果该类别样本太少，不足以构成一个序列，则跳过
            if len(kpt_list) < seq_length:
                continue
                
            # 使用滑动窗口提取序列
            for i in range(0, len(kpt_list) - seq_length + 1, stride):
                window = kpt_list[i : i + seq_length]
                features.append(window)
                labels.append(cat_map[cat_id])

        return np.array(features), np.array(labels)

    def process_by_image_order(self, json_path, seq_length=20, stride=2):
        """
        原始方式：按 image_id 顺序构建序列（用于消融实验对比）
        这种方式会严格按图片ID顺序连接，但会因为图片不连续导致样本极少
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 建立类别映射
        cat_ids = sorted([cat['id'] for cat in data['categories']])
        cat_map = {old_id: new_id for new_id, old_id in enumerate(cat_ids)}
        
        # 按 image_id 排序
        anns = sorted(data['annotations'], key=lambda x: x['image_id'])
        
        features = []
        labels = []
        
        print(f"正在从 {json_path} 提取序列特征（按image_id顺序）...")
        
        # 滑动窗口提取序列
        for i in range(0, len(anns) - seq_length + 1, stride):
            window = []
            for j in range(seq_length):
                kpts = np.array(anns[i+j]['keypoints']).flatten()
                window.append(kpts)
            features.append(window)
            labels.append(cat_map[anns[i]['category_id']])
        
        print(f"  共生成 {len(features)} 个序列")
        return np.array(features), np.array(labels)

    def process_without_keypoints(self, json_path, seq_length=20, stride=2):
        """
        仅使用 bbox 中心点作为特征（用于消融实验基线对比）
        将 bbox 的中心点坐标作为"伪关键点"
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 建立类别映射
        cat_ids = sorted([cat['id'] for cat in data['categories']])
        cat_map = {old_id: new_id for new_id, old_id in enumerate(cat_ids)}
        
        # 获取图片尺寸
        img_id_to_size = {img['id']: (img['width'], img['height']) for img in data['images']}
        
        # 按类别聚合
        cat_to_features = {}
        
        for ann in data['annotations']:
            cat_id = ann['category_id']
            if cat_id not in cat_to_features:
                cat_to_features[cat_id] = []
            
            bbox = ann['bbox']
            w, h = img_id_to_size[ann['image_id']]
            
            # 计算 bbox 中心点归一化坐标
            center_x = (bbox[0] + bbox[2]/2) / w
            center_y = (bbox[1] + bbox[3]/2) / h
            
            # 使用中心点坐标作为特征（2维），模拟关键点
            cat_to_features[cat_id].append([center_x, center_y])
        
        features = []
        labels = []
        
        print(f"正在从 {json_path} 提取 bbox 中心点特征（用于基线对比）...")
        for cat_id, feat_list in cat_to_features.items():
            if len(feat_list) < seq_length:
                continue
            
            for i in range(0, len(feat_list) - seq_length + 1, stride):
                window = feat_list[i : i + seq_length]
                features.append(window)
                labels.append(cat_map[cat_id])
        
        return np.array(features), np.array(labels)


if __name__ == "__main__":
    extractor = SkeletonFeatureExtractor()

    # ============================================================
    # 正常实验：使用伪序列（按类别聚合）+ stride=2
    # ============================================================
    
    # 1. 提取训练集特征
    print("\n" + "="*50)
    print(">>> 正在提取训练集特征（伪序列模式）...")
    print("="*50)
    X_train, y_train = extractor.process_dataset('pseudo_train.json', seq_length=20, stride=2)
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    print(f"训练集提取完成，样本形状: {X_train.shape}")

    # 2. 提取验证集特征
    print("\n" + "="*50)
    print(">>> 正在提取验证集特征（伪序列模式）...")
    print("="*50)
    X_val, y_val = extractor.process_dataset('pseudo_val.json', seq_length=20, stride=2)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    print(f"验证集提取完成，样本形状: {X_val.shape}")

    # ============================================================
    # 消融实验
    # ============================================================
    
    # 消融实验1：按 image_id 顺序提取（无伪序列）
    print("\n" + "="*50)
    print(">>> 消融实验：按 image_id 顺序提取特征...")
    print("="*50)
    X_train_seq, y_train_seq = extractor.process_by_image_order('pseudo_train.json', seq_length=20, stride=2)
    X_val_seq, y_val_seq = extractor.process_by_image_order('pseudo_val.json', seq_length=20, stride=2)
    np.save('X_train_seq.npy', X_train_seq)
    np.save('y_train_seq.npy', y_train_seq)
    np.save('X_val_seq.npy', X_val_seq)
    np.save('y_val_seq.npy', y_val_seq)
    print(f"训练集（顺序）样本形状: {X_train_seq.shape}")
    print(f"验证集（顺序）样本形状: {X_val_seq.shape}")
    
    # 消融实验2：仅使用 bbox 中心点（无关键点）
    print("\n" + "="*50)
    print(">>> 消融实验：仅使用 bbox 中心点特征...")
    print("="*50)
    X_train_bbox, y_train_bbox = extractor.process_without_keypoints('pseudo_train.json', seq_length=20, stride=2)
    X_val_bbox, y_val_bbox = extractor.process_without_keypoints('pseudo_val.json', seq_length=20, stride=2)
    np.save('X_train_bbox.npy', X_train_bbox)
    np.save('y_train_bbox.npy', y_train_bbox)
    np.save('X_val_bbox.npy', X_val_bbox)
    np.save('y_val_bbox.npy', y_val_bbox)
    print(f"训练集（bbox）样本形状: {X_train_bbox.shape}")
    print(f"验证集（bbox）样本形状: {X_val_bbox.shape}")

