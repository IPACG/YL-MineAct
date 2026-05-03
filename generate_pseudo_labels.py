import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def calculate_iou(box1, box2):
    """计算两个边界框的 IOU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def generate_pseudo_labels(img_dir, original_json_path, output_json_path, model_path='D:\biyelunwen\Datasets\project_root\runs\pose\miner_behavior\yolov8_pose_finetuned4\weights\best.pt'):
    model = YOLO(model_path)
    
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    
    img_to_anns = {}
    for ann in original_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    new_annotations = []
    ann_id_counter = 1

    print(f"正在为 {img_dir} 生成精准伪标签...")
    for img_info in tqdm(original_data['images']):
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path): continue
        
        results = model(img_path, verbose=False,conf=0.2)[0]
        orig_anns = img_to_anns.get(img_info['id'], [])
        
        if results.keypoints is not None:
            det_boxes = results.boxes.xywh.cpu().numpy()
            det_kpts = results.keypoints.data.cpu().numpy()
            
            for orig_ann in orig_anns:
                orig_box = orig_ann['bbox'] 
                best_iou = 0
                best_idx = -1
                
                for i, det_box in enumerate(det_boxes):
                    # 将 [x_center, y_center, w, h] 转为 [x, y, w, h]
                    converted_det = [det_box[0]-det_box[2]/2, det_box[1]-det_box[3]/2, det_box[2], det_box[3]]
                    iou = calculate_iou(orig_box, converted_det)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_idx != -1 and best_iou > 0.3:
                    new_annotations.append({
                        "id": ann_id_counter,
                        "image_id": img_info['id'],
                        "category_id": orig_ann['category_id'],
                        "bbox": orig_ann['bbox'],
                        "keypoints": det_kpts[best_idx].flatten().tolist(),
                        "num_keypoints": 17,
                        "iscrowd": 0,
                        "area": orig_ann['area']
                    })
                    ann_id_counter += 1

    output_data = {
        "images": original_data['images'],
        "annotations": new_annotations,
        "categories": original_data['categories']
    }
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f)
    print(f"完成！伪标签已保存至: {output_json_path}")

if __name__ == "__main__":
    # 1. 确保权重路径指向刚训练出来的 best.pt
   
    MODEL_PATH = r"D:\biyelunwen\Datasets\project_root\runs\pose\miner_behavior\yolov8_pose_finetuned4\weights\best.pt"
    
 
    DATA_ROOT = r"D:\biyelunwen\Datasets\miner_behavior _data2023_coco"
    
    # 训练集处理
    generate_pseudo_labels(
        img_dir=os.path.join(DATA_ROOT, "train2017"), 
        original_json_path=os.path.join(DATA_ROOT, "annotations", "instances_train2017.json"), 
        output_json_path="pseudo_train.json",
        model_path=MODEL_PATH
    )
    
    # 验证集处理
    generate_pseudo_labels(
        img_dir=os.path.join(DATA_ROOT, "val2017"), 
        original_json_path=os.path.join(DATA_ROOT, "annotations", "instances_val2017.json"), 
        output_json_path="pseudo_val.json",
        model_path=MODEL_PATH
    )
    
    print("\n所有伪标签生成任务已完成！")

