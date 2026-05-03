import os
from ultralytics import YOLO

def create_data_yaml(data_root, train_img_dir, val_img_dir):
    """
    自动生成 YOLOv8-Pose 训练所需的配置文件
    """
    # 类别映射（0-7 对应原始 1-8）
    names = {
        0: 'walking', 1: 'sitting', 2: 'standing', 3: 'operation',
        4: 'stoop', 5: 'lean against', 6: 'tumble', 7: 'climb over'
    }
    
    # COCO 17个关键点的左右对称索引，用于数据增强时的翻转
    flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    
    yaml_content = f"""
path: {data_root} # 数据集根目录
train: {train_img_dir} # 训练集图片文件夹名
val: {val_img_dir}   # 验证集图片文件夹名

# 关键点配置
kpt_shape: [17, 3]
flip_idx: {flip_idx}

# 类别配置
names:
{chr(10).join([f"  {i}: {name}" for i, name in names.items()])}
"""
    yaml_path = 'miner_pose_data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    return yaml_path

def fine_tune_yolo_pose(data_yaml_path, epochs=50, imgsz=640):
    """
    启动 YOLOv8-Pose 微调
    """
    # 加载预训练模型 (yolov8n-pose 是平衡速度与精度的最佳选择)
    model = YOLO('yolov8n-pose.pt')
    
    # 开始训练
   
    model.train(
        data=data_yaml_path, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=16,           
        device=0,           
        project='miner_behavior',
        name='yolov8_pose_finetuned',
        augment=True,       # 必须开启：针对井下低光照环境进行增强
        mosaic=0.5,         # 适当开启 mosaic 增强
        mixup=0.1,          # 开启 mixup 增强
        save=True,
        cache=False         # 设为 False，避免因为旧缓存导致找不到标签
    )

if __name__ == "__main__":
    # ================================================================
    #
    # 1. 确保 labels 文件夹已移动到 DATA_ROOT 目录下
    # 2. 手动删除 DATA_ROOT 下的 train2017.cache 和 val2017.cache
    # ================================================================
    
    # 1. 配置路径
    DATA_ROOT = r"D:\biyelunwen\Datasets\miner_finetune_clean"
    TRAIN_DIR = "images/train"
    VAL_DIR = "images/val" # 现在有真正的验证集了！

    
    # 2. 生成 YAML
    print(">>> 正在生成 miner_pose_data.yaml...")
    yaml_path = create_data_yaml(DATA_ROOT, TRAIN_DIR, VAL_DIR)
    
    # 3. 启动训练
    print(">>> 启动微调训练。请观察控制台，确保 'images' 数量不是 0...")
    fine_tune_yolo_pose(yaml_path, epochs=50)
