import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

def visualize_pose_on_image(img_path, model_path):
    """
    可视化 YOLOv8-Pose 的检测结果
    """
    if not os.path.exists(img_path):
        print(f"错误：找不到图片文件 {img_path}")
        return

    # 加载模型
    model = YOLO(model_path)
    
    # 执行推理
    results = model(img_path)[0]
    
    # 使用 YOLO 自带的绘图功能
    annotated_frame = results.plot()
    
    # 转换为 RGB 并在 Matplotlib 中显示
    img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('YOLOv8-Pose Detection Result (Miner Behavior)')
    
    # 保存结果
    output_name = 'pose_detection_sample5.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f">>> 结果已保存为 {output_name}")
    plt.show()

if __name__ == "__main__":
    # 1. 确保这里的路径前带有 r，以防止转义字符错误
    # 2. 确保 best.pt 就在当前目录下
    IMG_PATH = r'D:\biyelunwen\Datasets\miner_behavior _data2023_coco\train2017\000000002818.jpg'
    MODEL_PATH = r'D:\biyelunwen\Datasets\project_root\runs\pose\miner_behavior\yolov8_pose_finetuned4\weights\best.pt'
    
    # 执行可视化
    visualize_pose_on_image(IMG_PATH, MODEL_PATH)
