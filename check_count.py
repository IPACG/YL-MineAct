import json

def check_json_stats(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    ann_count = len(data['annotations'])
    img_count = len(data['images'])
    
    # 统计每个类别的数量
    cat_stats = {}
    cat_map = {cat['id']: cat['name'] for cat in data['categories']}
    for ann in data['annotations']:
        cat_name = cat_map.get(ann['category_id'], "Unknown")
        cat_stats[cat_name] = cat_stats.get(cat_name, 0) + 1
        
    print(f"文件: {json_path}")
    print(f"总图片数: {img_count}")
    print(f"成功提取关键点的目标数: {ann_count}")
    print("-" * 30)
    print("各类别分布:")
    for cat_name, count in sorted(cat_stats.items(), key=lambda x: x[1], reverse=True):
        print(f" - {cat_name}: {count}")

# 运行检查
check_json_stats('pseudo_train.json')
