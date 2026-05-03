import json
from collections import Counter

def analyze_distribution(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    # Get all category IDs in annotations
    ann_cat_ids = [ann['category_id'] for ann in data.get('annotations', [])]
    
    distribution = Counter(ann_cat_ids)
    
    print(f"\nClass Distribution for {path}:")
    print("-" * 30)
    for cat_id, count in sorted(distribution.items()):
        name = cat_id_to_name.get(cat_id, "Unknown")
        percentage = (count / len(ann_cat_ids)) * 100
        print(f"{name:15} (ID: {cat_id}): {count:6} ({percentage:5.2f}%)")

if __name__ == "__main__":
    analyze_distribution('/home/ubuntu/upload/instances_train2017.json')
