"""
功能：根据 YOLO 标注文件，将原图中的缺陷区域裁剪出来，按类别保存。
用于制作分类器的训练数据。
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to raw images")
    parser.add_argument("--labels", required=True, help="Path to YOLO .txt labels")
    parser.add_argument("--output", required=True, help="Output folder for cropped patches")
    args = parser.parse_args()

    # 定义类别名称映射（根据你的数据集实际情况修改）
    # 假设 Roboflow 导出顺序：0: Scratch, 1: Dent, 2: Spot ...
    # 如果不确定，先用 id_0, id_1 代替
    class_names = {
        0: "defect_0",
        1: "defect_1",
        2: "defect_2",
        3: "defect_3"
    }

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in valid_exts]
    
    count = 0
    print(f"Processing {len(files)} images to extract crops...")

    for f in tqdm(files):
        img_path = os.path.join(args.images, f)
        label_name = os.path.splitext(f)[0] + ".txt"
        label_path = os.path.join(args.labels, label_name)
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        with open(label_path, 'r') as lf:
            lines = lf.readlines()
            
        for idx, line in enumerate(lines):
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            
            # YOLO format: class x_center y_center w h
            # 我们需要转换成 x1, y1, x2, y2
            if len(parts) >= 5:
                cx, cy, bw, bh = parts[1:5]
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # 边界保护
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                
                if x2 - x1 < 5 or y2 - y1 < 5: continue # 太小的忽略

                crop = img[y1:y2, x1:x2]
                
                # 保存
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                save_dir = os.path.join(args.output, cls_name)
                os.makedirs(save_dir, exist_ok=True)
                
                save_name = f"{os.path.splitext(f)[0]}_crop_{idx}.png"
                cv2.imwrite(os.path.join(save_dir, save_name), crop)
                count += 1

    print(f"Done! Extracted {count} defect patches to {args.output}")

if __name__ == '__main__':
    main()