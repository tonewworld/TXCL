"""Convert YOLO segmentation labels (polygons) to binary masks.

Usage:
python scripts/convert_labels_to_masks.py --images data/raw/train/images --labels data/raw/train/labels --output data/raw/train/masks_gt
"""
import os
import argparse
import cv2
import numpy as np

def parse_polygon(line, width, height):
    parts = line.strip().split()
    # class_id = int(parts[0]) # not used for binary mask
    coords = [float(x) for x in parts[1:]]
    points = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * width)
        y = int(coords[i+1] * height)
        points.append([x, y])
    return np.array(points, dtype=np.int32)

def convert_folder(img_dir, label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Converting labels for {len(img_files)} images...")
    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        
        # Try to find corresponding label file
        # Label file usually has same name but .txt extension
        # Sometimes image has .jpg and label has .txt
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        if not os.path.exists(label_path):
            # print(f"Label not found for {img_name}, skipping")
            continue
            
        # Read image to get size
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if not line.strip():
                continue
            pts = parse_polygon(line, w, h)
            cv2.fillPoly(mask, [pts], 255)
            
        # Save mask
        out_path = os.path.join(out_dir, base_name + ".png")
        cv2.imwrite(out_path, mask)

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO polygon labels to binary masks")
    parser.add_argument("--images", required=True, help="Folder containing images")
    parser.add_argument("--labels", required=True, help="Folder containing .txt labels")
    parser.add_argument("--output", required=True, help="Output folder for masks")
    args = parser.parse_args()
    
    convert_folder(args.images, args.labels, args.output)

if __name__ == "__main__":
    main()
