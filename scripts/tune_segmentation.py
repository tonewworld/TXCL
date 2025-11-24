"""
脚本功能：遍历不同的分割参数组合，在验证集上寻找 IoU 最高的参数配置。
包含调试功能，可输出文件匹配失败的原因。
"""
import os
import sys
import argparse
import numpy as np
import itertools
from tqdm import tqdm

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from txcl_enhance.utils import read_image
from txcl_enhance.segment import segment_defects, compute_iou
from txcl_enhance.enhance import enhance_pipeline

def gather_image_pairs(input_dir, gt_dir, debug=True):
    """收集图像和对应的 Ground Truth 掩膜路径，支持相对路径匹配"""
    pairs = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    input_dir = os.path.abspath(input_dir)
    gt_dir = os.path.abspath(gt_dir)
    
    print(f"Searching for images in: {input_dir}")
    print(f"Searching for labels in: {gt_dir}")
    
    count_scanned = 0
    failures = [] # 记录前几个失败的例子用于调试

    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                count_scanned += 1
                img_path = os.path.join(root, f)
                
                # 计算相对于 input_dir 的相对路径 (例如: subdir/image.jpg)
                rel_path = os.path.relpath(root, input_dir)
                if rel_path == '.':
                    rel_path = ''
                
                base_name = os.path.splitext(f)[0]
                
                # 在 gt_dir 中寻找对应的子目录
                current_gt_dir = os.path.join(gt_dir, rel_path)
                
                # 候选列表：这里列出了所有脚本尝试寻找的文件名
                # 你可以根据你的实际文件名在这里添加规则
                candidates = [
                    os.path.join(current_gt_dir, f),                        # 同名同扩展名
                    os.path.join(current_gt_dir, base_name + ".png"),       # 同名.png
                    os.path.join(current_gt_dir, base_name + ".jpg"),       # 同名.jpg
                    os.path.join(current_gt_dir, base_name + "_mask.png"),  # 后缀 _mask
                    os.path.join(current_gt_dir, base_name + "_label.png"), # 后缀 _label
                    os.path.join(current_gt_dir, base_name + "_gt.png"),    # 后缀 _gt
                ]
                
                found = False
                for cand in candidates:
                    if os.path.exists(cand):
                        pairs.append((img_path, cand))
                        found = True
                        break
                
                if not found and len(failures) < 5:
                    failures.append((img_path, candidates))

    print(f"Scanned {count_scanned} images.")
    
    if not pairs:
        print("\n" + "!"*50)
        print("ERROR: No matching pairs found!")
        print("!"*50)
        if failures:
            print("\nExample failures (what we looked for but didn't find):")
            for img, cands in failures:
                print(f"\nImage: {os.path.basename(img)}")
                print("Tried looking for:")
                for c in cands:
                    print(f"  [X] {c}")
        print("\nSuggestion: Check your filenames or add a new rule to 'candidates' list in the script.")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Tune segmentation parameters for max IoU")
    parser.add_argument("--input", required=True, help="Input folder with raw images")
    parser.add_argument("--gt", required=True, help="Ground truth folder with mask images")
    parser.add_argument("--enhance", action="store_true", help="Apply enhancement before segmentation")
    parser.add_argument("--method", default="clahe", choices=["clahe", "he", "stretch"], help="Enhancement method")
    args = parser.parse_args()

    # 1. 准备数据
    pairs = gather_image_pairs(args.input, args.gt)
    if not pairs:
        return
        
    print(f"Found {len(pairs)} pairs. Loading data into memory...")

    # 预加载数据
    data_cache = []
    for img_path, gt_path in tqdm(pairs, desc="Loading"):
        img = read_image(img_path, as_gray=False)
        if img is None: 
            continue
            
        if args.enhance:
            img = enhance_pipeline(img, method=args.method)
        
        gt = read_image(gt_path, as_gray=True)
        if gt is None:
            continue
            
        # 确保 GT 是二值的
        gt = (gt > 127).astype(np.uint8)
        data_cache.append((img, gt))

    if not data_cache:
        print("Error: Failed to load valid images from the pairs.")
        return

    # 2. 定义参数搜索空间
    param_grid = {
        'win_size': [41, 51, 61, 81, 101],
        't': [0.10, 0.15, 0.20, 0.25],
        'open_radius': [1, 3, 5],
        'min_size': [20, 50]
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    
    print(f"\nStarting grid search over {len(combinations)} combinations...")
    
    best_iou = -1.0
    best_params = None
    
    # 3. 开始搜索
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        total_iou = 0.0
        count = 0
        
        for img, gt in data_cache:
            try:
                mask = segment_defects(
                    img, 
                    win_size=params['win_size'],
                    t=params['t'],
                    mode='both',
                    min_size=params['min_size'],
                    open_radius=params['open_radius'],
                    keep_n=3
                )
                iou = compute_iou(mask, gt)
                total_iou += iou
                count += 1
            except Exception:
                continue

        if count == 0: continue
        avg_iou = total_iou / count
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_params = params
            print(f"[New Best] IoU: {best_iou:.4f} | Params: {params}")
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(combinations)}...", end='\r')

    print("\n" + "="*50)
    print(f"Optimization Finished!")
    print(f"Best Average IoU: {best_iou:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*50)

if __name__ == '__main__':
    main()