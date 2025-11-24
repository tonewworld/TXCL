"""
批量运行分割流程，并保存所有中间结果（增强图、掩膜、可视化图）。
使用方法：
    python scripts/run_segment.py --input data/raw --output data/results --enhance --win 41 --t 0.1
"""
import os
import argparse
import sys
import numpy as np
import cv2

# 确保项目根目录在 python path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from txcl_enhance.utils import read_image, save_image, ensure_uint8
from txcl_enhance.enhance import enhance_pipeline
from txcl_enhance.segment import segment_defects, compute_iou

def overlay_mask(img, mask, alpha=0.4):
    if img is None: return None
    vis = img.copy()
    # 将掩膜区域涂红
    red_layer = np.zeros_like(vis)
    red_layer[:, :, 0] = 0   # B
    red_layer[:, :, 1] = 0   # G
    red_layer[:, :, 2] = 255 # R (OpenCV is BGR)
    
    # 如果是 RGB 读取的 (txcl_enhance.utils 默认是 RGB)，则：
    if vis.shape[-1] == 3:
        # check if read_image returns RGB or BGR. Usually PIL->RGB.
        # Let's assume RGB for consistency with previous code.
        red_layer = np.zeros_like(vis)
        red_layer[:, :, 0] = 255 # R
    
    m = (mask > 0)
    vis[m] = (vis[m] * (1 - alpha) + red_layer[m] * alpha).astype('uint8')
    return vis

def process(args):
    input_dir = args.input
    output_dir = args.output
    gt_dir = args.gt
    
    # 1. 创建输出目录
    dirs = {
        'enhanced': os.path.join(output_dir, 'enhanced'),
        'masks': os.path.join(output_dir, 'masks'),
        'vis': os.path.join(output_dir, 'vis')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 2. 收集图片
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not files:
        print("No images found in input directory.")
        return

    print(f"Processing {len(files)} images...")
    print(f"Parameters: Win={args.win}, T={args.t}, Radius={args.open_radius}, MinSize={args.min_size}")

    ious = []

    for name in files:
        img_path = os.path.join(input_dir, name)
        img = read_image(img_path, as_gray=False)
        if img is None: continue
        
        # --- 步骤 1: 图像增强 ---
        proc = img
        if args.enhance:
            # 这里可以扩展传入 denoise 等参数，暂时使用默认
            proc = enhance_pipeline(img, method=args.method)
            # [新功能] 保存增强后的图片
            save_image(os.path.join(dirs['enhanced'], name), proc)
        
        # --- 步骤 2: 图像分割 ---
        # 明确传入所有调优参数
        mask = segment_defects(
            proc, 
            win_size=args.win, 
            t=args.t, 
            mode=args.mode, 
            min_size=args.min_size, 
            open_radius=args.open_radius, 
            keep_n=args.keep_n
        )
        mask_u8 = ensure_uint8(mask)
        
        # [保存] 掩膜
        # 保存为 png 以防压缩损失
        mask_name = os.path.splitext(name)[0] + ".png"
        save_image(os.path.join(dirs['masks'], mask_name), mask_u8)
        
        # --- 步骤 3: 可视化 ---
        try:
            vis = overlay_mask(img, mask_u8)
            save_image(os.path.join(dirs['vis'], name), vis)
        except Exception as e:
            print(f"Vis error on {name}: {e}")

        # --- 步骤 4: (可选) 计算 IoU ---
        if gt_dir:
            gt_path = os.path.join(gt_dir, mask_name)
            if os.path.exists(gt_path):
                gt_mask = read_image(gt_path, as_gray=True)
                if gt_mask is not None:
                    gt_mask = (gt_mask > 127).astype(np.uint8)
                    iou = compute_iou(mask_u8, gt_mask)
                    ious.append(iou)

    if ious:
        print(f"\nProcessed finished.")
        print(f"Mean Pixel IoU: {np.mean(ious):.4f}")
        print(f"Results saved to: {output_dir}")

def main():
    p = argparse.ArgumentParser(description='Batch segmentation with result saving')
    p.add_argument('--input', required=True, help='Input folder')
    p.add_argument('--output', required=True, help='Output folder')
    p.add_argument('--gt', required=False, default=None, help='Ground-truth mask folder')
    
    # 增强参数
    p.add_argument('--enhance', action='store_true', help='Apply enhancement')
    p.add_argument('--method', default='clahe', choices=['clahe', 'he', 'stretch'])
    
    # 分割参数 (调优对象)
    p.add_argument('--win', type=int, default=51, help='Bradley window size')
    p.add_argument('--t', type=float, default=0.15, help='Bradley threshold constant')
    p.add_argument('--mode', default='both', choices=['lower', 'higher', 'both'])
    p.add_argument('--min_size', type=int, default=50, help='Min component size')
    p.add_argument('--open_radius', type=int, default=3, help='Opening radius')
    p.add_argument('--keep_n', type=int, default=3, help='Keep largest N components')
    
    args = p.parse_args()
    process(args)

if __name__ == '__main__':
    main()