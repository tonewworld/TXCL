"""
评估脚本：将分割掩膜转换为边界框 (Bounding Box)，计算 Box IoU 和 召回率 (Recall)。
适用于：标注是粗糙矩形框，而预测是精细像素的情况。
"""
import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# 确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from txcl_enhance.utils import read_image
from txcl_enhance.segment import segment_defects
from txcl_enhance.enhance import enhance_pipeline

def get_bounding_box(mask):
    """获取二值掩膜中所有非零像素的最小外接矩形 [x, y, w, h]"""
    if np.count_nonzero(mask) == 0:
        return None
    
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # 简单起见，我们将所有轮廓合并为一个大框来计算（也可以逐个匹配，但这里先看整体）
    # 获取所有轮廓点的集合
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    return (x, y, w, h)

def compute_box_iou(boxA, boxB):
    """计算两个矩形框的 IoU"""
    if boxA is None or boxB is None:
        return 0.0
        
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def main():
    parser = argparse.ArgumentParser(description="Evaluate using Bounding Box IoU")
    parser.add_argument("--input", required=True, help="Input folder with raw images")
    parser.add_argument("--gt", required=True, help="Ground truth folder with mask images (rectangles)")
    parser.add_argument("--enhance", action="store_true", help="Apply enhancement")
    # 使用之前调优得到的参数默认值
    parser.add_argument("--win", type=int, default=41)
    parser.add_argument("--t", type=float, default=0.10)
    parser.add_argument("--open_radius", type=int, default=1)
    args = parser.parse_args()

    input_dir = args.input
    gt_dir = args.gt
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    box_ious = []
    detected_count = 0
    total_defects = 0

    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Evaluating {len(files)} images using Box IoU...")

    for f in tqdm(files):
        img_path = os.path.join(input_dir, f)
        basename = os.path.splitext(f)[0]
        
        # 寻找对应的 GT 掩膜
        gt_path = os.path.join(gt_dir, basename + ".png")
        if not os.path.exists(gt_path):
            continue

        # 1. 读取数据
        img = read_image(img_path, as_gray=False)
        gt_mask = read_image(gt_path, as_gray=True)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # 如果 GT 是空的（无缺陷图），跳过或单独处理
        if np.count_nonzero(gt_mask) == 0:
            continue

        total_defects += 1

        # 2. 运行预测管道
        if args.enhance:
            img = enhance_pipeline(img, method="clahe")
        
        pred_mask = segment_defects(
            img, 
            win_size=args.win, 
            t=args.t, 
            open_radius=args.open_radius,
            min_size=20
        )

        # 3. 转换为 Box 并计算 IoU
        gt_box = get_bounding_box(gt_mask)
        pred_box = get_bounding_box(pred_mask)

        iou = compute_box_iou(pred_box, gt_box)
        box_ious.append(iou)

        # 4. 计算是否“命中” (Hit / Recall)
        # 如果 IoU > 0，或者预测框中心在真值框内，都算检测到了
        is_hit = False
        if iou > 0.1: # 只要框重叠超过 10% 就算检测成功
            is_hit = True
        elif pred_box and gt_box:
            # 检查包含关系：预测框是否在真值框内部
            px, py, pw, ph = pred_box
            gx, gy, gw, gh = gt_box
            # 计算交集面积
            xA = max(px, gx)
            yA = max(py, gy)
            xB = min(px+pw, gx+gw)
            yB = min(py+ph, gy+gh)
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter > 0: # 只要有任何像素重叠
                is_hit = True
        
        if is_hit:
            detected_count += 1

    if box_ious:
        mean_iou = np.mean(box_ious)
        recall = detected_count / total_defects if total_defects > 0 else 0
        
        print("\n" + "="*40)
        print(f"Evaluation Results (Box-Level):")
        print(f"  Mean Box IoU: {mean_iou:.4f}")
        print(f"  Recall (Hit Rate): {recall:.2%} ({detected_count}/{total_defects})")
        print("="*40)
        print("Note: Mean Box IoU > 0.5 is usually considered good.")
    else:
        print("No valid comparisons made.")

if __name__ == '__main__':
    main()