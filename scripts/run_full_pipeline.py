"""
全流程脚本：图像增强 -> 缺陷分割 -> 缺陷分类。
在输出图像上绘制不同颜色的框来表示不同类型的缺陷。
"""
import os
import sys
import cv2
import numpy as np
import argparse
import pickle
from skimage.feature import local_binary_pattern

# 导入你的项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from txcl_enhance.utils import read_image, save_image, ensure_uint8
from txcl_enhance.enhance import enhance_pipeline
from txcl_enhance.segment import segment_defects

# === 特征提取函数 (复制自 train_classifier.py 以避免模块导入错误) ===
def extract_lbp_features(img, radius=1, n_points=8):
    # 转换为灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # 计算 LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    # 计算直方图作为特征
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # 归一化
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_color_features(img, bins=(8, 8, 8)):
    # 转换为 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def get_features(img):
    # 统一尺寸以便提取特征
    img = cv2.resize(img, (64, 64)) 
    f_lbp = extract_lbp_features(img)
    f_color = extract_color_features(img)
    # 拼接特征向量
    return np.hstack([f_lbp, f_color])
# =================================================================

def load_model(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['classes']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", required=True, help="Path to .pkl model file")
    # 分割参数 (使用你调优后的)
    parser.add_argument("--win", type=int, default=41)
    parser.add_argument("--t", type=float, default=0.10)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # 加载分类器
    print(f"Loading classifier from {args.model}...")
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Please run train_classifier.py first.")
        return
        
    clf, scaler, classes = load_model(args.model)
    print(f"Classes: {classes}")
    
    # 定义颜色 (BGR)
    # 这里根据实际类别名称动态分配颜色，防止越界
    colors_palette = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
    ]
    
    colors = {}
    for i, cls_name in enumerate(classes):
        colors[cls_name] = colors_palette[i % len(colors_palette)]
    
    default_color = (128, 128, 128) # Gray

    files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    print(f"Processing {len(files)} images...")

    for name in files:
        path = os.path.join(args.input, name)
        img = read_image(path, as_gray=False)
        if img is None: continue
        
        # 1. 增强
        enhanced = enhance_pipeline(img, method='clahe')
        
        # 2. 分割
        mask = segment_defects(enhanced, win_size=args.win, t=args.t, min_size=20, open_radius=1)
        
        # 3. 提取连通域并分类
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vis = img.copy() # 在原图上画图
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 稍微扩大一点裁剪范围，包含一点背景有助于纹理识别
            pad = 5
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad); y2 = min(img.shape[0], y + h + pad)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            if w < 10 or h < 10: continue # 忽略太小的

            # 特征提取
            feat = get_features(crop)
            feat_scaled = scaler.transform([feat])
            
            # 预测
            pred_cls = clf.predict(feat_scaled)[0]
            
            # 绘制
            color = colors.get(pred_cls, default_color)
            label_text = f"{pred_cls}"
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        save_image(os.path.join(args.output, name), vis)

    print(f"Done! Results saved to {args.output}")

if __name__ == '__main__':
    main()