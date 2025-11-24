"""
功能：读取裁剪好的缺陷图像,提取 LBP 和 颜色特征,训练 SVM 分类器。
输出：分类性能指标,PCA 可视化图,保存的模型。
"""
import os
import cv2
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === 特征提取函数 ===
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
    # 统一尺寸以便提取特征 (可选,直方图特征其实不需要统一尺寸,但为了稳定建议resize)
    img = cv2.resize(img, (64, 64)) 
    f_lbp = extract_lbp_features(img)
    f_color = extract_color_features(img)
    # 拼接特征向量
    return np.hstack([f_lbp, f_color])

# === 主流程 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cropped data folder")
    parser.add_argument("--output", default="models", help="Folder to save model")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading data and extracting features...")
    data = []
    labels = []
    class_names = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    
    if not class_names:
        print("No classes found!")
        return

    print(f"Found classes: {class_names}")

    for label_id, class_name in enumerate(class_names):
        class_dir = os.path.join(args.data, class_name)
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.png', '.jpg')):
                img_path = os.path.join(class_dir, f)
                img = cv2.imread(img_path)
                if img is None: continue
                
                feat = get_features(img)
                data.append(feat)
                labels.append(class_name) # 使用名称作为标签

    X = np.array(data)
    y = np.array(labels)
    
    print(f"Total samples: {len(X)}, Feature dim: {X.shape[1]}")
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练 SVM
    print("Training SVM classifier...")
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    
    # 评估
    print("\nEvaluating...")
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # 保存模型
    model_path = os.path.join(args.output, "defect_classifier.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler, 'classes': class_names}, f)
    print(f"Model saved to {model_path}")
    
    # === 可视化 PCA ===
    print("Generating PCA visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y_train)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        indices = np.where(y_train == label)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], 
                   c=[color], label=label, alpha=0.7, edgecolors='k')
    
    plt.title("PCA of Defect Features (LBP + Color)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    pca_path = os.path.join(args.output, "pca_features.png")
    plt.savefig(pca_path)
    print(f"PCA plot saved to {pca_path}")

if __name__ == '__main__':
    main()