# Automotive Paint Defect Detection System (TXCL)
# 汽车漆面缺陷检测与分类系统

本项目是一个基于传统数字图像处理和机器学习的汽车漆面缺陷自动检测系统。针对工业场景下光照不均、缺陷微小等痛点，提出了一套轻量级、高召回率的解决方案。

---

##  目录

- [项目背景](#-项目背景)
- [核心功能](#-核心功能)
- [算法原理](#-算法原理)
  - [1. 图像增强 (CLAHE)](#1-图像增强-clahe)
  - [2. 缺陷分割 (Bradley + Morphology)](#2-缺陷分割-bradley--morphology)
  - [3. 缺陷分类 (LBP + SVM)](#3-缺陷分类-lbp--svm)
- [快速开始](#-快速开始)
  - [环境安装](#环境安装)
  - [运行流程](#运行流程)
- [目录结构](#-目录结构)
- [实验结果](#-实验结果)

---

##  项目背景

在汽车制造与维修行业，漆面质量直接影响车辆的美观与耐用性。传统的人工目视检测效率低、主观性强且容易漏检。深度学习方法虽然精度高，但对算力要求苛刻。本项目旨在探索一种**无需GPU、可部署于低端工控机**的自动化检测方案。

##  核心功能

1.  **自适应图像增强**：有效解决高反光漆面上的光照不均问题，突显微小划痕。
2.  **高精度分割**：利用积分图技术实现快速自适应阈值分割，精准定位缺陷区域。
3.  **缺陷分类**：自动识别划痕（Scratches）、凹坑（Dents）、污渍（Spots）等类型。
4.  **全流程工具链**：包含数据预处理、参数寻优、模型训练及可视化评估的全套脚本。

---

##  算法原理

本系统采用串行流水线设计：**增强 -> 分割 -> 分类**。

### 1. 图像增强 (CLAHE)
由于车身曲面反光，图像往往存在局部过亮或过暗。
- **原理**：对比度受限自适应直方图均衡化 (Contrast Limited Adaptive Histogram Equalization)。
- **操作**：将图像划分为若干小块（Grid），对每一块独立进行直方图均衡化。同时引入“剪切限制（Clip Limit）”，防止背景噪声被过度放大。
- **效果**：在保留图像整体亮度的同时，显著增强了局部纹理（缺陷）的对比度。

### 2. 缺陷分割 (Bradley + Morphology)
如何从背景中提取出缺陷？
- **Bradley 自适应阈值**：这是一种局部阈值算法。它计算每个像素周围 $s \times s$ 窗口内的平均亮度。如果当前像素比平均亮度低 $t\%$，则判定为缺陷。利用**积分图 (Integral Image)** 技术，使计算复杂度降为 $O(1)$，速度极快。
- **形态学后处理**：
  - **开运算 (Opening)**：先腐蚀后膨胀，去除孤立的噪点。
  - **面积过滤**：剔除面积过小的连通域，保留真实缺陷。

### 3. 缺陷分类 (LBP + SVM)
如何区分是划痕还是凹坑？
- **特征提取**：
  - **LBP (Local Binary Pattern)**：提取局部纹理特征，对光照变化具有不变性，适合描述划痕的粗糙度。
  - **颜色直方图 (Color Histogram)**：提取HSV空间的颜色分布，用于区分不同颜色的污渍。
- **分类器**：使用支持向量机 (**SVM**)，将提取的特征向量映射到高维空间进行分类。

---

##  快速开始

### 环境安装

建议使用 Python 3.8+ 环境：

```bash
# 1. 克隆仓库
git clone https://github.com/tonewworld/TXCL.git
cd TXCL

# 2. 创建虚拟环境 (推荐)
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

*(如果缺少 requirements.txt，请手动安装: `pip install numpy opencv-python scikit-image scikit-learn matplotlib tqdm`)*

### 运行流程

#### 步骤 1：制作掩膜 (如果只有 YOLO txt 标签)
将 YOLO 格式的标注转换为用于分割评估的 PNG 掩膜。
```bash
python scripts/yolo_to_mask.py --images data/raw/train/images --labels data/raw/train/labels --output data/raw/train/masks_gt
```

#### 步骤 2：参数寻优 (可选)
在训练集上搜索最佳分割参数（窗口大小、阈值等）。
```bash
python scripts/tune_segmentation.py --input data/raw/train/images --gt data/raw/train/masks_gt --enhance
# 记录下输出的最佳参数，例如: --win 41 --t 0.10
```

#### 步骤 3：运行分割并保存结果
使用最佳参数处理图片，生成分割掩膜和可视化图。
```bash
python scripts/run_segment.py --input data/raw/test/images --output data/results_segment --enhance --win 41 --t 0.10
```

#### 步骤 4：训练分类器
提取缺陷小图并训练 SVM 模型。
```bash
# 1. 从原图中裁剪出缺陷样本
python scripts/prepare_crops.py --images data/raw/train/images --labels data/raw/train/labels --output data/crops

# 2. 训练模型 (生成 .pkl 文件)
python scripts/train_classifier.py --data data/crops --output models
```

#### 步骤 5：全流程演示
在一张新图片上同时完成检测与分类，并画框。
```bash
python scripts/run_full_pipeline.py --input data/raw/test/images --output data/final_result --model models/defect_classifier.pkl --win 41 --t 0.10
```

---

##  目录结构

```text
TXCL/
├── data/                   # 数据存放目录
│   ├── raw/                # 原始数据 (images/labels)
│   ├── crops/              # 裁剪出的分类训练样本
│   └── results/            # 运行结果输出
├── models/                 # 存放训练好的 SVM 模型 (.pkl)
├── scripts/                # 功能脚本
│   ├── tune_segmentation.py    # 参数寻优 (Grid Search)
│   ├── run_segment.py          # 批量分割
│   ├── train_classifier.py     # 特征提取与模型训练
│   ├── run_full_pipeline.py    # 端到端检测演示
│   ├── evaluate_detection.py   # Box IoU/Recall 评估
│   └── yolo_to_mask.py         # 格式转换工具
├── txcl_enhance/           # 核心算法库
│   ├── enhance.py          # CLAHE 等增强算法
│   ├── segment.py          # Bradley 分割算法
│   └── utils.py            # IO 工具
└── README.md
```

---

##  实验结果

在测试集上的评估表现如下：

| 指标 | 数值 | 说明 |
| :--- | :--- | :--- |
| **Recall (召回率)** | **97.5%** | 极高的检出率，几乎不漏检 |
| **Mean Box IoU** | 0.34 | 定位精准，主要受标注框大小影响 |
| **分类准确率** | ~89% | SVM 对常见缺陷分类效果良好 |


---

**Author:** @tonewworld
**Date:** 2025-11