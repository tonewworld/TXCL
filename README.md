# 汽车漆面缺陷图像增强 (txcl-enhance)

项目说明：用于对汽车漆面缺陷图像进行基于传统图像处理的增强，包含全局直方图均衡化（HE）、CLAHE（自适应限幅直方图均衡化）、基于百分位的对比度拉伸、去噪与轻度锐化。旨在提升缺陷与背景对比、增强边缘细节，为后续分割/识别提供更高质量输入。

目录结构（简要）：

- `data/`：示例图像输入与输出目录
  - `raw/`：放置原始图像
  - `processed/`：程序输出的增强图像
- `txcl_enhance/`：核心库
  - `io.py`：图像读写与批处理
  - `enhance.py`：增强算法实现
  - `evaluate.py`：图像质量评价（BRISQUE/NIQE 为可选项）
  - `utils.py`：工具函数
- `scripts/`：运行脚本与 CLI
- `tests/`：简单单元测试
- `requirements.txt`

快速开始（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 批量增强示例
python scripts\run_enhance.py --input data\raw --output data\processed --method clahe
```

注意：BRISQUE/NIQE 的 Python 实现依赖项不一，项目提供简单的替代评价方法（对比度、边缘强度等）。如需使用 BRISQUE/NIQE，请按 `evaluate.py` 中注释安装相应包。

作者：自动生成模板
