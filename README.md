# YOLOv11 Instance Segmentation Inference with MNN  
## 基于 MNN 的 YOLOv11 实例分割模型推理（支持 x86 与 ARM64 平台）

[English](README_EN.md) | [中文](#chinese)

---

<a id="chinese"></a>
## 中文版

### 📌 概述  
本项目展示了如何使用 **MNN 推理引擎**部署 **YOLOv11 实例分割模型**，支持 **x86** 和 **ARM64（如 RDX x5）** 两种硬件平台。通过预编译依赖库（Eigen、OpenCV、MNN），并利用 YOLO 官方导出功能生成 `.mnn` 模型，完成端到端推理与可视化。

---

### 1️⃣ 环境准备

在 `3rdparty/` 目录下，使用自动化脚本分别为目标平台编译依赖库：

- **x86_64（Linux）**
- **ARM64（如 RDX x5 开发板）**

### 2️⃣ 模型准备

使用 Ultralytics YOLO 库导出 YOLOv11 分割模型为 MNN 格式：

```python
from ultralytics import YOLO

# 加载 YOLOv11 分割模型
model = YOLO("./yolo11l-seg.pt")

# 导出为 MNN 格式（支持 FP32 / INT8 量化）
model.export(format="mnn")              # 默认 FP32，生成 yolo11l-seg.mnn
# model.export(format="mnn", int8=True) # 启用 INT8 量化

# 加载导出的 MNN 模型进行推理
mnn_model = YOLO("./yolo11l-seg.mnn")
results = mnn_model("./bus.jpg")

# 可视化结果
import matplotlib.pyplot as plt
plt.imshow(results[0].plot())
plt.axis('off')
plt.show()
```

### 3️⃣ 编译项目
在项目根目录运行编译脚本：

```bash
./build.sh
```
根据提示选择目标平台：

- 输入 1：编译 x86_64 版本，生成build_x86文件夹
- 输入 2：编译 ARM64（如 RDX x5）版本，生成build_arm64文件夹


编译完成后，生成可执行文件 test_YOLOv11Seg


### 4️⃣ 运行与结果
运行可执行文件 test_YOLOv11Seg，在arm64平台上也相同，不需要添加其他静态或动态链接库，输入模型路径和图片路径：

```bash
./test_YOLOv11Seg yolo11l-seg_int8.mnn ./bus.jpg
```

结果：


![!\[bus\](./bus.jpg)](yolo_seg_results/result.jpg)