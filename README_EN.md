# YOLOv11 Instance Segmentation Inference with MNN  
## YOLOv11 Instance Segmentation Model Inference Based on MNN (Supporting x86 and ARM64 Platforms)

[English](#english) | [中文](README.md)

---

<a id="english"></a>
## English Version

### 📌 Overview  
This project demonstrates how to deploy the **YOLOv11 instance segmentation model** using the **MNN inference engine**, supporting both **x86** and **ARM64 (such as RDK x5)** hardware platforms. Through pre-compiled dependency libraries (Eigen, OpenCV, MNN) and utilizing YOLO's official export function to generate `.mnn` models, end-to-end inference and visualization are completed.

---

### 1️⃣ Environment Preparation

In the `3rdparty/` directory, use automated scripts to compile dependency libraries for target platforms respectively:

- **x86_64 (Linux)**
- **ARM64 (such as RDK x5 development board)**

### 2️⃣ Model Preparation

Export the YOLOv11 segmentation model to MNN format using the Ultralytics YOLO library:

```python
from ultralytics import YOLO

# Load YOLOv11 segmentation model
model = YOLO("./yolo11l-seg.pt")

# Export to MNN format (supporting FP32 / INT8 quantization)
model.export(format="mnn")              # Default FP32, generates yolo11l-seg.mnn
# model.export(format="mnn", int8=True) # Enable INT8 quantization

# Load the exported MNN model for inference
mnn_model = YOLO("./yolo11l-seg.mnn")
results = mnn_model("./bus.jpg")

# Visualize results
import matplotlib.pyplot as plt
plt.imshow(results[0].plot())
plt.axis('off')
plt.show()
```

### 3️⃣ Project Compilation
Run the compilation script in the project root directory:

```bash
./build.sh
```
Select the target platform according to the prompts:

- Enter 1: Compile x86_64 version, generating build_x86 folder
- Enter 2: Compile ARM64 (such as RDK x5) version, generating build_arm64 folder


After compilation is complete, the executable file test_YOLOv11Seg is generated


### 4️⃣ Execution and Results
Run the executable file test_YOLOv11Seg, which is the same on arm64 platforms, without needing to add other static or dynamic libraries, input the model path and image path:

```bash
./test_YOLOv11Seg yolo11l-seg_int8.mnn ./bus.jpg
```

Results:



![!\[bus\](./bus.jpg)](yolo_seg_results/result.jpg)
