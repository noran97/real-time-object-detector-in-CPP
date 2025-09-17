# YOLO Inference in C++

This project provides a simple **C++ implementation of YOLOv7 inference** using [ONNX Runtime](https://onnxruntime.ai/).  
It allows you to run **object detection** on images and videos efficiently, with optional **CUDA acceleration** for faster performance.

---

## 🚀 Features

- 📦 Load YOLOv7 ONNX models with **ONNX Runtime**  
- 🖼️ Run inference on **images and videos**  
- 📊 Display results with **bounding boxes, class names, and confidence scores**  
- ⚡ Support for both **CPU** and **CUDA (GPU)** execution  
- 🧩 **Modular structure**: code separated into `Detector.h`, `Detector.cpp`, and `main.cpp`  

---
## 🔧 Requirements

- C++17 or later  
- CMake (>= 3.12)  
- OpenCV (>= 4.0)  
- ONNX Runtime (with CUDA support if using GPU)  


