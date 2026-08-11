BeadSketch Studio 可选 AI 模型包
================================

semantic_encoder.onnx 是官方预训练 MobileNetV3-Small 的视觉特征编码器。
程序即使没有此文件，也会使用轻量视觉算法完成智能调参。

设备优先级：
1. 若已运行 install_gpu_runtime.ps1，使用独立 ONNX Runtime 环境，优先 NVIDIA CUDA。
2. CUDA 不可用时，ONNX Runtime 自动回退 CPU。
3. 未安装独立运行时时，主程序使用 OpenCV DNN CPU 读取同一模型。

模型只在本地提取图片特征，不上传图片，不需要联网。
