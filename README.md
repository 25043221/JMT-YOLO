# JMT-YOLO: Multi-task Domain Adaptation via Segmentation-Guided Consistency Learning

**[📄 Paper Title Placeholder - To be updated after acceptance]**

This repository will provide the official PyTorch implementation of **JMT-YOLO**, a joint multi-task learning framework for domain adaptive object detection. Code will be released upon acceptance of the paper.

## 🔍 Introduction

Domain adaptive object detection faces challenges due to the domain shift between labeled source data and unlabeled target data. To address this, we propose **JMT-YOLO**, a novel framework that incorporates:

- A **Mean Teacher architecture** for semi-supervised learning,
- **pseudo image generation** to bridge the domain gap at the image level,
- A **semantic segmentation auxiliary task** to enhance shared feature representations,
- A customized **DenseRFB module** for improved multi-scale feature extraction,
- A carefully designed **joint optimization loss** to guide both detection and segmentation tasks.

This framework demonstrates superior performance in cross-domain object detection benchmarks.

<p align="center">
  <img src="assets/architecture.png" alt="Framework Architecture" width="600"/>
</p>

## 📦 Features
- 🔁 **Multi-task learning**: Joint optimization of detection and segmentation.
- 🔍 **Domain adaptation**: Reduces domain shift via image-level and feature-level alignment.
- 🧠 **Teacher-student mechanism**: Enforces consistency on target predictions.
## 📁 Project Structure (To be released)
