# 🔥 Robust Fire Detection using Mixture of YOLOv8 Experts

This project presents a modular fire detection system that leverages a **Mixture of Experts (MoE)** architecture built on top of **YOLOv8** models. Each expert is trained on a distinct fire scenario—**indoor**, **outdoor**, **far-field**, and **satellite imagery**. The system uses a lightweight **gating CNN** with **self-attention** to dynamically assign soft weights to each expert and applies **Weighted Box Fusion (WBF)** for final prediction refinement.

## 📌 Project Highlights

- 🧠 **Mixture of YOLOv8 Experts** trained on scenario-specific datasets.
- ⚖️ **Self-Attention-based Gating Network** to adaptively weight expert predictions.
- 🧩 **Weighted Box Fusion (WBF)** replaces Non-Maximum Suppression for improved aggregation.
- 📊 Achieved significant performance gains over a single YOLO baseline.
- 🧪 Extensive ablation: baseline vs. MoE, attention, TTA, confusion matrix comparisons.
- 🌐 **Streamlit App**: Upload an image and see fire detection results in real time.

---



# Mixture of Experts Fire Detection (moe-fire-detection)

A Deep Learning-based Fire Detection Web App built with **Streamlit**, featuring a **Mixture of Experts** architecture for improved detection across various real-world scenarios.

**Live App**: [moe-fire-detection Streamlit](https://moe-fire-detection-mgxc8wzgchajjkobm6go5q.streamlit.app/)

---

## Project Overview

This project utilizes:
- **4 Pre-trained YOLOv8 models** fine-tuned on distinct fire scenarios:
  - Indoor fires
  - Outdoor fires
  - Tower-mounted surveillance (far-field)
  - Satellite imagery
- A **CNN-based Gating Network** that dynamically assigns weights to experts based on the input context.
- A complete **Mixture of Experts (MoE)** framework to fuse predictions and enhance detection performance using confidence thresholding and Non-Maximum Suppression (NMS).

---

## Core Components

- `utils.py`: Contains helper functions for the forward pass through the MoE system, bounding box processing, confidence scoring, thresholding, and NMS.
- `app.py`: Main Streamlit app to upload an image and view the model’s fire detection results.
- `model/`: Folder with model loading utilities, inference logic, and pretrained weights.

---

## Running Locally

To host the app on your local machine:

```bash
python -m streamlit run app.py
