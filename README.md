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
