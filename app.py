# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from moe_utils import run_moe, SCENARIOS

st.set_page_config(page_title="🔥 Fire Detection - Mixture of Experts", layout="centered")

st.title("🔥 Fire Detection with Mixture of Experts")
st.write("Upload an image and let the Mixture of Experts detect fire based on scene type.")

# Sidebar sliders for thresholds
st.sidebar.header("⚙ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, step=0.01)
iou_threshold = st.sidebar.slider("IoU Threshold (To Remove OverLaps)", 0.1, 1.0, 0.5, step=0.05)

# Upload and process image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running Mixture of Experts..."):
        all_boxes, gate_weights = run_moe(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    
    # Draw detections
    img_array = np.array(image)
    img_bgr = img_array[..., ::-1].copy()

    for boxes in all_boxes:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Fire {conf:.2f}"
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    st.subheader("📸 Detection Output")
    st.image(img_bgr[..., ::-1], caption="Detections", use_container_width=True)