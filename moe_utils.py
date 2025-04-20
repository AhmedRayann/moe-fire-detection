# moe_utils.py
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import os
from copy import deepcopy
from torchvision.ops import nms


class GatingCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(GatingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
device = torch.device( "cpu")
gating_model = GatingCNN().to(device)
gating_model.load_state_dict(torch.load("gating_cnn.pth", map_location=device))
gating_model.eval()


# Load YOLO expert models
expert_paths = [
    "models/indoor.pt",
    "models/outdoor.pt",
    "models/farfield.pt",
    "models/satellite.pt"
]
experts = [YOLO(p) for p in expert_paths]
SCENARIOS = ["Indoor", "Outdoor", "Far Field", "Satellite"]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

@torch.no_grad()
def get_gate_weights(img_tensor):
    device = next(gating_model.parameters()).device  # Get model device (cpu or cuda)
    img_tensor = img_tensor.to(device)
    logits = gating_model(img_tensor)
    return torch.softmax(logits, dim=1)[0]

def run_moe(pil_img, conf_threshold=0.3, iou_threshold=0.5):
    img_tensor = preprocess_image(pil_img)
    gate_weights = get_gate_weights(img_tensor)

    img_cv = np.array(pil_img.convert("RGB"))[..., ::-1]  # PIL to BGR
    h, w = img_cv.shape[:2]

    all_boxes_raw = []

    for i, expert in enumerate(experts):
        result = expert(img_cv, verbose=False)[0]
        weight = gate_weights[i]

        if result.boxes is not None and result.boxes.conf is not None:
            conf = result.boxes.conf * weight
            keep_mask = conf > conf_threshold

            if keep_mask.sum() > 0:
                xyxy = result.boxes.xyxy[keep_mask]
                conf = conf[keep_mask]
                cls = result.boxes.cls[keep_mask]

                combined = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1)
                all_boxes_raw.append(combined)

    if not all_boxes_raw:
        return [], gate_weights.numpy()

    all_boxes_combined = torch.cat(all_boxes_raw, dim=0)
    boxes = all_boxes_combined[:, :4]
    scores = all_boxes_combined[:, 4]
    classes = all_boxes_combined[:, 5]

    keep_indices = nms(boxes, scores, iou_threshold=iou_threshold)

    kept = all_boxes_combined[keep_indices]
    kept_boxes = Boxes(kept, orig_shape=(h, w))

    return [kept_boxes], gate_weights.numpy()