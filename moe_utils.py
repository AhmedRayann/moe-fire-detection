# moe_utils.py
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ensemble_boxes import weighted_boxes_fusion
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import os
from copy import deepcopy
from torchvision.ops import nms


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim // 8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling

    def forward(self, x):
        batch_size, C, width, height = x.size()

        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key(x).view(batch_size, -1, width * height)  # B x C' x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value(x).view(batch_size, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x  # Residual connection
        return out

class GatingCNNWithAttention(nn.Module):
    def __init__(self, num_classes=4):
        super(GatingCNNWithAttention, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 224, 224]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 32, 112, 112]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 112, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 64, 56, 56]

            SelfAttention(64),  # Add Attention here!

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 56, 56]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling [B, 128, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)  # Final output for 4 scenarios
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits  # raw logits (we apply softmax separately when needed)

device = torch.device( "cpu")
gating_model = GatingCNNWithAttention().to(device)
gating_model.load_state_dict(torch.load("improved_gating_cnn.pth", map_location=device))
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
    # Step 1: Preprocess PIL image for gating model
    img_tensor = preprocess_image(pil_img)  # Assume preprocess_image works on PIL.Image

    # Step 2: Get expert weights from attention-based gating network
    gate_weights = get_gate_weights(img_tensor)  # Softmax weights (shape: [4])
    
    # Step 3: Convert PIL to OpenCV format (for YOLO)
    img_cv = np.array(pil_img.convert("RGB"))[..., ::-1]
    h, w = img_cv.shape[:2]

    all_boxes_raw = []

    # Step 4: Run each expert model and collect weighted predictions
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

    # Step 5: Handle case where no expert produced valid boxes
    if not all_boxes_raw:
        return [], gate_weights.numpy()

    # Step 6: Concatenate predictions from all experts
    all_boxes_combined = torch.cat(all_boxes_raw, dim=0)
    boxes = all_boxes_combined[:, :4]
    scores = all_boxes_combined[:, 4]
    classes = all_boxes_combined[:, 5]

    # Step 7: Normalize coordinates for WBF
    norm_boxes = boxes.clone()
    norm_boxes[:, [0, 2]] /= w
    norm_boxes[:, [1, 3]] /= h

    boxes_list = [norm_boxes.cpu().numpy().tolist()]
    scores_list = [scores.cpu().numpy().tolist()]
    labels_list = [classes.cpu().numpy().tolist()]

    # Step 8: Apply Weighted Box Fusion
    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_threshold,
        skip_box_thr=conf_threshold
    )

    # Step 9: Rescale boxes back to original size
    boxes_wbf = np.array(boxes_wbf)
    boxes_wbf[:, [0, 2]] *= w
    boxes_wbf[:, [1, 3]] *= h

    final_tensor = torch.tensor(np.hstack([
        boxes_wbf,
        np.array(scores_wbf).reshape(-1, 1),
        np.array(labels_wbf).reshape(-1, 1)
    ]), dtype=torch.float32)

    kept_boxes = Boxes(final_tensor, orig_shape=(h, w))

    # Step 10: Return predictions and gating weights
    return [kept_boxes], gate_weights.numpy()