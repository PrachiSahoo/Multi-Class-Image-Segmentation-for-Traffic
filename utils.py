import os
import torch
import numpy as np
import cv2

from classes import CLASS_COLORS, NUM_CLASSES

def save_checkpoint(state, filename="unet_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # load to CPU first regardless of target device
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

def calculate_iou(preds, labels, num_classes, ignore_index=255):
    valid_mask = labels != ignore_index

    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls) & valid_mask
        target_inds = (labels == cls) & valid_mask

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        ious.append(float('nan') if union == 0 else float(intersection) / float(union))  # nan = class absent from both pred and target this batch

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    if not valid_ious:
        return 0.0
    return np.mean(valid_ious)

def apply_color_map(mask, num_classes=NUM_CLASSES):
    if num_classes == len(CLASS_COLORS):
        colors = CLASS_COLORS
    else:
        colors = np.random.RandomState(42).randint(0, 255, size=(num_classes, 3), dtype=np.uint8)  # seeded so an unfamiliar class count still gets a stable palette

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_mask[mask == cls] = colors[cls]

    return color_mask
