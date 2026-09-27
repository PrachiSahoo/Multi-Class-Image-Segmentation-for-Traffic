import torch
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

from model import create_unet_model
from utils import load_checkpoint, apply_color_map
from dataset import get_val_transform
from classes import NUM_CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_AUTOCAST_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(model_path):
    model = create_unet_model(num_classes=NUM_CLASSES).to(DEVICE)

    if os.path.exists(model_path):
        load_checkpoint(model_path, model)
    else:
        print(f"WARNING: Checkpoint {model_path} not found. Using untrained weights.")

    model.eval()
    return model


def segment_frame(model, transform, image_rgb, alpha=0.5):  # shared by predict_image and predict_video so per-frame behavior stays identical
    original_size = (image_rgb.shape[1], image_rgb.shape[0])

    augmented = transform(image=image_rgb)
    image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type=_AUTOCAST_DEVICE, enabled=(DEVICE == "cuda")):
            preds = model(image_tensor)
            preds = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()

    color_mask = apply_color_map(preds, num_classes=NUM_CLASSES)
    color_mask = cv2.resize(color_mask, original_size, interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)
    return overlay


def predict_image(image_path, model_path, output_path):
    print(f"Setting up model for inference on {DEVICE}...")
    model = load_model(model_path)
    transform = get_val_transform()

    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"WARNING: Image {image_path} not found or not provided. Generating dummy noise image.")
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    print("Running forward pass...")
    overlay = segment_frame(model, transform, image)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    if output_path:
        cv2.imwrite(output_path, overlay_bgr)
        print(f"Saved prediction overlay to {output_path}")


def predict_video(video_path, model_path, output_path):
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Setting up model for inference on {DEVICE}...")
    model = load_model(model_path)
    transform = get_val_transform()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    root, ext = os.path.splitext(output_path)
    if ext.lower() not in VIDEO_EXTENSIONS:
        output_path = root + ".mp4"  # e.g. the CLI's image-default "output.jpg" gets redirected to a video container

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open video writer for: {output_path}")

    print(f"Processing video ({frame_count if frame_count > 0 else '?'} frames at {fps:.1f} fps)...")
    try:
        with tqdm(total=frame_count if frame_count > 0 else None, desc="Segmenting frames") as pbar:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                overlay_rgb = segment_frame(model, transform, frame_rgb)
                writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                pbar.update(1)
    finally:
        cap.release()
        writer.release()

    print(f"Saved segmented video to {output_path}")


def predict(input_path, model_path, output_path):
    ext = os.path.splitext(input_path)[1].lower() if input_path else ""
    if ext in VIDEO_EXTENSIONS:
        predict_video(input_path, model_path, output_path)
    else:
        predict_image(input_path, model_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net segmentation on a dashcam image or video")
    parser.add_argument("--image", type=str, help="Path to input image or video", default="")
    parser.add_argument("--model", type=str, help="Path to trained model checkpoint", default="unet_checkpoint_best.pth.tar")
    parser.add_argument("--output", type=str, help="Path to save the output overlay (image or video)", default="output.jpg")

    args = parser.parse_args()
    predict(args.image, args.model, args.output)
