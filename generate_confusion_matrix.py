import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns

from dataset import UrbanDashcamDataset, get_val_transform
from model import create_unet_model
from utils import load_checkpoint
from classes import CLASS_NAMES, NUM_CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_AUTOCAST_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 4
BEST_CHECKPOINT = "unet_checkpoint_best.pth.tar"
VAL_IMG_DIR = "data/CamVid/val/"
VAL_MASK_DIR = "data/CamVid/val_labels/"

def main():
    print(f"Using device: {DEVICE}")

    val_transform = get_val_transform()
    val_dataset = UrbanDashcamDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform, mask_ext="_L.png")
    
    if len(val_dataset) == 0:
        print("No validation data found. Cannot compute confusion matrix.")
        return

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    model = create_unet_model(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Loading weights from {BEST_CHECKPOINT}...")
    load_checkpoint(BEST_CHECKPOINT, model, None)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Evaluating on Validation Set"):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            with torch.amp.autocast(device_type=_AUTOCAST_DEVICE, enabled=(DEVICE=='cuda')):
                predictions = model(data)
            
            # Get the predicted class for each pixel
            preds_cls = torch.argmax(predictions, dim=1).cpu().numpy().flatten()
            targets_flat = targets.cpu().numpy().flatten()

            all_preds.append(preds_cls)
            all_targets.append(targets_flat)

    # Combine all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    print("Computing confusion matrix (this might take a few seconds)...")
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(NUM_CLASSES))

    # Calculate Overall Pixel Accuracy
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Pixel Accuracy: {overall_accuracy * 100:.2f}%")

    # Calculate mean Intersection over Union (mIoU)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    # Avoid division by zero
    iou = np.divide(intersection.astype(float), union.astype(float), out=np.zeros_like(intersection, dtype=float), where=union!=0)
    # Calculate mean IoU only for classes that were present in the targets or predictions
    present_classes = union > 0
    miou = np.mean(iou[present_classes])
    print(f"Mean Intersection over Union (mIoU): {miou * 100:.2f}%\n")

    # Normalize the confusion matrix by row (i.e. by the true number of pixels in each class)
    # Adding a tiny epsilon to prevent division by zero for classes that aren't in the validation set
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    print("Plotting confusion matrix...")
    plt.figure(figsize=(22, 18))
    
    # We use a custom formatting to only show numbers if they are > 0.01 (1%)
    # to keep the grid clean. We'll also use the 'Blues' colormap.
    sns.heatmap(cm_normalized, annot=False, cmap="Blues", fmt=".2f",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Normalized Accuracy (Recall)'})
    
    plt.title("Normalized Confusion Matrix (CamVid Validation Set)", fontsize=24, pad=20)
    plt.ylabel('True Class', fontsize=18, labelpad=20)
    plt.xlabel('Predicted Class', fontsize=18, labelpad=20)
    plt.xticks(rotation=90, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    output_filename = "confusion_matrix.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Successfully saved high-resolution plot to '{output_filename}'!")

if __name__ == "__main__":
    main()
