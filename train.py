import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UrbanDashcamDataset, get_train_transform, get_val_transform
from model import create_unet_model
from losses import CombinedLoss
from utils import save_checkpoint, load_checkpoint, calculate_iou
from classes import NUM_CLASSES

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_AUTOCAST_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = True
BEST_CHECKPOINT = "unet_checkpoint_best.pth.tar"
TRAIN_IMG_DIR = "data/CamVid/train/"
TRAIN_MASK_DIR = "data/CamVid/train_labels/"
VAL_IMG_DIR = "data/CamVid/val/"
VAL_MASK_DIR = "data/CamVid/val_labels/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    total_iou = 0
    
    model.train()
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        with torch.amp.autocast(device_type=_AUTOCAST_DEVICE, enabled=(DEVICE=='cuda')):  # mixed precision only on CUDA
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds_cls = torch.argmax(predictions, dim=1)
        iou = calculate_iou(preds_cls, targets, num_classes=NUM_CLASSES)
        total_iou += iou
        loop.set_postfix(loss=loss.item(), mIoU=iou)

    return total_loss / len(loader), total_iou / len(loader)

def eval_fn(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validating"):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            with torch.amp.autocast(device_type=_AUTOCAST_DEVICE, enabled=(DEVICE=='cuda')):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            total_loss += loss.item()
            preds_cls = torch.argmax(predictions, dim=1)
            total_iou += calculate_iou(preds_cls, targets, num_classes=NUM_CLASSES)

    return total_loss / len(loader), total_iou / len(loader)

def main():
    print(f"Using device: {DEVICE}")

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = UrbanDashcamDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform, mask_ext="_L.png")

    if len(train_dataset) == 0:
        print("WARNING: No training data found. Please make sure images are in 'data/train_images/' and masks in 'data/train_masks/'.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True
    )

    val_dataset = UrbanDashcamDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform, mask_ext="_L.png")
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=False,
        )
    else:
        print(f"WARNING: No validation data found in '{VAL_IMG_DIR}'. Skipping validation "
              f"(mIoU will only be tracked on the training set, and no 'best' checkpoint will be saved).")

    model = create_unet_model(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = CombinedLoss(num_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None

    if LOAD_MODEL:
        if os.path.exists(BEST_CHECKPOINT):
            load_checkpoint(BEST_CHECKPOINT, model, optimizer)
        else:
            print(f"WARNING: LOAD_MODEL is True but '{BEST_CHECKPOINT}' was not found. Starting from scratch.")

    best_val_iou = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss, train_iou = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_iou:.4f}")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"unet_checkpoint_epoch_{epoch+1}.pth.tar")

        if val_loader is not None:
            val_loss, val_iou = eval_fn(val_loader, model, loss_fn)
            print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                save_checkpoint(checkpoint, filename=BEST_CHECKPOINT)
                print(f"=> New best model (val mIoU: {best_val_iou:.4f})")

if __name__ == "__main__":
    main()
