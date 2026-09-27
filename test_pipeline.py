import torch
from dataset import UrbanDashcamDataset, get_train_transform
from model import create_unet_model
from losses import CombinedLoss
from classes import NUM_CLASSES
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2

def test_pipeline():
    print("Running end-to-end pipeline test with dummy data...")
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    os.makedirs("dummy_test_img", exist_ok=True)
    os.makedirs("dummy_test_mask", exist_ok=True)
    
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_mask = np.zeros((720, 1280), dtype=np.uint8)
    cv2.imwrite("dummy_test_img/test.png", dummy_img)
    cv2.imwrite("dummy_test_mask/test.png", dummy_mask)

    transform = get_train_transform()
    dataset = UrbanDashcamDataset("dummy_test_img", "dummy_test_mask", transform=transform)
    loader = DataLoader(dataset, batch_size=1)

    model = create_unet_model(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = CombinedLoss(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for img, mask in loader:
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)
        
        preds = model(img)
        loss = loss_fn(preds, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Forward and backward pass successful! Loss: {loss.item():.4f}")
        break

    os.remove("dummy_test_img/test.png")
    os.remove("dummy_test_mask/test.png")
    os.rmdir("dummy_test_img")
    os.rmdir("dummy_test_mask")
    print("Pipeline test passed successfully!")

if __name__ == "__main__":
    test_pipeline()
