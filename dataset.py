import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from classes import CLASS_COLORS

class UrbanDashcamDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_ext=".png"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_ext = mask_ext  # masks are matched to images by basename + this extension

        if os.path.exists(image_dir):
            self.images = sorted(f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
        else:
            self.images = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        mask_name = os.path.splitext(img_name)[0] + self.mask_ext
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(  # fail loudly instead of silently training on a blank image
                f"Could not read image '{img_path}'. Check that the file exists and is a valid image."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(mask_path)
        if mask_img is None:
            raise FileNotFoundError(  # same rationale — fail loudly rather than mask missing/mismatched files
                f"Could not read mask '{mask_path}' for image '{img_name}'. "
                f"Expected a mask with the same basename and extension '{self.mask_ext}' in '{self.mask_dir}'."
            )
        
        # If the mask is RGB (like CamVid), convert it to class indices
        if len(mask_img.shape) == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            mask = np.zeros(mask_img.shape[:2], dtype=np.int64)
            for idx, color in enumerate(CLASS_COLORS):
                mask[(mask_img == color).all(axis=2)] = idx
        else:
            mask = mask_img

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask = mask.to(torch.long) if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.long)  # required dtype for CrossEntropyLoss

        return image, mask

def get_train_transform():
    return A.Compose([
        A.Resize(height=384, width=640),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(height=384, width=640),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

if __name__ == "__main__":
    print("Testing Dataset and Transforms...")
    transform = get_train_transform()

    dummy_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 19, (720, 1280), dtype=np.uint8)

    augmented = transform(image=dummy_img, mask=dummy_mask)
    img_tensor = augmented['image']
    mask_tensor = augmented['mask']

    print(f"Image tensor shape: {img_tensor.shape}")  # expected [3, 384, 640]
    print(f"Mask tensor shape: {mask_tensor.shape}")   # expected [384, 640]

    os.makedirs("dummy_images", exist_ok=True)
    os.makedirs("dummy_masks", exist_ok=True)
    cv2.imwrite("dummy_images/test.png", dummy_img)
    cv2.imwrite("dummy_masks/test.png", dummy_mask)

    dataset = UrbanDashcamDataset("dummy_images", "dummy_masks", transform=transform)
    img, mask = dataset[0]
    print(f"__getitem__ output shapes: image={tuple(img.shape)}, mask={tuple(mask.shape)}")

    try:
        UrbanDashcamDataset("dummy_images", "nonexistent_masks", transform=transform)[0]
        raise AssertionError("Expected FileNotFoundError for a missing mask, but none was raised.")
    except FileNotFoundError:
        print("Confirmed: missing mask files raise FileNotFoundError instead of failing silently.")

    os.remove("dummy_images/test.png")
    os.remove("dummy_masks/test.png")
    os.rmdir("dummy_images")
    os.rmdir("dummy_masks")

    print("Dataset pipeline is ready.")
