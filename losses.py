import torch
import torch.nn as nn
import torch.nn.functional as F

from classes import NUM_CLASSES

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes  # falls back to inputs.shape[1] if None

    def forward(self, inputs, targets, mask=None):
        num_classes = self.num_classes or inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if mask is not None:
            mask = mask.unsqueeze(1).float()  # broadcast (B,H,W) over the class dim
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask

        dims = (0, 2, 3)
        intersection = (inputs * targets_one_hot).sum(dim=dims)
        cardinality = inputs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)  # per-class, then mean — keeps rare classes from being drowned out by dominant ones

        return 1 - dice_per_class.mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=NUM_CLASSES):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=255)  # 255 is the standard "unlabelled" value
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)

        valid_mask = targets != 255
        if not valid_mask.any():
            return ce_loss

        dice_target = torch.where(valid_mask, targets, torch.tensor(0, device=targets.device))  # 255 isn't a valid class index for one_hot, so swap it to 0 — dice's own mask excludes it from the sums anyway
        dice_loss = self.dice(inputs, dice_target, mask=valid_mask)

        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)

if __name__ == "__main__":
    print("Testing Combined Loss...")
    criterion = CombinedLoss(num_classes=NUM_CLASSES)
    dummy_pred = torch.randn(2, NUM_CLASSES, 384, 640)
    dummy_target = torch.randint(0, NUM_CLASSES, (2, 384, 640))
    dummy_target[0, :10, :10] = 255  # a block of ignore-index pixels
    
    loss = criterion(dummy_pred, dummy_target)
    print(f"Calculated Loss: {loss.item()}")
