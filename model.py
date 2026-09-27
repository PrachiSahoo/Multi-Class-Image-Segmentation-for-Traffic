import torch
import segmentation_models_pytorch as smp

def create_unet_model(num_classes, encoder_name="resnet34", encoder_weights="imagenet"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    return model

if __name__ == "__main__":
    from classes import NUM_CLASSES

    model = create_unet_model(num_classes=NUM_CLASSES)
    dummy_input = torch.randn(2, 3, 384, 640)  # batch 2, 3 channels, 384x640

    print("Testing U-Net model forward pass...")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [2, 19, 384, 640]
    print("Model test successful!")
