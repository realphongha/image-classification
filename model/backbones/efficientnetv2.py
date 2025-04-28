import torch.nn as nn
import torchvision.models as models
import torch


def get_efficientnet_v2(size="s", pretrained=True):
    assert size in ["s", "m", "l"], "Size must be one of 's', 'm', 'l'"
    if size == "s":
        model = models.efficientnet_v2_s(pretrained=pretrained)
    elif size == "m":
        model = models.efficientnet_v2_m(pretrained=pretrained)
    elif size == "l":
        model = models.efficientnet_v2_l(pretrained=pretrained)
    out_channels = model.features[-1].out_channels
    model = nn.Sequential(*(list(model.children())[:-2]))
    return model, out_channels


if __name__ == "__main__":
    model, channels = get_efficientnet_v2("s", True)
    print("Model Architecture (Features Only):")
    # Printing the whole feature extractor can be very long, print a summary instead
    print(f"Type: {type(model)}")
    print(f"Output Channels: {channels}")

    print("\nParams:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Example usage with dummy data
    test_data = torch.rand(5, 3, 384, 384)  # Batch size 5, 3 channels, 384x384 image
    print(f"\nInput shape: {test_data.shape}")

    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        test_outputs = model(test_data)

    print(f"Output shape: {test_outputs.size()}")

