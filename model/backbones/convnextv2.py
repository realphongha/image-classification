import torch
import torch.nn as nn
import torchvision.models as models


def get_convnext_v2(size="base", pretrained=True):
    assert size in ["tiny", "base", "large"], "Size must be one of 'tiny', 'base', 'large'"
    if size == "tiny":
        model = models.convnext_tiny(pretrained=pretrained)
    elif size == "base":
        model = models.convnext_base(pretrained=pretrained)
    elif size == "large":
        model = models.convnext_large(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported size: {size}")
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_output = model.features(dummy_input)
    out_channels = dummy_output.shape[1]
    model = nn.Sequential(*(list(model.children())[:-2]))
    return model, out_channels


if __name__ == "__main__":
    import torch
    model, channels = get_convnext_v2("base", True)
    print(f"Output Channels: {channels}")
    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(f"Output shape: {test_outputs.size()}")
