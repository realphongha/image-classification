import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet(size=18, pretrained=True):
    resnet_mapping = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    assert size in resnet_mapping, f"ResNet size must be one of {list(resnet_mapping.keys())}"

    model_func = resnet_mapping[size]
    model = model_func(pretrained=pretrained)

    modules = list(model.children())[:-2]
    feature_extractor = nn.Sequential(*modules)

    dummy_input = torch.randn(1, 3, 224, 224) # Standard ResNet input size
    dummy_output = feature_extractor(dummy_input)
    out_channels = dummy_output.shape[1]
    return feature_extractor, out_channels


if __name__ == "__main__":
    import torch
    model, channels = get_resnet(size=18, pretrained=True)
    print(model)
    print(f"Model: ResNet-18 Feature Extractor")
    # print(model) # Can be very long
    print(f"Output Channels: {channels}")
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Test with dummy data
    test_data = torch.rand(5, 3, 224, 224) # Standard ResNet input size
    test_outputs = model(test_data)
    print(f"Output shape for input {test_data.shape}: {test_outputs.size()}") # Shape will depend on input size

    # Example usage: Get ResNet-50
    model_50, channels_50 = get_resnet(size=50, pretrained=True)
    print(f"\nModel: ResNet-50 Feature Extractor")
    print(f"Output Channels: {channels_50}")
    print("Params:", sum(p.numel() for p in model_50.parameters() if p.requires_grad))
    test_outputs_50 = model_50(test_data)
    print(f"Output shape for input {test_data.shape}: {test_outputs_50.size()}")

