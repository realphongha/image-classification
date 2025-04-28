import torch.nn as nn
import torchvision.models as models


def get_shufflenetv2(width_mult="1.0x", pretrained=True):
    assert width_mult in ["0.5x", "1.0x", "1.5x", "2.0x"]
    if width_mult == "0.5x":
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
    elif width_mult == "1.0x":
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    elif width_mult == "1.5x":
        model = models.shufflenet_v2_x1_5(pretrained=pretrained)
    elif width_mult == "2.0x":
        model = models.shufflenet_v2_x2_0(pretrained=pretrained)
    model = nn.Sequential(*(list(model.children())[:-1]))
    out_channels = model[-1][-2].num_features
    return model, out_channels


if __name__ == "__main__":
    import torch
    model, _ = get_shufflenetv2("0.5x", True)
    print(model)
    print("Params:", sum(p.numel() for p in model.parameters()))

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())

