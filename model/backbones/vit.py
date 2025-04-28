import torch
import torch.nn as nn
import torchvision.models as models

def get_vit(model_name="vit_b_16", pretrained=True):
    model = models.get_model(model_name, pretrained=pretrained)
    if not isinstance(model, models.VisionTransformer):
         raise ValueError(f"Model {model_name} is not a VisionTransformer instance.")
    model.heads = nn.Identity()
    out_channels = model.hidden_dim
    return model, out_channels

if __name__ == "__main__":
    model_name = "vit_b_16"
    vit_model, channels = get_vit(model_name=model_name, pretrained=True)

    print(f"Model: {model_name} Feature Extractor")
    print(vit_model) # Model structure can be large
    print(f"Output Embedding Dimension: {channels}")
    print("Params:", sum(p.numel() for p in vit_model.parameters() if p.requires_grad))

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = vit_model(test_data)

    print(f"Output shape for input {test_data.shape}: {test_outputs.size()}")

