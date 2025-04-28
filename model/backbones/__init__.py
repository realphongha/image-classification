from .mobilenetv3 import *
from .shufflenetv2 import *
from .resnet import *
from .vit import *
from .efficientnetv2 import *


def get_backbone(configs):
    name = configs["model"]["backbone"]["name"]
    if name == "mobilenetv3":
        return get_mobilenet_v3(**configs["model"]["backbone"][name])
    elif name == "shufflenetv2":
        return get_shufflenetv2(**configs["model"]["backbone"][name])
    elif name == "resnet":
        return get_resnet(**configs["model"]["backbone"][name])
    elif name == "vit":
        return get_vit(**configs["model"]["backbone"][name])
    elif name == "efficientnetv2":
        return get_efficientnet_v2(**configs["model"]["backbone"][name])
    else:
        raise ValueError(f"Unsupported backbone name: {name}")

