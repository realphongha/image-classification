from .linear import *
from .stacked_linear import *


def get_head(configs, in_features):
    num_classes = len(configs["data"]["cls"])
    head_cfg = configs["model"]["head"]
    name = head_cfg["name"]
    if name == "linear":
        return Linear(in_features, num_classes, **head_cfg[name])
    elif name == "stacked_linear":
        return StackedLinear(in_features, num_classes, **head_cfg[name])
    else:
        raise ValueError(f"Unsupported head name: {name}")

