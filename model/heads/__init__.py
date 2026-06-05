from .linear import *
from .stacked_linear import *
from .regression import *


def get_head(configs, in_features):
    head_cfg = configs["model"]["head"]
    name = head_cfg["name"]
    if name == "linear":
        num_classes = len(configs["data"]["cls"])
        return Linear(in_features, num_classes, **head_cfg[name])
    elif name == "stacked_linear":
        num_classes = len(configs["data"]["cls"])
        return StackedLinear(in_features, num_classes, **head_cfg[name])
    elif name == "regression":
        out_features = head_cfg[name].get("out_features", 1)
        return RegressionHead(in_features, out_features, **head_cfg[name])
    else:
        raise ValueError(f"Unsupported head name: {name}")

