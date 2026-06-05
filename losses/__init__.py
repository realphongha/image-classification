import torch
from torch.nn import CrossEntropyLoss, HuberLoss
from .focal_loss import FocalLoss
from .poly_loss import PolyLoss


def get_loss(cfg, device, cls_count):
    name = cfg["name"]
    if name == "ce":
        weighted = cfg[name]["weighted"]
        if weighted:
            counts = torch.tensor(cls_count)
            weights = 1.0 / counts.float()     # inverse frequency
            weights = weights / weights.sum()  # (optional) normalize
            criterion = CrossEntropyLoss(weight=weights.to(device))
        else:
            criterion = CrossEntropyLoss()
    elif name == "poly":
        epsilon = cfg[name].get("epsilon", 1.0)
        criterion = PolyLoss(
            num_classes=len(cls_count),
            epsilon=epsilon,
        )
    elif name == "huber":
        delta = cfg[name].get("delta", 1.0)
        criterion = HuberLoss(delta=delta)
    elif name == "focal":
        gamma = cfg[name]["gamma"]
        alpha = cfg[name].get("alpha", None)
        if not gamma:
            gamma = 2
        criterion = FocalLoss(gamma=gamma, alpha=torch.tensor(alpha, dtype=torch.float32).to(device) if alpha is not None else None)
    else:
        raise NotImplementedError(f"{name} loss is not implemented!")
    return criterion

