import torch
from torch.nn import CrossEntropyLoss
from .focal_loss import FocalLoss


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
    elif name == "focal":
        gamma = cfg[name]["gamma"]
        alpha = cfg[name]["alpha"]
        if not gamma:
            gamma = 2
        if not alpha:
            alpha = None
        criterion = FocalLoss(gamma=gamma, alpha=torch.Tensor(alpha).to(device))
    else:
        raise NotImplementedError(f"{name} loss is not implemented!")
    return criterion

