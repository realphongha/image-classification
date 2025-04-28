import torch.optim as optim
from .lr_scheduler import *


def get_scheduler(cfg, optimizer, last_epoch, total_steps):
    name = cfg["name"]
    warmup = cfg["warmup"]
    if not warmup:
        warmup = 0
    if name == "multistep":
        lr_scheduler = WarmupMultiStepSchedule(
            optimizer, warmup, last_epoch=last_epoch, **cfg[name])
    elif name == "constant":
        lr_scheduler = WarmupConstantSchedule(
            optimizer, warmup, last_epoch=last_epoch
        )
    elif name == "cosine":
        lr_scheduler = WarmupCosineSchedule(
            optimizer, warmup, last_epoch=last_epoch, t_total=total_steps, **cfg[name]
        )
    else:
        raise NotImplementedError(f"{name} lr scheduler is not implemented!")
    return lr_scheduler

