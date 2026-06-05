import torch.optim as optim
from .lion import Lion


def get_optimizer(model, cfg):
    name = cfg["name"]
    base_kwargs = dict(cfg[name])
    backbone_lr_mult = cfg.get("backbone_lr_mult", None)

    if backbone_lr_mult is not None:
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in n:
                backbone_params.append(p)
            else:
                head_params.append(p)
        param_groups = [
            {"params": head_params, **base_kwargs},
            {"params": backbone_params, **base_kwargs, "lr": base_kwargs["lr"] * backbone_lr_mult},
        ]
        if name == "adam":
            optimizer = optim.Adam(param_groups)
        elif name == "sgd":
            optimizer = optim.SGD(param_groups)
        elif name == "adamw":
            optimizer = optim.AdamW(param_groups)
        elif name == "lion":
            optimizer = Lion(param_groups)
        else:
            raise NotImplementedError(f"Optimizer {name} is not supported!")
    else:
        if name == "adam":
            optimizer = optim.Adam(model.parameters(), **base_kwargs)
        elif name == "sgd":
            optimizer = optim.SGD(model.parameters(), **base_kwargs)
        elif name == "adamw":
            optimizer = optim.AdamW(model.parameters(), **base_kwargs)
        elif name == "lion":
            optimizer = Lion(model.parameters(), **base_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {name} is not supported!")
    return optimizer

