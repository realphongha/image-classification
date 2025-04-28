import torch.optim as optim


def get_optimizer(model, cfg):
    name = cfg["name"]
    if name == "adam":
        optimizer = optim.Adam(model.parameters(), **cfg[name])
    elif name == "sgd":
        optimizer = optim.SGD(model.parameters(), **cfg[name])
    elif name == "adamw":
        optimizer = optim.AdamW(model.parameters(), **cfg[name])
    else:
        raise NotImplementedError(f"Optimizer {name} is not supported!")
    return optimizer

