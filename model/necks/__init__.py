from .gap import *


def get_neck(configs):
    name = configs["model"]["neck"]["name"]
    if name == "gap":
        return GAP()
    elif not name:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported neck name: {name}")
