import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def seed_everything(seed: int = 42):
    """Seed random, numpy, and torch globally for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_cuda_cudnn(cfg):
    cudnn.benchmark = cfg["cudnn"]["benchmark"]
    cudnn.deterministic = cfg["cudnn"]["deterministic"]
    cudnn.enabled = cfg["cudnn"]["enabled"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if cfg.get("gpus", None):
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    assert torch.cuda.is_available(), "CUDA is not available!"
    torch.set_float32_matmul_precision('high')
