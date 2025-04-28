import os

import torch
import torch.backends.cudnn as cudnn


def init_cuda_cudnn(cfg):
    cudnn.benchmark = cfg["cudnn"]["benchmark"]
    cudnn.deterministic = cfg["cudnn"]["deterministic"]
    cudnn.enabled = cfg["cudnn"]["enabled"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if cfg["gpus"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    assert torch.cuda.is_available(), "CUDA is not available!"
    torch.set_float32_matmul_precision('high')

