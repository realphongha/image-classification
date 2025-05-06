import random
import math
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F


def get_aug(cfg, is_train):
    aug_cfg = cfg["data"]["augmentation"]
    pipeline = []
    if is_train:
        if aug_cfg.get("autoaug", None) and aug_cfg["autoaug"]["prob"] > random.random():
            if "policy" in aug_cfg["autoaug"]["args"]:
                policy = aug_cfg["autoaug"]["args"]["policy"]
                if isinstance(policy, str):
                    policy = eval("T.autoaugment.AutoAugmentPolicy." + policy)
                    aug_cfg["autoaug"]["args"]["policy"] = policy
            pipeline.append(T.AutoAugment(**aug_cfg["autoaug"]["args"]))
        if aug_cfg.get("gaussian_blur", None) and aug_cfg["gaussian_blur"]["prob"] > random.random():
            pipeline.append(T.GaussianBlur(**aug_cfg["gaussian_blur"]["args"]))
        if aug_cfg.get("grayscale", None) and aug_cfg["grayscale"]["prob"] > random.random():
            pipeline.append(T.Grayscale(num_output_channels=3))
        if aug_cfg.get("perspective", None):
            pipeline.append(T.RandomPerspective(**aug_cfg["perspective"]["args"]))
    pipeline.append(ResizeWithPad(cfg["data"]["input_w"], cfg["data"]["input_h"]))
    return T.Compose(pipeline)


class ResizeWithPad:
    def __init__(self, w, h, fill=127):
        self.w = w
        self.h = h
        self.fill = fill

    def __call__(self, image):
        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        if round(ratio_1, 2) > round(ratio_f, 2):
            # Image is too wide → pad height
            new_h = int(w_1 / ratio_f)
            pad = new_h - h_1
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            image = F.pad(image, (0, pad_top, 0, pad_bottom), fill=self.fill)
        elif round(ratio_1, 2) < round(ratio_f, 2):
            # Image is too tall → pad width
            new_w = int(h_1 * ratio_f)
            pad = new_w - w_1
            pad_left = pad // 2
            pad_right = pad - pad_left
            image = F.pad(image, (pad_left, 0, pad_right, 0), fill=self.fill)

        return F.resize(image, [self.h, self.w])

