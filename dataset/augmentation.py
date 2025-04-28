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

        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):
            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), self.fill, "constant")
                return F.resize(image, [self.h, self.w])
            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), self.fill, "constant")
                return F.resize(image, [self.h, self.w])
        else:
            return F.resize(image, [self.h, self.w])

