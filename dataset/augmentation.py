import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from typing import Dict, Any

class ResizeWithPad(nn.Module):
    """Symmetrically pads and resizes an image to preserve exact aspect ratio."""
    def __init__(self, w: int, h: int, fill: int = 127):
        super().__init__()
        self.w = w
        self.h = h
        self.fill = fill

    def forward(self, image: Any) -> Any:
        h_in, w_in = F.get_size(image)
        ratio_target = self.w / self.h
        ratio_in = w_in / h_in

        if ratio_in > ratio_target:
            new_h = int(w_in / ratio_target)
            pad_total = new_h - h_in
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            image = F.pad(image, [0, pad_top, 0, pad_bottom], fill=self.fill)
        elif ratio_in < ratio_target:
            new_w = int(h_in * ratio_target)
            pad_total = new_w - w_in
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            image = F.pad(image, [pad_left, 0, pad_right, 0], fill=self.fill)

        return F.resize(image, [self.h, self.w])


def get_aug(cfg: Dict[str, Any], is_train: bool) -> T.Compose:
    """
    The Ultimate 2026 Image Classification Augmentation Suite.
    Parses an expansive, modern YAML config mapping out all SOTA transforms.
    """
    data_cfg = cfg["data"]
    aug_cfg = data_cfg.get("augmentation", {})
    pipeline = []

    # Always initialize by converting to TVTensor Image format
    pipeline.append(T.ToImage())

    if is_train:
        # ==========================================
        # 1. MODERN FOUNDATIONAL POLICIES (High Entropy)
        # ==========================================
        if "policy_type" in aug_cfg:
            p_type = aug_cfg["policy_type"].lower()
            if p_type == "trivial":
                pipeline.append(T.TrivialAugmentWide())
            elif p_type == "randaug":
                pipeline.append(T.RandAugment(num_ops=2, magnitude=9))
            elif p_type == "autoaug":
                pipeline.append(T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))
            else:  # no
                p_type = "no"

        # ==========================================
        # 2. ADVANCED GEOMETRIC TRANSFORMS
        # ==========================================
        # The SOTA standard: breaks localized coordinate dependency
        if aug_cfg.get("random_resized_crop", {}).get("enabled", False):
            rrc_cfg = aug_cfg["random_resized_crop"]
            pipeline.append(T.RandomResizedCrop(
                size=(data_cfg["input_h"], data_cfg["input_w"]),
                scale=rrc_cfg.get("scale", [0.08, 1.0]),
                ratio=rrc_cfg.get("ratio", [0.75, 1.333])
            ))

        if aug_cfg.get("horizontal_flip_p", 0) > 0:
            pipeline.append(T.RandomHorizontalFlip(p=aug_cfg["horizontal_flip_p"]))

        if aug_cfg.get("vertical_flip_p", 0) > 0:
            pipeline.append(T.RandomVerticalFlip(p=aug_cfg["vertical_flip_p"]))

        if "random_rotation" in aug_cfg and p_type == "no":
            rot_cfg = aug_cfg["random_rotation"]
            pipeline.append(T.RandomApply(
                [T.RandomRotation(degrees=rot_cfg.get("degrees", 15))],
                p=rot_cfg["prob"]
            ))

        if "perspective" in aug_cfg:
            args = aug_cfg["perspective"].get("args", {})
            pipeline.append(T.RandomPerspective(
                distortion_scale=args.get("distortion_scale", 0.5),
                p=aug_cfg["perspective"]["prob"],
                fill=args.get("fill", 127)
            ))

        # ==========================================
        # 3. ADVANCED PHOTOMETRIC & CHROMATIC TRANSFORMS
        # ==========================================
        if "color_jitter" in aug_cfg:
            node = aug_cfg["color_jitter"]
            pipeline.append(T.RandomApply(
                [T.ColorJitter(**node.get("args", {}))], p=node["prob"]
            ))

        if "gaussian_blur" in aug_cfg:
            node = aug_cfg["gaussian_blur"]
            pipeline.append(T.RandomApply(
                [T.GaussianBlur(**node.get("args", {}))], p=node["prob"]
            ))

        if "grayscale" in aug_cfg:
            pipeline.append(T.RandomApply(
                [T.Grayscale(num_output_channels=3)], p=aug_cfg["grayscale"]["prob"]
            ))

        # 2026 Staples: Extreme pixel value inversions to fight sensor/domain bias
        if "solarize" in aug_cfg and p_type == "no":
            node = aug_cfg["solarize"]
            pipeline.append(T.RandomApply(
                [T.RandomSolarize(threshold=node.get("threshold", 128), p=1.0)], p=node["prob"]
            ))

        if "posterize" in aug_cfg and p_type == "no":
            node = aug_cfg["posterize"]
            pipeline.append(T.RandomApply(
                [T.RandomPosterize(bits=node.get("bits", 4), p=1.0)], p=node["prob"]
            ))

        if "equalize" in aug_cfg and p_type == "no":
            pipeline.append(T.RandomEqualize(p=aug_cfg["equalize"]["prob"]))

    # ==========================================
    # 4. SPATIAL HARMONIZATION & RESIZING
    # ==========================================
    # If RandomResizedCrop wasn't active, fallback to safe aspect padding
    if not aug_cfg.get("random_resized_crop", {}).get("enabled", False) or not is_train:
        pipeline.append(ResizeWithPad(data_cfg["input_w"], data_cfg["input_h"]))

    # ==========================================
    # 5. TENSOR CONDITIONING & REGULARIZATION
    # ==========================================
    pipeline.append(T.ToDtype(torch.float32, scale=True))
    pipeline.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if is_train and "random_erasing" in aug_cfg:
        # Mandatory for ViTs: Prevents models from depending on singular local features
        node = aug_cfg["random_erasing"]
        args = node.get("args", {})
        pipeline.append(T.RandomErasing(
            p=node["prob"],
            scale=args.get("scale", [0.02, 0.33]),
            value=args.get("value", "random")
        ))

    return T.Compose(pipeline)

