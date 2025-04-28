import logging
import os
from collections import Counter

import torch
import numpy as np
from PIL import Image

from .augmentation import get_aug


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, cfg):
        self.data_path = data_path if isinstance(data_path, list) else [data_path]
        self.is_train = is_train
        self.cfg = cfg
        self.cls = cfg["data"]["cls"]
        self.cls_dict = {}
        for i, cls in enumerate(self.cls):
            self.cls_dict[cls] = i
        self.data = []
        self.labels = []
        exts = cfg["data"]["img_ext"]
        for data_path in self.data_path:
            for d in os.listdir(data_path):
                dir = os.path.join(data_path, d)
                if not os.path.isdir(dir) or d not in self.cls: continue
                for fn in os.listdir(dir):
                    if os.path.splitext(fn)[-1] not in exts:
                        continue
                    fp = os.path.join(dir, fn)
                    self.data.append(fp)
                    self.labels.append(self.cls_dict[d])
        lbl_counter = Counter(self.labels).items()
        self.cls_count = [0] * len(self.cls)
        for cls, c in lbl_counter:
            self.cls_count[cls] = c
        lbl_counter = [(self.cls[cls], c) for cls, c in lbl_counter]
        logging.info("Classes count:")
        logging.info(lbl_counter)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        fp = self.data[index]
        label = self.labels[index]
        raw_img = Image.open(fp)
        raw_img = raw_img.convert('RGB')
        aug = get_aug(self.cfg, self.is_train)
        img = aug(raw_img)
        img = np.array(img).astype(np.float32)
        # import os
        # import cv2
        # os.makedirs("runs/aug", exist_ok=True)
        # cv2.imwrite(f"runs/aug/cls{label}-{index}.jpg", img)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        data = torch.Tensor(img)
        return data, label

