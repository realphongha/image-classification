import torch.nn as nn
import logging

from .backbones import get_backbone
from .heads import get_head
from .necks import get_neck


class ClassificationModel(nn.Module):
    def __init__(self, cfg, training):
        super(ClassificationModel, self).__init__()
        self.backbone, self.in_channels = get_backbone(cfg)
        self.neck = get_neck(cfg)
        self.head = get_head(cfg, self.in_channels)
        self.training = training

    def remove_fc(self):
        self.head[-1] = nn.Identity()

    def freeze(self, parts):
        for part in parts:
            if part not in ("backbone", "neck", "head"):
                raise NotImplementedError("Cannot freeze %s!" % part)
            logging.info("Freezing %s..." % part)
            for name, p in self.named_parameters():
                if part in name:
                    logging.info("Freezing %s..." % name)
                    p.requires_grad = False

    def free(self, parts):
        for part in parts:
            if part not in ("backbone", "neck", "head"):
                raise NotImplementedError("Cannot free %s!" % part)
            logging.info("Freeing %s..." % part)
            for name, p in self.named_parameters():
                if part in name:
                    logging.info("Freeing %s..." % name)
                    p.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    import yaml
    import torch

    with open("./configs/cifar100/cifar100_mobilenetv3_small.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            quit()
    model = ClassificationModel(cfg, False)
    print(model)
    print("Params:", sum(p.numel() for p in model.parameters()))
    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())

