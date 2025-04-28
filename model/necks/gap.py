import torch.nn as nn


class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x

