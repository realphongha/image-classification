import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.0):
        super(Linear, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
