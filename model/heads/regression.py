import torch.nn as nn


class RegressionHead(nn.Module):
    """Regression head that outputs continuous values."""

    def __init__(self, in_features: int, out_features: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
