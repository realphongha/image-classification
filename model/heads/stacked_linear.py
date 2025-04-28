import torch.nn as nn


class StackedLinear(nn.Module):
    def __init__(self, in_features, num_classes, hidden_channel,
                 activation="relu", dropout_rate=0.0):
        super(StackedLinear, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_channel)
        assert activation in ("relu", "hardswish")
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "hardswish":
            self.activation = nn.Hardswish(inplace=True)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        else:
            self.dropout = nn.Identity()
        self.linear2 = nn.Linear(hidden_channel, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
