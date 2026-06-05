import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    """PolyLoss: A Polynomial Expansion Loss for Classification.

    Ref: https://arxiv.org/abs/2204.12564

    PolyLoss decomposes the cross-entropy loss into a series of polynomial terms
    to better model the relationship between predicted probabilities and true labels.
    """

    def __init__(self, num_classes: int, epsilon: float = 1.0, weight: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = F.softmax(logits, dim=1)[torch.arange(logits.shape[0], device=logits.device), targets]
        poly1_loss = 1.0 - pt + self.epsilon
        poly1_ce = ce_loss * poly1_loss
        return poly1_ce.mean()
