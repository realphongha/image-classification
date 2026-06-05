import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class Lion(Optimizer):
    """Lion: Evolved Sign Momentum optimizer.

    Ref: https://arxiv.org/abs/2302.06675

    Optimizer discovered via program search. Uses only sign operations for
    parameter and momentum updates, making it memory-efficient.
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                update = update.sign()

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
