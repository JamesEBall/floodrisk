"""Loss functions for hydrological prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class NSELoss(nn.Module):
    """Nash-Sutcliffe Efficiency loss for streamflow prediction.

    NSE = 1 - sum((pred - obs)^2) / sum((obs - mean(obs))^2)
    Loss = (1 - NSE).mean()  -- averaged over batch, minimized toward NSE=1.

    When target variance is near zero the denominator is clamped so the loss
    degrades gracefully to MSE-like behaviour rather than exploding.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds, targets: [batch, forecast_horizon]
        mean_obs = targets.mean(dim=-1, keepdim=True)
        numerator = ((preds - targets) ** 2).sum(dim=-1)
        denominator = ((targets - mean_obs) ** 2).sum(dim=-1)
        nse = 1.0 - numerator / (denominator + self.eps)
        return (1.0 - nse).mean()
