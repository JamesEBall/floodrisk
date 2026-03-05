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


class CRPSLoss(nn.Module):
    """Fair CRPS loss for ensemble predictions.

    fCRPS(x^{1:N}, y) = (1/N) sum_n |x_n - y|
                       - 1/(2N(N-1)) sum_{n,n'} |x_n - x_n'|

    Parameters
    ----------
    ensemble_preds : Tensor [batch, n_ensemble, forecast_horizon]
    targets : Tensor [batch, forecast_horizon]
    """

    def forward(self, ensemble_preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N = ensemble_preds.shape[1]
        targets_exp = targets.unsqueeze(1)  # [B, 1, H]

        # Reliability: mean absolute error across ensemble members
        mae = (ensemble_preds - targets_exp).abs().mean(dim=1)  # [B, H]

        # Sharpness: pairwise differences between ensemble members
        if N > 1:
            diffs = (ensemble_preds.unsqueeze(2) - ensemble_preds.unsqueeze(1)).abs()
            spread = diffs.sum(dim=(1, 2)) / (2 * N * (N - 1))  # [B, H]
        else:
            spread = torch.zeros_like(mae)

        crps = mae - spread
        return crps.mean()
