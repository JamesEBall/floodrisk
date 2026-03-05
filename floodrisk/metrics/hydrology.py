"""Hydrological efficiency metrics (NSE, KGE)."""

from __future__ import annotations

import torch

from floodrisk.torchharness import Metric


class NSEMetric(Metric):
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - SS_res / SS_tot

    Perfect prediction gives 1.0; predicting the mean gives 0.0.
    """

    def __init__(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    @property
    def name(self) -> str:
        return "nse"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> float:
        preds = torch.cat(self._preds).flatten()
        targets = torch.cat(self._targets).flatten()
        ss_res = ((targets - preds) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        if ss_tot == 0:
            return 0.0
        return (1.0 - ss_res / ss_tot).item()

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()


class KGEMetric(Metric):
    """Kling-Gupta Efficiency.

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)

    where r = Pearson correlation, alpha = std_pred / std_obs,
    beta = mean_pred / mean_obs.
    """

    def __init__(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    @property
    def name(self) -> str:
        return "kge"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> float:
        preds = torch.cat(self._preds).flatten()
        targets = torch.cat(self._targets).flatten()

        mean_pred = preds.mean()
        mean_obs = targets.mean()
        std_pred = preds.std(correction=0)
        std_obs = targets.std(correction=0)

        if std_obs == 0 or mean_obs == 0:
            return float("-inf")

        # Pearson correlation
        cov = ((preds - mean_pred) * (targets - mean_obs)).mean()
        r = cov / (std_pred * std_obs) if std_pred > 0 else torch.tensor(0.0)

        alpha = std_pred / std_obs
        beta = mean_pred / mean_obs

        kge = 1.0 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2).sqrt()
        return kge.item()

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()
