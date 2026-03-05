"""Ensemble evaluation metrics for probabilistic streamflow forecasts."""

from __future__ import annotations

import torch

from floodrisk.torchharness import Metric


class CRPSMetric(Metric):
    """Fair Continuous Ranked Probability Score (CRPS) for ensembles.

    fCRPS = (1/N) sum|x_n - y| - 1/(2N(N-1)) sum|x_n - x_n'|

    Lower is better.  Expects ensemble predictions of shape
    ``[batch, n_ensemble, forecast_horizon]``.
    """

    def __init__(self) -> None:
        self._values: list[float] = []
        self._counts: list[int] = []

    @property
    def name(self) -> str:
        return "crps"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # preds: [B, N, H] or [B, H] (deterministic fallback)
        if preds.dim() == 2:
            # Deterministic → treat as single-member ensemble, CRPS = MAE
            mae = (preds - targets).abs().mean()
            self._values.append(mae.item())
            self._counts.append(preds.size(0))
            return

        N = preds.size(1)
        targets_exp = targets.unsqueeze(1)  # [B, 1, H]

        # Reliability term
        mae = (preds - targets_exp).abs().mean(dim=1)  # [B, H]

        # Sharpness term
        if N > 1:
            diffs = (preds.unsqueeze(2) - preds.unsqueeze(1)).abs()  # [B, N, N, H]
            spread = diffs.sum(dim=(1, 2)) / (2 * N * (N - 1))  # [B, H]
        else:
            spread = torch.zeros_like(mae)

        crps = (mae - spread).mean()
        self._values.append(crps.item())
        self._counts.append(preds.size(0))

    def compute(self) -> float:
        if not self._values:
            return float("nan")
        total = sum(v * c for v, c in zip(self._values, self._counts))
        return total / sum(self._counts)

    def reset(self) -> None:
        self._values.clear()
        self._counts.clear()


class SpreadSkillMetric(Metric):
    """Spread-skill ratio for ensemble predictions.

    Ratio of ensemble spread (mean std across members) to RMSE of the
    ensemble mean.  Ideal value is 1.0 — under-dispersive < 1, over-dispersive > 1.
    """

    def __init__(self) -> None:
        self._spreads: list[torch.Tensor] = []
        self._errors: list[torch.Tensor] = []

    @property
    def name(self) -> str:
        return "spread_skill"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.dim() == 2:
            return  # Skip deterministic predictions
        # Ensemble spread: std across members → mean over horizon
        spread = preds.std(dim=1).mean(dim=-1)  # [B]
        # Ensemble mean error
        mean_pred = preds.mean(dim=1)  # [B, H]
        sq_err = ((mean_pred - targets) ** 2).mean(dim=-1)  # [B]
        self._spreads.append(spread.detach().cpu())
        self._errors.append(sq_err.detach().cpu())

    def compute(self) -> float:
        if not self._spreads:
            return float("nan")
        spread = torch.cat(self._spreads).mean()
        rmse = torch.cat(self._errors).mean().sqrt()
        if rmse == 0:
            return float("inf")
        return (spread / rmse).item()

    def reset(self) -> None:
        self._spreads.clear()
        self._errors.clear()


class EnsembleNSEMetric(Metric):
    """NSE computed on the ensemble mean prediction.

    Accepts ensemble preds ``[B, N, H]`` and reduces to mean before computing
    standard NSE.  Also works with deterministic ``[B, H]`` inputs.
    """

    def __init__(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    @property
    def name(self) -> str:
        return "ensemble_nse"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.dim() == 3:
            preds = preds.mean(dim=1)
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> float:
        if not self._preds:
            return float("nan")
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
