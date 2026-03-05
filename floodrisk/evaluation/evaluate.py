"""Evaluation pipeline for streamflow models."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


class EvaluationPipeline:
    """Run inference on a test dataset and compute per-basin metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained streamflow model.
    test_dataset : Dataset
        A :class:`~floodrisk.datasets.streamflow.CatchmentDataset` for the
        test period.
    device : str
        Torch device string.
    """

    def __init__(self, model: torch.nn.Module, test_dataset, device: str = "cpu") -> None:
        self.model = model
        self.test_dataset = test_dataset
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def run(self) -> dict:
        """Run inference and compute per-basin NSE and KGE.

        Returns
        -------
        dict
            ``{"basin_id": {"nse": float, "kge": float, "obs": array, "pred": array}, ...}``
        """
        # Group predictions by basin
        basin_preds: dict[str, list] = {}
        basin_obs: dict[str, list] = {}

        for idx in range(len(self.test_dataset)):
            basin_id, _ = self.test_dataset.samples[idx]
            x, y = self.test_dataset[idx]
            x = x.unsqueeze(0).to(self.device)
            pred = self.model(x).cpu().numpy().flatten()
            obs = y.numpy().flatten()

            basin_preds.setdefault(basin_id, []).append(pred)
            basin_obs.setdefault(basin_id, []).append(obs)

        results = {}
        for basin_id in basin_preds:
            pred_all = np.concatenate(basin_preds[basin_id])
            obs_all = np.concatenate(basin_obs[basin_id])
            nse = self._compute_nse(obs_all, pred_all)
            kge = self._compute_kge(obs_all, pred_all)
            results[basin_id] = {
                "nse": nse,
                "kge": kge,
                "obs": obs_all,
                "pred": pred_all,
            }

        return results

    def summary(self, results: dict) -> dict:
        """Compute summary statistics across basins.

        Parameters
        ----------
        results : dict
            Output of :meth:`run`.

        Returns
        -------
        dict
            Aggregated statistics: median, mean, and percentiles of
            basin-level NSE and KGE.
        """
        nse_values = [r["nse"] for r in results.values()]
        kge_values = [r["kge"] for r in results.values()]

        return {
            "n_basins": len(results),
            "nse_median": float(np.median(nse_values)),
            "nse_mean": float(np.mean(nse_values)),
            "nse_p10": float(np.percentile(nse_values, 10)),
            "nse_p25": float(np.percentile(nse_values, 25)),
            "nse_p75": float(np.percentile(nse_values, 75)),
            "nse_p90": float(np.percentile(nse_values, 90)),
            "kge_median": float(np.median(kge_values)),
            "kge_mean": float(np.mean(kge_values)),
        }

    @staticmethod
    def _compute_nse(obs: np.ndarray, pred: np.ndarray) -> float:
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def _compute_kge(obs: np.ndarray, pred: np.ndarray) -> float:
        if np.std(obs) == 0 or np.mean(obs) == 0:
            return float("-inf")
        r = float(np.corrcoef(obs, pred)[0, 1])
        alpha = float(np.std(pred) / np.std(obs))
        beta = float(np.mean(pred) / np.mean(obs))
        return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
