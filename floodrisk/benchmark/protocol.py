"""Benchmark protocol for standardized model evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# Standard CAMELS 531-basin subset (IDs loaded from file or default list)
STANDARD_TRAIN_START = "1980-10-01"
STANDARD_TRAIN_END = "1995-09-30"
STANDARD_TEST_START = "1995-10-01"
STANDARD_TEST_END = "2008-09-30"


@dataclass
class BenchmarkResults:
    """Container for benchmark evaluation results."""

    model_name: str
    n_basins: int = 0
    # Per-basin arrays
    basin_ids: list[str] = field(default_factory=list)
    basin_nse: list[float] = field(default_factory=list)
    basin_kge: list[float] = field(default_factory=list)

    # Aggregate metrics
    nse_median: float = float("nan")
    nse_mean: float = float("nan")
    kge_median: float = float("nan")
    kge_mean: float = float("nan")
    nse_high: float = float("nan")
    fhv: float = float("nan")
    peak_timing_mae: float = float("nan")
    csi: float = float("nan")
    pod: float = float("nan")
    far: float = float("nan")
    crps: float = float("nan")

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from per-basin arrays."""
        if not self.basin_nse:
            return
        arr = np.array(self.basin_nse)
        self.nse_median = float(np.median(arr))
        self.nse_mean = float(np.mean(arr))
        self.n_basins = len(self.basin_nse)

        if self.basin_kge:
            karr = np.array(self.basin_kge)
            self.kge_median = float(np.median(karr))
            self.kge_mean = float(np.mean(karr))

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_basins": self.n_basins,
            "nse_median": self.nse_median,
            "nse_mean": self.nse_mean,
            "kge_median": self.kge_median,
            "kge_mean": self.kge_mean,
            "nse_high": self.nse_high,
            "fhv": self.fhv,
            "peak_timing_mae": self.peak_timing_mae,
            "csi": self.csi,
            "pod": self.pod,
            "far": self.far,
            "crps": self.crps,
        }


class BenchmarkProtocol:
    """Standardized evaluation protocol for streamflow models.

    Implements the Kratzert et al. (2019) evaluation protocol:
    - Temporal split: train 1980-1995, test 1995-2008
    - Basin set: 531 standard CAMELS basins (or user-specified)
    - Metrics: NSE, KGE, high-flow NSE, FHV, peak timing, CSI, POD, FAR

    Parameters
    ----------
    basin_ids : list[str]
        Basin IDs to evaluate.
    train_start : str
        Training period start date.
    train_end : str
        Training period end date.
    test_start : str
        Test period start date.
    test_end : str
        Test period end date.
    flood_threshold_percentile : float
        Percentile of observed flow to define flood events (default: 95th).
    high_flow_percentile : float
        Percentile above which to compute high-flow NSE (default: 75th).
    """

    def __init__(
        self,
        basin_ids: list[str],
        train_start: str = STANDARD_TRAIN_START,
        train_end: str = STANDARD_TRAIN_END,
        test_start: str = STANDARD_TEST_START,
        test_end: str = STANDARD_TEST_END,
        flood_threshold_percentile: float = 95.0,
        high_flow_percentile: float = 75.0,
    ) -> None:
        self.basin_ids = basin_ids
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.flood_threshold_percentile = flood_threshold_percentile
        self.high_flow_percentile = high_flow_percentile

    def evaluate(
        self,
        obs: dict[str, np.ndarray],
        pred: dict[str, np.ndarray],
        model_name: str = "model",
    ) -> BenchmarkResults:
        """Evaluate predictions against observations.

        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Mapping of basin_id to observed streamflow array.
        pred : dict[str, np.ndarray]
            Mapping of basin_id to predicted streamflow array.
        model_name : str
            Name for the results.

        Returns
        -------
        BenchmarkResults
        """
        results = BenchmarkResults(model_name=model_name)

        all_obs = []
        all_pred = []

        for basin_id in self.basin_ids:
            if basin_id not in obs or basin_id not in pred:
                continue

            o = np.asarray(obs[basin_id], dtype=np.float64)
            p = np.asarray(pred[basin_id], dtype=np.float64)

            # Remove NaN pairs
            valid = ~(np.isnan(o) | np.isnan(p))
            o, p = o[valid], p[valid]
            if len(o) < 10:
                continue

            nse = self._compute_nse(o, p)
            kge = self._compute_kge(o, p)

            results.basin_ids.append(basin_id)
            results.basin_nse.append(nse)
            results.basin_kge.append(kge)

            all_obs.append(o)
            all_pred.append(p)

        results.compute_aggregates()

        # Compute global flood metrics across all basins
        if all_obs:
            obs_all = np.concatenate(all_obs)
            pred_all = np.concatenate(all_pred)

            results.nse_high = self._compute_high_flow_nse(
                obs_all, pred_all, self.high_flow_percentile
            )
            results.fhv = self._compute_fhv(obs_all, pred_all)
            results.peak_timing_mae = self._compute_peak_timing(obs_all, pred_all)

            threshold = np.percentile(obs_all, self.flood_threshold_percentile)
            csi, pod, far = self._compute_flood_detection(obs_all, pred_all, threshold)
            results.csi = csi
            results.pod = pod
            results.far = far

        return results

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

    @staticmethod
    def _compute_high_flow_nse(
        obs: np.ndarray, pred: np.ndarray, percentile: float
    ) -> float:
        threshold = np.percentile(obs, percentile)
        mask = obs > threshold
        if mask.sum() < 10:
            return float("nan")
        o_high = obs[mask]
        p_high = pred[mask]
        ss_res = np.sum((o_high - p_high) ** 2)
        ss_tot = np.sum((o_high - np.mean(o_high)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def _compute_fhv(obs: np.ndarray, pred: np.ndarray, top_pct: float = 2.0) -> float:
        """Bias in top-X% flow volume (FHV), returned as percentage."""
        n = max(1, int(len(obs) * top_pct / 100.0))
        obs_sorted = np.sort(obs)[::-1][:n]
        pred_sorted = np.sort(pred)[::-1][:n]
        obs_sum = obs_sorted.sum()
        if obs_sum == 0:
            return 0.0
        return float(100.0 * (pred_sorted.sum() - obs_sum) / obs_sum)

    @staticmethod
    def _compute_peak_timing(
        obs: np.ndarray, pred: np.ndarray, window: int = 365
    ) -> float:
        """MAE of annual peak occurrence timing (in days)."""
        n_windows = max(1, len(obs) // window)
        diffs = []
        for i in range(n_windows):
            start = i * window
            end = min(start + window, len(obs))
            o_slice = obs[start:end]
            p_slice = pred[start:end]
            if len(o_slice) == 0:
                continue
            obs_peak = np.argmax(o_slice)
            pred_peak = np.argmax(p_slice)
            diffs.append(abs(int(obs_peak) - int(pred_peak)))
        return float(np.mean(diffs)) if diffs else float("nan")

    @staticmethod
    def _compute_flood_detection(
        obs: np.ndarray, pred: np.ndarray, threshold: float
    ) -> tuple[float, float, float]:
        """Compute CSI, POD, FAR for flood event detection."""
        obs_flood = obs > threshold
        pred_flood = pred > threshold

        hits = int(np.sum(obs_flood & pred_flood))
        misses = int(np.sum(obs_flood & ~pred_flood))
        false_alarms = int(np.sum(~obs_flood & pred_flood))

        csi_denom = hits + misses + false_alarms
        csi = hits / csi_denom if csi_denom > 0 else 0.0

        pod_denom = hits + misses
        pod = hits / pod_denom if pod_denom > 0 else 0.0

        far_denom = hits + false_alarms
        far = false_alarms / far_denom if far_denom > 0 else 0.0

        return float(csi), float(pod), float(far)
