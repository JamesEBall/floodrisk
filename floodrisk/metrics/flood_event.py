"""Flood-event verification metrics (CSI, POD, FAR).

All metrics classify timesteps as flood/no-flood by comparing values
against a user-specified threshold.  A "hit" means both prediction and
observation exceed the threshold.
"""

from __future__ import annotations

import torch

from floodrisk.torchharness import Metric


class CSIMetric(Metric):
    """Critical Success Index (Threat Score).

    CSI = hits / (hits + misses + false_alarms)
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._hits = 0
        self._misses = 0
        self._false_alarms = 0

    @property
    def name(self) -> str:
        return "csi"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        pred_flood = preds > self.threshold
        obs_flood = targets > self.threshold
        self._hits += (pred_flood & obs_flood).sum().item()
        self._misses += (~pred_flood & obs_flood).sum().item()
        self._false_alarms += (pred_flood & ~obs_flood).sum().item()

    def compute(self) -> float:
        denom = self._hits + self._misses + self._false_alarms
        if denom == 0:
            return 0.0
        return self._hits / denom

    def reset(self) -> None:
        self._hits = 0
        self._misses = 0
        self._false_alarms = 0


class PODMetric(Metric):
    """Probability of Detection (Hit Rate).

    POD = hits / (hits + misses)
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._hits = 0
        self._misses = 0

    @property
    def name(self) -> str:
        return "pod"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        pred_flood = preds > self.threshold
        obs_flood = targets > self.threshold
        self._hits += (pred_flood & obs_flood).sum().item()
        self._misses += (~pred_flood & obs_flood).sum().item()

    def compute(self) -> float:
        denom = self._hits + self._misses
        if denom == 0:
            return 0.0
        return self._hits / denom

    def reset(self) -> None:
        self._hits = 0
        self._misses = 0


class FARMetric(Metric):
    """False Alarm Ratio.

    FAR = false_alarms / (hits + false_alarms)
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._hits = 0
        self._false_alarms = 0

    @property
    def name(self) -> str:
        return "far"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        pred_flood = preds > self.threshold
        obs_flood = targets > self.threshold
        self._hits += (pred_flood & obs_flood).sum().item()
        self._false_alarms += (pred_flood & ~obs_flood).sum().item()

    def compute(self) -> float:
        denom = self._hits + self._false_alarms
        if denom == 0:
            return 0.0
        return self._false_alarms / denom

    def reset(self) -> None:
        self._hits = 0
        self._false_alarms = 0
