"""Tests for loss functions and metrics."""

import pytest
import torch

from floodrisk.losses import NSELoss
from floodrisk.metrics.hydrology import KGEMetric, NSEMetric
from floodrisk.metrics.flood_event import CSIMetric, FARMetric, PODMetric


class TestNSELoss:
    def test_perfect_prediction(self):
        loss_fn = NSELoss()
        targets = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        preds = targets.clone()
        loss = loss_fn(preds, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_returns_scalar(self):
        loss_fn = NSELoss()
        preds = torch.randn(8, 1)
        targets = torch.randn(8, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = NSELoss()
        preds = torch.randn(8, 1, requires_grad=True)
        targets = torch.randn(8, 1)
        loss = loss_fn(preds, targets)
        loss.backward()
        assert preds.grad is not None

    def test_constant_target_no_nan(self):
        """Constant targets should not produce NaN (eps clamps denominator)."""
        loss_fn = NSELoss()
        preds = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([[5.0, 5.0, 5.0]])
        loss = loss_fn(preds, targets)
        assert not torch.isnan(loss)


class TestNSEMetric:
    def test_perfect_prediction(self):
        m = NSEMetric()
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        m.update(targets, targets)
        assert m.compute() == pytest.approx(1.0)

    def test_mean_prediction(self):
        m = NSEMetric()
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = torch.full_like(targets, targets.mean().item())
        m.update(preds, targets)
        assert m.compute() == pytest.approx(0.0, abs=1e-5)

    def test_reset(self):
        m = NSEMetric()
        m.update(torch.tensor([1.0]), torch.tensor([1.0]))
        m.reset()
        m.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
        assert m.compute() == pytest.approx(1.0)

    def test_multi_batch(self):
        m = NSEMetric()
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m.update(targets[:2], targets[:2])
        m.update(targets[2:], targets[2:])
        assert m.compute() == pytest.approx(1.0)

    def test_name(self):
        assert NSEMetric().name == "nse"


class TestKGEMetric:
    def test_perfect_prediction(self):
        m = KGEMetric()
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        m.update(targets, targets)
        assert m.compute() == pytest.approx(1.0)

    def test_name(self):
        assert KGEMetric().name == "kge"

    def test_reset(self):
        m = KGEMetric()
        m.update(torch.tensor([1.0]), torch.tensor([1.0]))
        m.reset()
        m.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
        assert m.compute() == pytest.approx(1.0)


class TestCSIMetric:
    def test_perfect_detection(self):
        m = CSIMetric(threshold=5.0)
        preds = torch.tensor([1.0, 6.0, 2.0, 8.0])
        targets = torch.tensor([1.0, 6.0, 2.0, 8.0])
        m.update(preds, targets)
        assert m.compute() == pytest.approx(1.0)

    def test_no_events(self):
        m = CSIMetric(threshold=10.0)
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        m.update(preds, targets)
        assert m.compute() == 0.0

    def test_name(self):
        assert CSIMetric(threshold=5.0).name == "csi"


class TestPODMetric:
    def test_all_detected(self):
        m = PODMetric(threshold=5.0)
        preds = torch.tensor([6.0, 7.0, 1.0])
        targets = torch.tensor([6.0, 7.0, 1.0])
        m.update(preds, targets)
        assert m.compute() == pytest.approx(1.0)

    def test_none_detected(self):
        m = PODMetric(threshold=5.0)
        preds = torch.tensor([1.0, 2.0])  # below threshold
        targets = torch.tensor([6.0, 7.0])  # above threshold
        m.update(preds, targets)
        assert m.compute() == pytest.approx(0.0)

    def test_name(self):
        assert PODMetric(threshold=5.0).name == "pod"


class TestFARMetric:
    def test_no_false_alarms(self):
        m = FARMetric(threshold=5.0)
        preds = torch.tensor([6.0, 7.0, 1.0])
        targets = torch.tensor([6.0, 7.0, 1.0])
        m.update(preds, targets)
        assert m.compute() == pytest.approx(0.0)

    def test_all_false_alarms(self):
        m = FARMetric(threshold=5.0)
        preds = torch.tensor([6.0, 7.0])  # above threshold
        targets = torch.tensor([1.0, 2.0])  # below threshold
        m.update(preds, targets)
        assert m.compute() == pytest.approx(1.0)

    def test_name(self):
        assert FARMetric(threshold=5.0).name == "far"
