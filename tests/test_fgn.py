"""Tests for FGN probabilistic streamflow models, CRPS loss, and ensemble metrics."""

import pytest
import torch

from floodrisk.losses import CRPSLoss
from floodrisk.metrics.ensemble import CRPSMetric, EnsembleNSEMetric, SpreadSkillMetric
from floodrisk.models.fgn_streamflow import (
    ConditionalLayerNorm,
    FGNStreamflowLSTM,
    FGNStreamflowTransformer,
)


# ---------------------------------------------------------------------------
# ConditionalLayerNorm
# ---------------------------------------------------------------------------


class TestConditionalLayerNorm:
    def test_output_shape_2d(self):
        cln = ConditionalLayerNorm(hidden_size=64, noise_dim=16)
        x = torch.randn(4, 64)
        z = torch.randn(4, 16)
        out = cln(x, z)
        assert out.shape == (4, 64)

    def test_output_shape_3d(self):
        cln = ConditionalLayerNorm(hidden_size=64, noise_dim=16)
        x = torch.randn(4, 30, 64)
        z = torch.randn(4, 16)
        out = cln(x, z)
        assert out.shape == (4, 30, 64)

    def test_gradient_flows(self):
        cln = ConditionalLayerNorm(hidden_size=32, noise_dim=8)
        x = torch.randn(2, 10, 32, requires_grad=True)
        z = torch.randn(2, 8, requires_grad=True)
        out = cln(x, z)
        out.sum().backward()
        assert x.grad is not None
        assert z.grad is not None
        for p in cln.parameters():
            assert p.grad is not None

    def test_different_noise_different_output(self):
        cln = ConditionalLayerNorm(hidden_size=32, noise_dim=8)
        x = torch.randn(2, 32)
        z1 = torch.randn(2, 8)
        z2 = torch.randn(2, 8)
        out1 = cln(x, z1)
        out2 = cln(x, z2)
        assert not torch.allclose(out1, out2), "Different noise should yield different outputs"


# ---------------------------------------------------------------------------
# FGNStreamflowLSTM
# ---------------------------------------------------------------------------


class TestFGNStreamflowLSTM:
    def test_forward_shape(self):
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=1, forecast_horizon=1, noise_dim=8
        )
        x = torch.randn(4, 30, 7)
        out = model(x)
        assert out.shape == (4, 1)

    def test_forward_multi_horizon(self):
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=1, forecast_horizon=7, noise_dim=8
        )
        x = torch.randn(4, 30, 7)
        out = model(x)
        assert out.shape == (4, 7)

    def test_ensemble_forward_shape(self):
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=1, forecast_horizon=1,
            noise_dim=8, n_ensemble=5,
        )
        x = torch.randn(4, 30, 7)
        ens = model.ensemble_forward(x)
        assert ens.shape == (4, 5, 1)

    def test_ensemble_forward_custom_n(self):
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=1, forecast_horizon=1, noise_dim=8
        )
        x = torch.randn(4, 30, 7)
        ens = model.ensemble_forward(x, n_ensemble=10)
        assert ens.shape == (4, 10, 1)

    def test_stochasticity(self):
        """Different calls with auto-sampled noise should produce different outputs."""
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=1, noise_dim=8
        )
        model.eval()
        x = torch.randn(2, 30, 7)
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2), "Auto-sampled noise should give different outputs"

    def test_same_noise_deterministic(self):
        """Same noise vector should produce identical output."""
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=1, noise_dim=8
        )
        model.eval()
        x = torch.randn(2, 30, 7)
        z = torch.randn(2, 8)
        out1 = model(x, z)
        out2 = model(x, z)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self):
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=2, dropout=0.1, noise_dim=8
        )
        x = torch.randn(4, 30, 7)
        ens = model.ensemble_forward(x, n_ensemble=2)
        loss = ens.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_variable_seq_length(self):
        model = FGNStreamflowLSTM(
            n_features=7, hidden_size=32, num_layers=1, noise_dim=8
        )
        for seq_len in [10, 30, 365]:
            x = torch.randn(2, seq_len, 7)
            out = model(x)
            assert out.shape == (2, 1)


# ---------------------------------------------------------------------------
# FGNStreamflowTransformer
# ---------------------------------------------------------------------------


class TestFGNStreamflowTransformer:
    def test_forward_shape(self):
        model = FGNStreamflowTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64,
            forecast_horizon=1, noise_dim=8,
        )
        x = torch.randn(4, 30, 7)
        out = model(x)
        assert out.shape == (4, 1)

    def test_forward_multi_horizon(self):
        model = FGNStreamflowTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64,
            forecast_horizon=7, noise_dim=8,
        )
        x = torch.randn(4, 30, 7)
        out = model(x)
        assert out.shape == (4, 7)

    def test_ensemble_forward_shape(self):
        model = FGNStreamflowTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64,
            noise_dim=8, n_ensemble=5,
        )
        x = torch.randn(4, 30, 7)
        ens = model.ensemble_forward(x)
        assert ens.shape == (4, 5, 1)

    def test_stochasticity(self):
        model = FGNStreamflowTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64, noise_dim=8,
        )
        model.eval()
        x = torch.randn(2, 30, 7)
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self):
        model = FGNStreamflowTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64, noise_dim=8,
        )
        x = torch.randn(4, 30, 7)
        ens = model.ensemble_forward(x, n_ensemble=2)
        loss = ens.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_variable_seq_length(self):
        model = FGNStreamflowTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64, noise_dim=8,
        )
        for seq_len in [10, 30, 365]:
            x = torch.randn(2, seq_len, 7)
            out = model(x)
            assert out.shape == (2, 1)


# ---------------------------------------------------------------------------
# CRPSLoss
# ---------------------------------------------------------------------------


class TestCRPSLoss:
    def test_output_is_scalar(self):
        loss_fn = CRPSLoss()
        preds = torch.randn(4, 5, 1)
        targets = torch.randn(4, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0

    def test_perfect_ensemble(self):
        """Ensemble collapsed onto the target should give CRPS near 0."""
        loss_fn = CRPSLoss()
        targets = torch.randn(4, 1)
        # All ensemble members = target
        preds = targets.unsqueeze(1).expand(-1, 10, -1)
        loss = loss_fn(preds, targets)
        assert loss.item() < 1e-6

    def test_spread_ensemble_positive(self):
        """Wider spread away from target should give positive CRPS."""
        loss_fn = CRPSLoss()
        targets = torch.zeros(4, 1)
        preds = torch.randn(4, 10, 1) * 5  # large spread, centered near 0
        loss = loss_fn(preds, targets)
        assert loss.item() > 0

    def test_gradient_flows(self):
        loss_fn = CRPSLoss()
        preds = torch.randn(4, 5, 1, requires_grad=True)
        targets = torch.randn(4, 1)
        loss = loss_fn(preds, targets)
        loss.backward()
        assert preds.grad is not None

    def test_single_member_equals_mae(self):
        """With N=1, CRPS should equal MAE (spread term is 0)."""
        loss_fn = CRPSLoss()
        preds = torch.randn(8, 1, 1)
        targets = torch.randn(8, 1)
        crps = loss_fn(preds, targets)
        mae = (preds.squeeze(1) - targets).abs().mean()
        assert torch.allclose(crps, mae, atol=1e-6)


# ---------------------------------------------------------------------------
# Ensemble Metrics
# ---------------------------------------------------------------------------


class TestCRPSMetric:
    def test_basic(self):
        metric = CRPSMetric()
        preds = torch.randn(8, 10, 1)
        targets = torch.randn(8, 1)
        metric.update(preds, targets)
        val = metric.compute()
        assert isinstance(val, float)

    def test_perfect_prediction(self):
        metric = CRPSMetric()
        targets = torch.randn(8, 1)
        preds = targets.unsqueeze(1).expand(-1, 10, -1)
        metric.update(preds, targets)
        val = metric.compute()
        assert abs(val) < 1e-6

    def test_reset(self):
        metric = CRPSMetric()
        metric.update(torch.randn(4, 5, 1), torch.randn(4, 1))
        metric.reset()
        assert metric.compute() != metric.compute() or True  # nan check
        metric.reset()
        import math
        assert math.isnan(metric.compute())

    def test_deterministic_fallback(self):
        metric = CRPSMetric()
        preds = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        metric.update(preds, targets)
        val = metric.compute()
        assert isinstance(val, float)


class TestSpreadSkillMetric:
    def test_basic(self):
        metric = SpreadSkillMetric()
        preds = torch.randn(8, 10, 1)
        targets = torch.randn(8, 1)
        metric.update(preds, targets)
        val = metric.compute()
        assert isinstance(val, float)

    def test_skips_deterministic(self):
        metric = SpreadSkillMetric()
        metric.update(torch.randn(4, 1), torch.randn(4, 1))
        import math
        assert math.isnan(metric.compute())


class TestEnsembleNSEMetric:
    def test_perfect_ensemble_mean(self):
        metric = EnsembleNSEMetric()
        targets = torch.arange(1.0, 11.0).unsqueeze(1)  # [10, 1]
        # Ensemble: mean = targets
        noise = torch.randn(10, 5, 1) * 0.01
        preds = targets.unsqueeze(1) + noise  # [10, 5, 1]
        metric.update(preds, targets)
        val = metric.compute()
        assert val > 0.99

    def test_accepts_deterministic(self):
        metric = EnsembleNSEMetric()
        targets = torch.arange(1.0, 11.0).unsqueeze(1)
        preds = targets.clone()
        metric.update(preds, targets)
        val = metric.compute()
        assert val > 0.99

    def test_multi_batch(self):
        metric = EnsembleNSEMetric()
        for _ in range(3):
            targets = torch.arange(1.0, 11.0).unsqueeze(1)
            preds = targets.unsqueeze(1).expand(-1, 5, -1)
            metric.update(preds, targets)
        val = metric.compute()
        assert val > 0.99


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_fgn_lstm_in_registry(self):
        from floodrisk.models import build_model
        model = build_model(
            "fgn_lstm", n_features=7, hidden_size=32, num_layers=1,
            noise_dim=8, forecast_horizon=1,
        )
        assert hasattr(model, "ensemble_forward")
        x = torch.randn(2, 30, 7)
        out = model(x)
        assert out.shape == (2, 1)

    def test_fgn_transformer_in_registry(self):
        from floodrisk.models import build_model
        model = build_model(
            "fgn_transformer", n_features=7, d_model=32, n_heads=2,
            n_layers=1, d_ff=64, noise_dim=8, forecast_horizon=1,
        )
        assert hasattr(model, "ensemble_forward")
        x = torch.randn(2, 30, 7)
        out = model(x)
        assert out.shape == (2, 1)
