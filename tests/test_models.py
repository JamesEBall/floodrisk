"""Tests for model forward passes."""

import pytest
import torch

from floodrisk.models.lstm import CatchmentLSTM
from floodrisk.models.transformer import CatchmentTransformer


class TestCatchmentLSTM:
    def test_output_shape(self):
        model = CatchmentLSTM(n_features=7, hidden_size=32, num_layers=1, forecast_horizon=1)
        x = torch.randn(4, 365, 7)
        out = model(x)
        assert out.shape == (4, 1)

    def test_multi_horizon(self):
        model = CatchmentLSTM(n_features=7, hidden_size=32, num_layers=1, forecast_horizon=7)
        x = torch.randn(4, 365, 7)
        out = model(x)
        assert out.shape == (4, 7)

    def test_variable_seq_length(self):
        model = CatchmentLSTM(n_features=7, hidden_size=32, num_layers=1)
        for seq_len in [30, 90, 365]:
            x = torch.randn(2, seq_len, 7)
            out = model(x)
            assert out.shape == (2, 1)

    def test_gradient_flows(self):
        model = CatchmentLSTM(n_features=7, hidden_size=32, num_layers=2, dropout=0.1)
        x = torch.randn(4, 30, 7)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_different_hidden_sizes(self):
        for hidden in [16, 64, 128]:
            model = CatchmentLSTM(n_features=7, hidden_size=hidden, num_layers=1)
            x = torch.randn(2, 30, 7)
            out = model(x)
            assert out.shape == (2, 1)


class TestCatchmentTransformer:
    def test_output_shape(self):
        model = CatchmentTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64, forecast_horizon=1
        )
        x = torch.randn(4, 365, 7)
        out = model(x)
        assert out.shape == (4, 1)

    def test_multi_horizon(self):
        model = CatchmentTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64, forecast_horizon=7
        )
        x = torch.randn(4, 365, 7)
        out = model(x)
        assert out.shape == (4, 7)

    def test_gradient_flows(self):
        model = CatchmentTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64
        )
        x = torch.randn(4, 30, 7)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_variable_seq_length(self):
        model = CatchmentTransformer(
            n_features=7, d_model=32, n_heads=2, n_layers=1, d_ff=64
        )
        for seq_len in [30, 90, 365]:
            x = torch.randn(2, seq_len, 7)
            out = model(x)
            assert out.shape == (2, 1)
