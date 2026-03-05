"""FGN-style probabilistic streamflow models.

Adapts the Functional Generative Network (FGN) approach from WeatherNext 2 for
streamflow forecasting.  All stochasticity comes from a single low-dimensional
noise vector z ~ N(0, I) injected via conditional LayerNorm layers, allowing a
single model to generate diverse ensemble forecasts in one forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConditionalLayerNorm(nn.Module):
    """LayerNorm with affine parameters generated from a noise vector.

    Given input ``x`` and noise ``z``:
        x_norm = LayerNorm(x)
        gamma, beta = split(Linear(z), 2)
        output = (1 + gamma) * x_norm + beta
    """

    def __init__(self, hidden_size: int, noise_dim: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.noise_proj = nn.Linear(noise_dim, 2 * hidden_size)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        # z: [batch, noise_dim] -> [batch, 2 * hidden_size]
        proj = self.noise_proj(z)
        gamma, beta = proj.chunk(2, dim=-1)
        # Broadcast over sequence dimension if x is 3-D [batch, seq, hidden]
        if x_norm.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return (1 + gamma) * x_norm + beta


# ---------------------------------------------------------------------------
# FGN-LSTM
# ---------------------------------------------------------------------------

class FGNStreamflowLSTM(nn.Module):
    """LSTM backbone with conditional LayerNorm noise injection.

    Parameters match :class:`CatchmentLSTM` plus ``noise_dim`` and
    ``n_ensemble`` for the number of ensemble members sampled during training.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
        noise_dim: int = 32,
        n_ensemble: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.noise_dim = noise_dim
        self.n_ensemble = n_ensemble

        self.input_proj = nn.Linear(n_features, hidden_size)
        self.cond_norm_pre = ConditionalLayerNorm(hidden_size, noise_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.cond_norm_post = ConditionalLayerNorm(hidden_size, noise_dim)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Single forward pass with one noise vector.

        Parameters
        ----------
        x : Tensor [batch, seq_length, n_features]
        z : Tensor [batch, noise_dim] or None (auto-sampled)

        Returns
        -------
        Tensor [batch, forecast_horizon]
        """
        if z is None:
            z = torch.randn(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype)

        h = self.input_proj(x)  # [B, T, H]
        h = self.cond_norm_pre(h, z)  # [B, T, H]
        lstm_out, _ = self.lstm(h)  # [B, T, H]
        last_hidden = lstm_out[:, -1, :]  # [B, H]
        last_hidden = self.cond_norm_post(last_hidden, z)  # [B, H]
        return self.fc(last_hidden)  # [B, forecast_horizon]

    def ensemble_forward(
        self, x: torch.Tensor, n_ensemble: int | None = None
    ) -> torch.Tensor:
        """Generate an ensemble of predictions.

        Parameters
        ----------
        x : Tensor [batch, seq_length, n_features]
        n_ensemble : int, optional
            Number of ensemble members (defaults to ``self.n_ensemble``).

        Returns
        -------
        Tensor [batch, n_ensemble, forecast_horizon]
        """
        n = n_ensemble or self.n_ensemble
        members = []
        for _ in range(n):
            z = torch.randn(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype)
            members.append(self.forward(x, z))
        return torch.stack(members, dim=1)  # [B, N, H]


# ---------------------------------------------------------------------------
# FGN-Transformer helpers
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (mirrors transformer.py)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class _ConditionalTransformerLayer(nn.Module):
    """Transformer encoder layer with conditional LayerNorm replacing standard LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        noise_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.cond_norm1 = ConditionalLayerNorm(d_model, noise_dim)
        self.cond_norm2 = ConditionalLayerNorm(d_model, noise_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.cond_norm1(x, z)
        # Feed-forward + residual
        x = x + self.ff(x)
        x = self.cond_norm2(x, z)
        return x


# ---------------------------------------------------------------------------
# FGN-Transformer
# ---------------------------------------------------------------------------

class FGNStreamflowTransformer(nn.Module):
    """Transformer backbone with conditional LayerNorm noise injection.

    Parameters match :class:`CatchmentTransformer` plus ``noise_dim`` and
    ``n_ensemble``.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        forecast_horizon: int = 1,
        noise_dim: int = 32,
        n_ensemble: int = 2,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.n_ensemble = n_ensemble

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                _ConditionalTransformerLayer(d_model, n_heads, d_ff, noise_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.fc = nn.Linear(d_model, forecast_horizon)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Single forward pass.

        Parameters
        ----------
        x : Tensor [batch, seq_length, n_features]
        z : Tensor [batch, noise_dim] or None

        Returns
        -------
        Tensor [batch, forecast_horizon]
        """
        if z is None:
            z = torch.randn(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype)

        h = self.input_proj(x)
        h = self.pos_encoder(h)
        for layer in self.layers:
            h = layer(h, z)
        h = h.mean(dim=1)  # mean pool over sequence
        return self.fc(h)

    def ensemble_forward(
        self, x: torch.Tensor, n_ensemble: int | None = None
    ) -> torch.Tensor:
        """Generate an ensemble of predictions.

        Returns
        -------
        Tensor [batch, n_ensemble, forecast_horizon]
        """
        n = n_ensemble or self.n_ensemble
        members = []
        for _ in range(n):
            z = torch.randn(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype)
            members.append(self.forward(x, z))
        return torch.stack(members, dim=1)
