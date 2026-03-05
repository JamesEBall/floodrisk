"""Transformer-based catchment model for streamflow forecasting."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
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

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CatchmentTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        forecast_horizon: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.fc = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        # x: [batch, seq_length, n_features]
        x = self.input_proj(x)  # [batch, seq_length, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [batch, seq_length, d_model]
        # Mean pooling over sequence dimension
        x = x.mean(dim=1)  # [batch, d_model]
        return self.fc(x)  # [batch, forecast_horizon]
