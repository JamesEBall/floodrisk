"""LSTM-based catchment model for streamflow forecasting."""

import torch
import torch.nn as nn


class CatchmentLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x: [batch, seq_length, n_features]
        # lstm_out: [batch, seq_length, hidden_size]
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]
        return self.fc(last_hidden)  # [batch, forecast_horizon]
