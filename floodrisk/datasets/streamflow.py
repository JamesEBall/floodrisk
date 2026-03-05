"""Streamflow catchment dataset for hydrological modeling."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CatchmentDataset(Dataset):
    """PyTorch Dataset for basin-level streamflow forecasting.

    Each sample is a (basin, time-window) pair yielding a sequence of forcing
    features (optionally concatenated with static catchment attributes) and a
    forecast-horizon target of streamflow values.

    Parameters
    ----------
    basins : list[str]
        Basin identifiers to include.
    forcing_data : dict[str, pd.DataFrame]
        Mapping of basin_id to a DataFrame of dynamic forcing variables,
        indexed by date.
    streamflow_data : dict[str, pd.Series]
        Mapping of basin_id to a Series of streamflow observations,
        indexed by date.
    static_attributes : dict[str, np.ndarray] | None
        Mapping of basin_id to a 1-D array of static catchment attributes.
    seq_length : int
        Number of timesteps in the input sequence.
    forecast_horizon : int
        Number of timesteps to predict.
    normalizer : object | None
        Optional normalizer with ``normalize_forcing(basin, data)`` and
        ``normalize_streamflow(basin, data)`` methods.
    """

    def __init__(
        self,
        basins: list[str],
        forcing_data: dict[str, pd.DataFrame],
        streamflow_data: dict[str, pd.Series],
        static_attributes: dict[str, np.ndarray] | None = None,
        seq_length: int = 365,
        forecast_horizon: int = 1,
        normalizer=None,
    ):
        self.forcing_data = forcing_data
        self.streamflow_data = streamflow_data
        self.static_attributes = static_attributes
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.normalizer = normalizer

        # Pre-compute aligned numpy arrays per basin and build sample index.
        self.samples: list[tuple[str, int]] = []
        self._forcing_arrays: dict[str, np.ndarray] = {}
        self._streamflow_arrays: dict[str, np.ndarray] = {}

        for basin in basins:
            forcing = forcing_data[basin]
            streamflow = streamflow_data[basin]

            # Align on the intersection of dates.
            common_idx = forcing.index.intersection(streamflow.index)
            common_idx = common_idx.sort_values()
            if len(common_idx) == 0:
                continue

            forcing_aligned = forcing.loc[common_idx]
            streamflow_aligned = streamflow.loc[common_idx]

            forcing_arr = forcing_aligned.values.astype(np.float32)
            streamflow_arr = streamflow_aligned.values.astype(np.float32)

            if normalizer is not None:
                forcing_arr = normalizer.normalize_forcing(basin, forcing_arr)
                streamflow_arr = normalizer.normalize_streamflow(basin, streamflow_arr)

            self._forcing_arrays[basin] = forcing_arr
            self._streamflow_arrays[basin] = streamflow_arr

            n_valid = len(common_idx) - seq_length - forecast_horizon + 1
            for i in range(max(0, n_valid)):
                self.samples.append((basin, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        basin, start = self.samples[idx]

        forcing_window = self._forcing_arrays[basin][start : start + self.seq_length]

        # Optionally concatenate static attributes (broadcast across time).
        if self.static_attributes is not None and basin in self.static_attributes:
            static = self.static_attributes[basin].astype(np.float32)
            static_broadcast = np.tile(static, (self.seq_length, 1))
            x = np.concatenate([forcing_window, static_broadcast], axis=1)
        else:
            x = forcing_window

        target_start = start + self.seq_length
        y = self._streamflow_arrays[basin][target_start : target_start + self.forecast_horizon]

        return torch.from_numpy(x), torch.from_numpy(y)
