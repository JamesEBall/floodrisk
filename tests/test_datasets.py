"""Tests for CatchmentDataset."""

import numpy as np
import pytest
import torch

from floodrisk.datasets.streamflow import CatchmentDataset

from .conftest import BASIN_IDS, N_FORCING_FEATURES


class TestCatchmentDataset:
    def test_len(self, forcing_data, streamflow_data):
        seq_len, horizon = 30, 1
        ds = CatchmentDataset(
            basins=BASIN_IDS,
            forcing_data=forcing_data,
            streamflow_data=streamflow_data,
            seq_length=seq_len,
            forecast_horizon=horizon,
        )
        n_days = len(next(iter(forcing_data.values())))
        expected_per_basin = n_days - seq_len - horizon + 1
        assert len(ds) == expected_per_basin * len(BASIN_IDS)

    def test_getitem_shapes(self, forcing_data, streamflow_data):
        seq_len, horizon = 30, 1
        ds = CatchmentDataset(
            basins=BASIN_IDS,
            forcing_data=forcing_data,
            streamflow_data=streamflow_data,
            seq_length=seq_len,
            forecast_horizon=horizon,
        )
        x, y = ds[0]
        assert x.shape == (seq_len, N_FORCING_FEATURES)
        assert y.shape == (horizon,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_getitem_with_static_attrs(
        self, forcing_data, streamflow_data, static_attributes
    ):
        seq_len, horizon = 30, 1
        n_static = 5
        ds = CatchmentDataset(
            basins=BASIN_IDS,
            forcing_data=forcing_data,
            streamflow_data=streamflow_data,
            static_attributes=static_attributes,
            seq_length=seq_len,
            forecast_horizon=horizon,
        )
        x, y = ds[0]
        assert x.shape == (seq_len, N_FORCING_FEATURES + n_static)

    def test_multi_step_horizon(self, forcing_data, streamflow_data):
        seq_len, horizon = 30, 7
        ds = CatchmentDataset(
            basins=BASIN_IDS,
            forcing_data=forcing_data,
            streamflow_data=streamflow_data,
            seq_length=seq_len,
            forecast_horizon=horizon,
        )
        x, y = ds[0]
        assert y.shape == (horizon,)

    def test_empty_basin_skipped(self, forcing_data, streamflow_data):
        """Basins with no valid windows are skipped."""
        seq_len = 9999  # longer than data
        ds = CatchmentDataset(
            basins=BASIN_IDS,
            forcing_data=forcing_data,
            streamflow_data=streamflow_data,
            seq_length=seq_len,
        )
        assert len(ds) == 0

    def test_dataloader_compatible(self, forcing_data, streamflow_data):
        """Dataset works with a PyTorch DataLoader."""
        ds = CatchmentDataset(
            basins=BASIN_IDS,
            forcing_data=forcing_data,
            streamflow_data=streamflow_data,
            seq_length=30,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] == 16
        assert batch_x.shape[1] == 30
