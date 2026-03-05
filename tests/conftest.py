"""Shared test fixtures for FloodRisk."""

import numpy as np
import pandas as pd
import pytest
import torch

from floodrisk.config import DataConfig, ExperimentConfig, ModelConfig
from floodrisk.torchharness import TrainerConfig


BASIN_IDS = ["01013500", "01022500", "01030500"]
N_DAYS = 500
N_FORCING_FEATURES = 7
FORCING_COLS = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]


@pytest.fixture
def date_index():
    return pd.date_range("1990-01-01", periods=N_DAYS, freq="D")


@pytest.fixture
def forcing_data(date_index):
    """Fake forcing data for 3 basins."""
    rng = np.random.default_rng(42)
    return {
        basin: pd.DataFrame(
            rng.standard_normal((N_DAYS, N_FORCING_FEATURES)).astype(np.float32),
            index=date_index,
            columns=FORCING_COLS,
        )
        for basin in BASIN_IDS
    }


@pytest.fixture
def streamflow_data(date_index):
    """Fake streamflow data for 3 basins."""
    rng = np.random.default_rng(99)
    return {
        basin: pd.Series(
            np.abs(rng.standard_normal(N_DAYS).astype(np.float32)) * 10,
            index=date_index,
            name="streamflow",
        )
        for basin in BASIN_IDS
    }


@pytest.fixture
def static_attributes():
    """Fake static attributes for 3 basins (5 features each)."""
    rng = np.random.default_rng(7)
    return {basin: rng.standard_normal(5).astype(np.float32) for basin in BASIN_IDS}


@pytest.fixture
def sample_config():
    return ExperimentConfig(
        trainer=TrainerConfig(lr=1e-3, epochs=2, batch_size=32, device="cpu"),
        data=DataConfig(
            dataset="camels",
            data_dir="data/camels",
            basins=BASIN_IDS,
            seq_length=30,
            forecast_horizon=1,
        ),
        model=ModelConfig(type="lstm", hidden_size=32, num_layers=1, dropout=0.0),
    )
