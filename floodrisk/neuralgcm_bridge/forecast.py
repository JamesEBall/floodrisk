"""NeuralGCM precipitation forecaster bridge.

Wraps NeuralGCM's PressureLevelModel to produce precipitation forecasts
that can be fed into the hydrological model pipeline.  All neuralgcm/JAX
imports are guarded so the rest of the package works without them.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import jax
    import neuralgcm
    from dinosaur import xarray_utils

    _HAS_NEURALGCM = True
except ImportError:
    _HAS_NEURALGCM = False

try:
    import xarray as xr
except ImportError:
    xr = None


def _require_neuralgcm() -> None:
    if not _HAS_NEURALGCM:
        raise ImportError(
            "neuralgcm, jax, and dinosaur are required for NeuralGCM forecasting. "
            "Install them with: pip install 'floodrisk[neuralgcm]'"
        )


class NeuralGCMForecaster:
    """Run NeuralGCM precipitation forecasts and cache results as NetCDF.

    Parameters
    ----------
    checkpoint_path : str
        Path to a NeuralGCM checkpoint file.
    output_dir : str
        Directory for saving forecast NetCDF files.
    """

    def __init__(self, checkpoint_path: str, output_dir: str = "cache/neuralgcm") -> None:
        _require_neuralgcm()
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint_path)

    def run_forecast(
        self,
        init_time: datetime | str,
        duration_hours: int = 240,
        output_freq_hours: int = 6,
    ) -> "xr.Dataset":
        """Run a precipitation forecast from the given initialization time.

        Parameters
        ----------
        init_time : datetime or str
            Initialization time for the forecast (ISO format if str).
        duration_hours : int
            Total forecast duration in hours.
        output_freq_hours : int
            Output frequency in hours.

        Returns
        -------
        xr.Dataset
            Dataset with precipitation fields on the NeuralGCM native grid.
        """
        _require_neuralgcm()

        if isinstance(init_time, str):
            init_time = datetime.fromisoformat(init_time)

        n_steps = duration_hours // output_freq_hours

        # Encode initial state
        state = self.model.encode(init_time)

        # Unroll forecast
        trajectories = self.model.unroll(state, steps=n_steps)

        # Convert to xarray
        ds = self.model.data_to_xarray(trajectories)

        # Save precipitation fields
        output_path = self._output_path(init_time)
        if "precipitation" in ds:
            precip_ds = ds[["precipitation"]]
        else:
            precip_ds = ds
        precip_ds.to_netcdf(output_path)

        return ds

    def _output_path(self, init_time: datetime) -> Path:
        """Build the output file path for a given init time."""
        ts = init_time.strftime("%Y%m%dT%H%M%S")
        return self.output_dir / f"neuralgcm_precip_{ts}.nc"
