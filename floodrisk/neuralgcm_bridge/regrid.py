"""Regrid NeuralGCM precipitation to catchment-mean values.

Performs area-weighted averaging from the NeuralGCM native grid (~2.8 deg)
to catchment-mean precipitation time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def regrid_to_catchment(
    precip: xr.DataArray,
    catchment_bounds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Area-weighted average of gridded precipitation over catchment bounding boxes.

    Parameters
    ----------
    precip : xr.DataArray
        Precipitation on a regular lat/lon grid with dimensions
        ``(time, lat, lon)`` or ``(time, latitude, longitude)``.
    catchment_bounds : dict[str, dict]
        Mapping of basin_id to a dict with keys ``lat_min``, ``lat_max``,
        ``lon_min``, ``lon_max`` defining the catchment bounding box.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time with one column per basin containing
        catchment-mean precipitation.
    """
    # Normalise coordinate names
    lat_name = "lat" if "lat" in precip.dims else "latitude"
    lon_name = "lon" if "lon" in precip.dims else "longitude"

    lat = precip[lat_name].values
    lon = precip[lon_name].values

    results: dict[str, np.ndarray] = {}

    for basin_id, bounds in catchment_bounds.items():
        lat_mask = (lat >= bounds["lat_min"]) & (lat <= bounds["lat_max"])
        lon_mask = (lon >= bounds["lon_min"]) & (lon <= bounds["lon_max"])

        if not lat_mask.any() or not lon_mask.any():
            # No grid cells overlap this catchment
            results[basin_id] = np.full(precip.sizes["time"], np.nan)
            continue

        subset = precip.isel({lat_name: lat_mask, lon_name: lon_mask})

        # Area weights: cos(latitude)
        subset_lats = subset[lat_name]
        weights = np.cos(np.deg2rad(subset_lats))
        # Broadcast weights to (lat, lon)
        weights = weights / weights.sum()

        # Weighted mean over spatial dimensions
        basin_mean = subset.weighted(weights).mean(dim=[lat_name, lon_name])
        results[basin_id] = basin_mean.values

    time_index = pd.DatetimeIndex(precip["time"].values, name="time")
    return pd.DataFrame(results, index=time_index)
