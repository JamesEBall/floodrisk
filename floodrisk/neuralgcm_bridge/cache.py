"""Cache manager for NeuralGCM forecast NetCDF files."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


class CacheManager:
    """Manage cached NeuralGCM forecast files.

    Parameters
    ----------
    cache_dir : str
        Directory containing cached forecast NetCDF files.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_forecasts(self) -> list[str]:
        """List cached forecast file names.

        Returns
        -------
        list[str]
            Sorted list of NetCDF file names in the cache directory.
        """
        return sorted(f.name for f in self.cache_dir.glob("*.nc"))

    def load_forecast(self, init_time: str) -> xr.Dataset:
        """Load a cached forecast by init time string.

        Parameters
        ----------
        init_time : str
            Init time formatted as ``YYYYMMDDTHHmmss``, matching the file
            naming convention from :class:`NeuralGCMForecaster`.

        Returns
        -------
        xr.Dataset
            The forecast dataset.

        Raises
        ------
        FileNotFoundError
            If no cached file exists for the given init time.
        """
        path = self.cache_dir / f"neuralgcm_precip_{init_time}.nc"
        if not path.exists():
            raise FileNotFoundError(f"No cached forecast for init_time={init_time}: {path}")
        return xr.open_dataset(path)

    def invalidate(self, init_time: str | None = None) -> None:
        """Delete cached forecast files.

        Parameters
        ----------
        init_time : str or None
            If provided, delete only the file matching this init time.
            If ``None``, delete all cached forecast files.
        """
        if init_time is not None:
            path = self.cache_dir / f"neuralgcm_precip_{init_time}.nc"
            if path.exists():
                path.unlink()
        else:
            for path in self.cache_dir.glob("*.nc"):
                path.unlink()
