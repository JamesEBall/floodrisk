"""Temporal feature engineering for forcing data."""

from __future__ import annotations

import math

import pandas as pd


class TemporalFeatures:
    """Utilities for adding time-derived features to forcing DataFrames."""

    @staticmethod
    def add_day_of_year(df: pd.DataFrame) -> pd.DataFrame:
        """Add sine/cosine encoded day-of-year columns.

        Adds ``sin_doy`` and ``cos_doy`` columns computed as cyclic encodings
        of the day of year: ``sin(2 * pi * doy / 365)`` and
        ``cos(2 * pi * doy / 365)``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``sin_doy`` and ``cos_doy`` appended.
        """
        df = df.copy()
        doy = df.index.dayofyear
        df["sin_doy"] = [math.sin(2 * math.pi * d / 365) for d in doy]
        df["cos_doy"] = [math.cos(2 * math.pi * d / 365) for d in doy]
        return df

    @staticmethod
    def add_lag_features(
        df: pd.DataFrame,
        columns: list[str],
        lags: list[int],
    ) -> pd.DataFrame:
        """Create lagged versions of specified columns.

        For each column and each lag value, a new column named
        ``{column}_lag{lag}`` is added containing the value shifted by *lag*
        timesteps.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame.
        columns : list[str]
            Column names to lag.
        lags : list[int]
            Positive integers specifying lag amounts.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with lagged columns appended.
        """
        df = df.copy()
        for col in columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df
