"""Static catchment attribute feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Named feature sets for catchment attributes.
FEATURE_SETS: dict[str, list[str]] = {
    "default": [
        "area",
        "mean_elev",
        "mean_slope",
        "frac_forest",
        "lai_max",
        "lai_diff",
        "gvf_max",
        "soil_depth",
        "soil_porosity",
        "max_water_content",
        "sand_frac",
        "clay_frac",
        "p_mean",
        "pet_mean",
        "aridity",
        "frac_snow",
        "high_prec_freq",
    ],
}


class CatchmentFeatures:
    """Utilities for selecting static catchment attributes."""

    @staticmethod
    def select_features(
        attributes: pd.Series,
        feature_set: str = "default",
    ) -> np.ndarray:
        """Select and return relevant static catchment attributes.

        Parameters
        ----------
        attributes : pd.Series
            Full set of catchment attributes keyed by attribute name.
        feature_set : str
            Name of a predefined feature set (see ``FEATURE_SETS``).

        Returns
        -------
        np.ndarray
            1-D float32 array of the selected attribute values, in the order
            defined by the feature set.

        Raises
        ------
        KeyError
            If *feature_set* is not a recognised set name.
        """
        if feature_set not in FEATURE_SETS:
            raise KeyError(
                f"Unknown feature set '{feature_set}'. "
                f"Available: {list(FEATURE_SETS.keys())}"
            )
        names = FEATURE_SETS[feature_set]
        return attributes[names].values.astype(np.float32)
