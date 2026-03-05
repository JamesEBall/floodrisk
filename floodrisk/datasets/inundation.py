"""Flood inundation mapping dataset."""

from __future__ import annotations

from torch.utils.data import Dataset


class FloodInundationDataset(Dataset):
    """Dataset for flood inundation extent prediction.

    Pairs streamflow forecasts and DEM-derived features with observed
    inundation extents for training spatial flood mapping models.
    Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FloodInundationDataset is not yet implemented. "
            "See the project roadmap for planned support."
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
