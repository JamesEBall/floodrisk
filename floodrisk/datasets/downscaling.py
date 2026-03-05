"""Precipitation downscaling dataset."""

from __future__ import annotations

from torch.utils.data import Dataset


class PrecipDownscalingDataset(Dataset):
    """Dataset for training precipitation downscaling models.

    Maps coarse-resolution NeuralGCM precipitation fields to high-resolution
    gridded precipitation observations.  Not yet implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PrecipDownscalingDataset is not yet implemented. "
            "See the project roadmap for planned support."
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
