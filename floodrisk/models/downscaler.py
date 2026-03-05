"""CNN for downscaling NeuralGCM 2.8-degree precipitation to higher resolution."""

import torch.nn as nn


class PrecipDownscaler(nn.Module):
    """Downscale coarse NeuralGCM 2.8-degree precipitation fields to higher resolution.

    Uses a convolutional neural network to learn the mapping from coarse-resolution
    precipitation grids to fine-resolution grids, conditioned on high-resolution
    topographic and land-use features.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PrecipDownscaler is not yet implemented.")

    def forward(self, x):
        raise NotImplementedError("PrecipDownscaler is not yet implemented.")
