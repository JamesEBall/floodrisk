"""CNN for predicting flood inundation maps from discharge and DEM data."""

import torch.nn as nn


class FloodInundationCNN(nn.Module):
    """Predict flood inundation maps from river discharge and digital elevation model (DEM).

    Takes discharge forecasts and DEM tiles as input and produces spatial
    inundation depth/extent maps as output.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FloodInundationCNN is not yet implemented.")

    def forward(self, x):
        raise NotImplementedError("FloodInundationCNN is not yet implemented.")
