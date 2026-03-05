"""Model implementations."""

from floodrisk.models.lstm import CatchmentLSTM
from floodrisk.models.transformer import CatchmentTransformer

MODEL_REGISTRY = {
    "lstm": CatchmentLSTM,
    "transformer": CatchmentTransformer,
}


def build_model(model_type: str, **kwargs):
    """Build a model by type name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](**kwargs)
