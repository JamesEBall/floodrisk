"""Model implementations."""

MODEL_REGISTRY = {
    "lstm": "floodrisk.models.lstm.CatchmentLSTM",
    "transformer": "floodrisk.models.transformer.CatchmentTransformer",
    "fgn_lstm": "floodrisk.models.fgn_streamflow.FGNStreamflowLSTM",
    "fgn_transformer": "floodrisk.models.fgn_streamflow.FGNStreamflowTransformer",
}


def build_model(model_type: str, **kwargs):
    """Build a model by type name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY)}")
    module_path, class_name = MODEL_REGISTRY[model_type].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
