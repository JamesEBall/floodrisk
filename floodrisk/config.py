"""Experiment configuration for FloodRisk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from floodrisk.torchharness import TrainerConfig


@dataclass
class DataConfig:
    dataset: str = "camels"
    data_dir: str = "data/camels"
    basins: list | str = "all"
    seq_length: int = 365
    forecast_horizon: int = 1
    train_start: str = "1980-10-01"
    train_end: str = "1995-09-30"
    val_start: str = "1995-10-01"
    val_end: str = "2000-09-30"
    test_start: str = "2000-10-01"
    test_end: str = "2014-09-30"


@dataclass
class ModelConfig:
    type: str = "lstm"
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class ExperimentConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        trainer = TrainerConfig(**raw.get("trainer", {}))
        data = DataConfig(**raw.get("data", {}))
        model = ModelConfig(**raw.get("model", {}))
        return cls(
            trainer=trainer,
            data=data,
            model=model,
            output_dir=raw.get("output_dir", "outputs"),
            cache_dir=raw.get("cache_dir", "cache"),
            seed=raw.get("seed", 42),
        )
