#!/usr/bin/env python
"""Train a streamflow forecasting model.

Usage::

    python scripts/train_streamflow.py --config configs/streamflow/lstm_camels.yaml
    python scripts/train_streamflow.py --config configs/streamflow/lstm_camels.yaml --device cpu
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floodrisk.callbacks.hydro_logger import HydroLogger
from floodrisk.config import ExperimentConfig
from floodrisk.data.camels import CAMELSLoader
from floodrisk.data.caravan import CaravanLoader
from floodrisk.data.normalization import BasinNormalizer
from floodrisk.datasets.streamflow import CatchmentDataset
from floodrisk.losses import CRPSLoss, NSELoss
from floodrisk.metrics.ensemble import CRPSMetric, EnsembleNSEMetric, SpreadSkillMetric
from floodrisk.metrics.hydrology import KGEMetric, NSEMetric
from floodrisk.models import build_model
from floodrisk.torchharness import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a streamflow model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="floodrisk", help="W&B project name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name.")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.device:
        cfg.trainer.device = args.device

    set_seed(cfg.seed)
    logger.info(f"Seed: {cfg.seed} | Device: {cfg.trainer.device}")

    # --- Data loading ---
    if cfg.data.dataset == "caravan":
        loader = CaravanLoader(cfg.data.data_dir)
    else:
        loader = CAMELSLoader(cfg.data.data_dir)
    basin_ids = loader.list_basins() if cfg.data.basins == "all" else cfg.data.basins
    logger.info(f"Loading data for {len(basin_ids)} basins")

    train_forcing, train_streamflow = {}, {}
    val_forcing, val_streamflow = {}, {}

    for basin_id in basin_ids:
        # Training period
        train_forcing[basin_id] = loader.load_basin_forcing(
            basin_id, cfg.data.train_start, cfg.data.train_end
        )
        train_streamflow[basin_id] = loader.load_basin_streamflow(
            basin_id, cfg.data.train_start, cfg.data.train_end
        )
        # Validation period
        val_forcing[basin_id] = loader.load_basin_forcing(
            basin_id, cfg.data.val_start, cfg.data.val_end
        )
        val_streamflow[basin_id] = loader.load_basin_streamflow(
            basin_id, cfg.data.val_start, cfg.data.val_end
        )

    # --- Normalizer ---
    normalizer = BasinNormalizer()
    normalizer.fit(train_forcing)
    normalizer.fit_streamflow(train_streamflow)
    logger.info("Fitted normalizer on training forcing + streamflow data")

    # Apply normalization to forcing data
    for basin_id in basin_ids:
        train_forcing[basin_id] = normalizer.transform(train_forcing[basin_id])
        val_forcing[basin_id] = normalizer.transform(val_forcing[basin_id])

    # Apply normalization to streamflow data
    for basin_id in basin_ids:
        train_streamflow[basin_id] = normalizer.transform_streamflow(train_streamflow[basin_id])
        val_streamflow[basin_id] = normalizer.transform_streamflow(val_streamflow[basin_id])

    # Save normalizer for later use
    normalizer.save(str(Path(cfg.output_dir) / "normalizer.json"))

    # --- Datasets ---
    train_dataset = CatchmentDataset(
        basins=basin_ids,
        forcing_data=train_forcing,
        streamflow_data=train_streamflow,
        seq_length=cfg.data.seq_length,
        forecast_horizon=cfg.data.forecast_horizon,
    )
    val_dataset = CatchmentDataset(
        basins=basin_ids,
        forcing_data=val_forcing,
        streamflow_data=val_streamflow,
        seq_length=cfg.data.seq_length,
        forecast_horizon=cfg.data.forecast_horizon,
    )
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False
    )

    # --- Model ---
    n_features = train_dataset[0][0].shape[-1] if len(train_dataset) > 0 else len(CAMELSLoader.FORCING_COLUMNS)
    model_kwargs = dict(
        n_features=n_features,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        forecast_horizon=cfg.data.forecast_horizon,
    )
    # Pass FGN-specific params for ensemble models
    is_fgn = cfg.model.type.startswith("fgn_")
    if is_fgn:
        model_kwargs["noise_dim"] = cfg.model.noise_dim
        model_kwargs["n_ensemble"] = cfg.model.n_ensemble
    model = build_model(cfg.model.type, **model_kwargs)
    logger.info(f"Model: {cfg.model.type} | params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training ---
    if is_fgn:
        loss_fn = CRPSLoss()
        metrics = [CRPSMetric(), EnsembleNSEMetric(), SpreadSkillMetric()]
    else:
        loss_fn = NSELoss()
        metrics = [NSEMetric(), KGEMetric()]
    callbacks = [HydroLogger(output_dir=cfg.output_dir)]

    if args.wandb:
        from dataclasses import asdict

        from floodrisk.callbacks.wandb_logger import WandbLogger

        wandb_config = {
            "trainer": asdict(cfg.trainer),
            "data": asdict(cfg.data),
            "model": asdict(cfg.model),
            "seed": cfg.seed,
            "n_basins": len(basin_ids),
            "n_train_samples": len(train_dataset),
            "n_val_samples": len(val_dataset),
            "n_params": sum(p.numel() for p in model.parameters()),
        }
        callbacks.append(
            WandbLogger(
                project=args.wandb_project,
                run_name=args.wandb_name,
                config=wandb_config,
            )
        )

    trainer = Trainer(
        model=model,
        config=cfg.trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics=metrics,
        callbacks=callbacks,
    )

    final_metrics = trainer.fit()
    logger.info(f"Training complete. Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
