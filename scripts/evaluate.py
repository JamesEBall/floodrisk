#!/usr/bin/env python
"""Evaluate a trained streamflow model on the test set.

Usage::

    python scripts/evaluate.py \
        --config configs/streamflow/lstm_camels.yaml \
        --checkpoint checkpoints/lstm_camels/best.pt \
        --output_dir outputs/eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from floodrisk.config import ExperimentConfig
from floodrisk.data.camels import CAMELSLoader
from floodrisk.datasets.streamflow import CatchmentDataset
from floodrisk.evaluation.evaluate import EvaluationPipeline
from floodrisk.evaluation.plots import plot_hydrograph, plot_nse_cdf, plot_scatter
from floodrisk.models import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a streamflow model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--output_dir", type=str, default="outputs/eval", help="Directory for evaluation outputs.")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load test data ---
    loader = CAMELSLoader(cfg.data.data_dir)
    basin_ids = loader.list_basins() if cfg.data.basins == "all" else cfg.data.basins
    logger.info(f"Evaluating on {len(basin_ids)} basins")

    test_forcing, test_streamflow = {}, {}
    for basin_id in basin_ids:
        test_forcing[basin_id] = loader.load_basin_forcing(
            basin_id, cfg.data.test_start, cfg.data.test_end
        )
        test_streamflow[basin_id] = loader.load_basin_streamflow(
            basin_id, cfg.data.test_start, cfg.data.test_end
        )

    test_dataset = CatchmentDataset(
        basins=basin_ids,
        forcing_data=test_forcing,
        streamflow_data=test_streamflow,
        seq_length=cfg.data.seq_length,
        forecast_horizon=cfg.data.forecast_horizon,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # --- Load model ---
    n_features = test_dataset[0][0].shape[-1] if len(test_dataset) > 0 else len(CAMELSLoader.FORCING_COLUMNS)
    model = build_model(
        cfg.model.type,
        n_features=n_features,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        forecast_horizon=cfg.data.forecast_horizon,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # --- Evaluate ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = EvaluationPipeline(model, test_dataset, device=device)
    results = pipeline.run()
    summary = pipeline.summary(results)

    logger.info(f"Summary: {summary}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # --- Generate plots ---
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # NSE CDF
    nse_values = [r["nse"] for r in results.values()]
    plot_nse_cdf(nse_values, save_path=plots_dir / "nse_cdf.png")
    logger.info(f"Saved NSE CDF plot")

    # Scatter plot (aggregate all basins)
    all_obs = np.concatenate([r["obs"] for r in results.values()])
    all_pred = np.concatenate([r["pred"] for r in results.values()])
    plot_scatter(all_obs, all_pred, save_path=plots_dir / "scatter.png")
    logger.info(f"Saved scatter plot")

    # Hydrographs for first 5 basins
    for i, (basin_id, res) in enumerate(results.items()):
        if i >= 5:
            break
        plot_hydrograph(res["obs"], res["pred"], basin_id, save_path=plots_dir / f"hydrograph_{basin_id}.png")

    logger.info(f"Evaluation complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
