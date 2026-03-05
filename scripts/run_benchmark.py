#!/usr/bin/env python
"""Run the FloodCastBench benchmark suite.

Usage::

    python scripts/run_benchmark.py --config configs/benchmark/standard_camels.yaml
    python scripts/run_benchmark.py --config configs/benchmark/standard_camels.yaml \\
        --checkpoint outputs/lstm_camels/best.pt --model-type lstm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from floodrisk.benchmark.baselines import ClimatologyBaseline, PersistenceBaseline
from floodrisk.benchmark.protocol import BenchmarkProtocol, BenchmarkResults
from floodrisk.benchmark.report import generate_report
from floodrisk.data.camels import CAMELSLoader
from floodrisk.data.caravan import CaravanLoader
from floodrisk.data.normalization import BasinNormalizer
from floodrisk.datasets.streamflow import CatchmentDataset
from floodrisk.models import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_benchmark_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model_on_basins(
    model: torch.nn.Module,
    loader,
    basin_ids: list[str],
    test_start: str,
    test_end: str,
    normalizer: BasinNormalizer | None,
    seq_length: int = 365,
    forecast_horizon: int = 1,
    device: str = "cpu",
    n_ensemble: int = 50,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Run inference per-basin and return obs/pred dicts.

    For ensemble models (those with ``ensemble_forward``), also returns a dict
    mapping basin_id to the full ensemble array ``[n_timesteps, n_ensemble]``.
    """
    model.to(device)
    model.eval()

    _is_ensemble = hasattr(model, "ensemble_forward")

    obs_dict: dict[str, np.ndarray] = {}
    pred_dict: dict[str, np.ndarray] = {}
    ensemble_dict: dict[str, np.ndarray] | None = {} if _is_ensemble else None

    for basin_id in basin_ids:
        try:
            forcing = loader.load_basin_forcing(basin_id, test_start, test_end)
            streamflow = loader.load_basin_streamflow(basin_id, test_start, test_end)
        except (FileNotFoundError, KeyError):
            continue

        if len(forcing) < seq_length + forecast_horizon:
            continue

        # Normalize forcing
        if normalizer is not None:
            forcing_norm = normalizer.transform(forcing)
        else:
            forcing_norm = forcing

        forcing_arr = forcing_norm.values.astype(np.float32)
        streamflow_arr = streamflow.values.astype(np.float64)

        preds = []
        ens_preds = []
        obs_vals = []
        n_windows = len(forcing_arr) - seq_length - forecast_horizon + 1

        with torch.no_grad():
            for i in range(n_windows):
                x = forcing_arr[i : i + seq_length]
                target_start = i + seq_length
                y = streamflow_arr[target_start : target_start + forecast_horizon]

                if np.any(np.isnan(y)) or np.any(np.isnan(x)):
                    continue

                x_t = torch.from_numpy(x).unsqueeze(0).to(device)

                if _is_ensemble:
                    ens = model.ensemble_forward(x_t, n_ensemble=n_ensemble)  # [1, N, H]
                    ens_np = ens.cpu().numpy().squeeze(0)  # [N, H]
                    p = ens_np.mean(axis=0)  # [H]
                    if normalizer is not None and normalizer.streamflow_mean is not None:
                        ens_np = ens_np * normalizer.streamflow_std + normalizer.streamflow_mean
                        p = p * normalizer.streamflow_std + normalizer.streamflow_mean
                    ens_preds.append(ens_np)
                else:
                    p = model(x_t).cpu().numpy().flatten()
                    if normalizer is not None and normalizer.streamflow_mean is not None:
                        p = p * normalizer.streamflow_std + normalizer.streamflow_mean

                preds.append(p)
                obs_vals.append(y)

        if preds:
            obs_dict[basin_id] = np.concatenate(obs_vals)
            pred_dict[basin_id] = np.concatenate(preds)
            if _is_ensemble and ens_preds:
                ensemble_dict[basin_id] = np.concatenate(ens_preds, axis=1)  # [N, T*H]

    return obs_dict, pred_dict, ensemble_dict


def run_persistence_baseline(
    loader, basin_ids, test_start, test_end
) -> tuple[dict, dict]:
    """Run persistence baseline (predict yesterday's flow)."""
    obs_dict = {}
    pred_dict = {}
    for basin_id in basin_ids:
        try:
            sf = loader.load_basin_streamflow(basin_id, test_start, test_end)
        except (FileNotFoundError, KeyError):
            continue
        arr = sf.values.astype(np.float64)
        valid = ~np.isnan(arr)
        if valid.sum() < 10:
            continue
        # Persistence: predict t-1 for t
        obs_dict[basin_id] = arr[1:]
        pred_dict[basin_id] = arr[:-1]
    return obs_dict, pred_dict


def run_climatology_baseline(
    loader, basin_ids, train_start, train_end, test_start, test_end
) -> tuple[dict, dict]:
    """Run climatology baseline (predict day-of-year mean from training)."""
    import pandas as pd

    # Build DOY means from training data
    train_flows = {}
    for basin_id in basin_ids:
        try:
            sf = loader.load_basin_streamflow(basin_id, train_start, train_end)
            train_flows[basin_id] = sf
        except (FileNotFoundError, KeyError):
            continue

    all_train = pd.concat(train_flows.values()).dropna()
    doy = all_train.index.dayofyear
    doy_means = {}
    for d in range(1, 367):
        mask = doy == d
        doy_means[d] = float(all_train[mask].mean()) if mask.any() else float(all_train.mean())

    obs_dict = {}
    pred_dict = {}
    for basin_id in basin_ids:
        try:
            sf = loader.load_basin_streamflow(basin_id, test_start, test_end)
        except (FileNotFoundError, KeyError):
            continue
        arr = sf.values.astype(np.float64)
        doys = sf.index.dayofyear
        pred_arr = np.array([doy_means.get(d, 0.0) for d in doys])
        valid = ~np.isnan(arr)
        obs_dict[basin_id] = arr[valid]
        pred_dict[basin_id] = pred_arr[valid]
    return obs_dict, pred_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FloodCastBench benchmark.")
    parser.add_argument("--config", type=str, required=True, help="Benchmark config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--normalizer", type=str, default=None, help="Normalizer JSON path.")
    parser.add_argument("--model-type", type=str, default="lstm", help="Model type.")
    parser.add_argument("--device", type=str, default="cpu", help="Device.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir.")
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config)

    # Setup
    data_cfg = cfg.get("data", {})
    bench_cfg = cfg.get("benchmark", {})
    model_cfg = cfg.get("model", {})

    dataset_type = data_cfg.get("dataset", "camels")
    data_dir = data_cfg.get("data_dir", "data/camels")
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/benchmark")

    if dataset_type == "caravan":
        loader = CaravanLoader(data_dir)
    else:
        loader = CAMELSLoader(data_dir)

    basin_ids = loader.list_basins()
    max_basins = bench_cfg.get("max_basins")
    if max_basins:
        basin_ids = basin_ids[:max_basins]

    train_start = bench_cfg.get("train_start", "1980-10-01")
    train_end = bench_cfg.get("train_end", "1995-09-30")
    test_start = bench_cfg.get("test_start", "1995-10-01")
    test_end = bench_cfg.get("test_end", "2008-09-30")
    seq_length = data_cfg.get("seq_length", 365)
    forecast_horizon = data_cfg.get("forecast_horizon", 1)

    protocol = BenchmarkProtocol(
        basin_ids=basin_ids,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    all_results: list[BenchmarkResults] = []

    # --- Persistence baseline ---
    logger.info("Running persistence baseline...")
    obs_p, pred_p = run_persistence_baseline(loader, basin_ids, test_start, test_end)
    results_persistence = protocol.evaluate(obs_p, pred_p, model_name="Persistence")
    all_results.append(results_persistence)
    logger.info(f"Persistence: NSE median={results_persistence.nse_median:.3f}")

    # --- Climatology baseline ---
    logger.info("Running climatology baseline...")
    obs_c, pred_c = run_climatology_baseline(
        loader, basin_ids, train_start, train_end, test_start, test_end
    )
    results_clim = protocol.evaluate(obs_c, pred_c, model_name="Climatology")
    all_results.append(results_clim)
    logger.info(f"Climatology: NSE median={results_clim.nse_median:.3f}")

    # --- Trained model (if checkpoint provided) ---
    if args.checkpoint:
        logger.info(f"Loading model from {args.checkpoint}...")
        normalizer = None
        if args.normalizer:
            normalizer = BasinNormalizer.load(args.normalizer)

        # Determine n_features from a sample basin
        sample_basin = basin_ids[0]
        sample_forcing = loader.load_basin_forcing(sample_basin, test_start, test_end)
        n_features = sample_forcing.shape[1]

        model_kwargs = dict(
            n_features=n_features,
            hidden_size=model_cfg.get("hidden_size", 256),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.0),
            forecast_horizon=forecast_horizon,
        )
        # Pass FGN-specific params for ensemble models
        if args.model_type.startswith("fgn_"):
            model_kwargs["noise_dim"] = model_cfg.get("noise_dim", 32)
            model_kwargs["n_ensemble"] = model_cfg.get("n_ensemble", 2)

        model = build_model(args.model_type, **model_kwargs)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        n_ens = model_cfg.get("n_ensemble_inference", 50)
        obs_m, pred_m, ens_m = evaluate_model_on_basins(
            model, loader, basin_ids, test_start, test_end,
            normalizer, seq_length, forecast_horizon, args.device,
            n_ensemble=n_ens,
        )
        model_name = f"CatchmentLSTM" if args.model_type == "lstm" else f"Catchment{args.model_type.title()}"
        results_model = protocol.evaluate(obs_m, pred_m, model_name=model_name)

        # Compute CRPS from ensemble if available
        if ens_m:
            crps_values = []
            for basin_id in results_model.basin_ids:
                if basin_id in ens_m and basin_id in obs_m:
                    ens_arr = ens_m[basin_id]  # [N, T]
                    obs_arr = obs_m[basin_id]  # [T]
                    # CRPS per timestep: (1/N)*sum|x_n-y| - 1/(2N(N-1))*sum|x_n-x_n'|
                    N = ens_arr.shape[0]
                    mae = np.abs(ens_arr - obs_arr[np.newaxis, :]).mean(axis=0)
                    if N > 1:
                        spread = np.abs(
                            ens_arr[:, np.newaxis, :] - ens_arr[np.newaxis, :, :]
                        ).sum(axis=(0, 1)) / (2 * N * (N - 1))
                    else:
                        spread = np.zeros_like(mae)
                    crps_values.append((mae - spread).mean())
            if crps_values:
                results_model.crps = float(np.mean(crps_values))

        all_results.append(results_model)
        logger.info(f"{model_name}: NSE median={results_model.nse_median:.3f}")

    # --- Generate report ---
    logger.info(f"Generating report to {output_dir}...")
    generate_report(all_results, output_dir)
    logger.info("Benchmark complete!")

    # Print summary table
    print("\n" + "=" * 70)
    print("FloodCastBench Results")
    print("=" * 70)
    print(f"{'Model':<20} {'NSE_med':>8} {'KGE_med':>8} {'NSE_high':>9} {'CSI':>6} {'POD':>6} {'FAR':>6} {'CRPS':>8}")
    print("-" * 78)
    for r in all_results:
        crps_str = f"{r.crps:>8.4f}" if not np.isnan(r.crps) else f"{'N/A':>8}"
        print(
            f"{r.model_name:<20} {r.nse_median:>8.3f} {r.kge_median:>8.3f} "
            f"{r.nse_high:>9.3f} {r.csi:>6.3f} {r.pod:>6.3f} {r.far:>6.3f} {crps_str}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
