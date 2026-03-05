#!/usr/bin/env python
"""Run a NeuralGCM precipitation forecast.

Standalone script that loads a NeuralGCM checkpoint and produces a
precipitation forecast saved as NetCDF.

Usage::

    python scripts/run_neuralgcm_forecast.py \
        --checkpoint path/to/checkpoint.pkl \
        --init_time 2020-01-01T00:00:00 \
        --duration 240 \
        --output_dir cache/neuralgcm
"""

from __future__ import annotations

import argparse

from floodrisk.neuralgcm_bridge.forecast import NeuralGCMForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NeuralGCM precipitation forecast.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NeuralGCM checkpoint.")
    parser.add_argument("--init_time", type=str, required=True, help="Forecast init time (ISO format).")
    parser.add_argument("--duration", type=int, default=240, help="Forecast duration in hours (default: 240).")
    parser.add_argument("--output_dir", type=str, default="cache/neuralgcm", help="Output directory for NetCDF files.")
    args = parser.parse_args()

    forecaster = NeuralGCMForecaster(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )

    ds = forecaster.run_forecast(
        init_time=args.init_time,
        duration_hours=args.duration,
    )

    output_path = forecaster._output_path(
        __import__("datetime").datetime.fromisoformat(args.init_time)
    )
    print(f"Forecast saved to: {output_path}")
    print(f"Variables: {list(ds.data_vars)}")


if __name__ == "__main__":
    main()
