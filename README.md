# FloodRisk

End-to-end ML flood forecasting pipeline coupling NeuralGCM atmospheric precipitation with hydrological streamflow models.

## Overview

FloodRisk builds a modular, config-driven research pipeline that connects atmospheric forecasts (NeuralGCM) with ML-based hydrological models for flood risk prediction. The pipeline spans from raw meteorological forcing data through streamflow prediction to flood event evaluation.

**Key features:**
- LSTM and Transformer models for multi-basin streamflow forecasting
- CAMELS-US (671 basins) and Caravan (6,830+ global basins) dataset support
- Nash-Sutcliffe Efficiency (NSE) loss and Kling-Gupta Efficiency (KGE) metrics
- Optional NeuralGCM precipitation integration (decoupled JAX process)
- Config-driven experiments via YAML
- Flood event verification metrics (CSI, POD, FAR)

## Installation

```bash
# Core installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With NeuralGCM support (JAX)
pip install -e ".[neuralgcm]"
```

## Quick Start

### 1. Download CAMELS-US data

```bash
python scripts/download_camels.py --output_dir data/camels
```

### 2. Train a streamflow model

```bash
python scripts/train_streamflow.py --config configs/streamflow/lstm_camels.yaml
```

### 3. Evaluate

```bash
python scripts/evaluate.py --config configs/streamflow/lstm_camels.yaml --checkpoint checkpoints/best.pt
```

## Pipeline

```
ERA5 / CAMELS forcing data
         |
         v
  +--------------+     Optional: NeuralGCM precip
  | Data Loading |<------ (cached NetCDF files)
  | & Normalize  |
  +------+-------+
         |
         v
  +--------------+
  | CatchmentDS  |  PyTorch Dataset: (x, y) tuples
  | [seq, feat]  |  x=[365, n_features], y=[horizon]
  +------+-------+
         |
         v
  +--------------+
  |   Trainer    |  Trainer.fit() with NSELoss
  |              |  Metrics: NSE, KGE, CSI
  +------+-------+
         |
         v
  +--------------+
  |  Evaluation  |  Per-basin NSE, flood event detection
  |  & Benchmark |  Compare vs GloFAS/GRDC
  +--------------+
```

## Project Structure

```
floodrisk/
  config.py              # ExperimentConfig (composes TrainerConfig)
  torchharness.py        # Vendored training harness (Trainer, Metric, Callback)
  losses.py              # NSELoss
  data/
    camels.py            # CAMELS-US loader (671 basins)
    normalization.py     # Per-basin & global normalization
    caravan.py           # Caravan global loader (extension)
  datasets/
    streamflow.py        # CatchmentDataset (PyTorch Dataset)
  features/
    temporal.py          # Day-of-year, lag features
    catchment.py         # Static catchment attributes
  models/
    lstm.py              # CatchmentLSTM
    transformer.py       # CatchmentTransformer
  metrics/
    hydrology.py         # NSE, KGE
    flood_event.py       # CSI, POD, FAR
  callbacks/
    hydro_logger.py      # Hydrograph visualization callback
  neuralgcm_bridge/
    forecast.py          # NeuralGCM forecaster (JAX, separate process)
    regrid.py            # Regrid 2.8deg to catchment-mean
    cache.py             # Forecast cache management
  evaluation/
    evaluate.py          # Full eval pipeline
    plots.py             # Hydrographs, maps, scatter plots
configs/                 # YAML experiment configs
scripts/                 # Training, evaluation, download entry points
notebooks/               # Jupyter notebooks for exploration
tests/                   # Unit tests
```

## Configuration

Experiments are configured via YAML files. Example (`configs/streamflow/lstm_camels.yaml`):

```yaml
trainer:
  lr: 0.001
  epochs: 100
  batch_size: 256
  device: cuda

data:
  dataset: camels
  seq_length: 365
  forecast_horizon: 1
  train_start: "1980-10-01"
  train_end: "1995-09-30"

model:
  type: lstm
  hidden_size: 256
  num_layers: 2
  dropout: 0.2
```

## Testing

```bash
pytest tests/ -v
```

## References

- Kratzert et al. (2024). "Caravan - A global community dataset for large-sample hydrology"
- Kochkov et al. (2024). "Neural General Circulation Models for Weather and Climate"
- Newman et al. (2015). "CAMELS: Catchment Attributes and Meteorology for Large-sample Studies"

## License

MIT
