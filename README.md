# FloodRisk

End-to-end ML flood forecasting pipeline coupling NeuralGCM atmospheric precipitation with hydrological streamflow models.

## Overview

FloodRisk builds a modular, config-driven research pipeline that connects atmospheric forecasts (NeuralGCM) with ML-based hydrological models for flood risk prediction. The pipeline spans from raw meteorological forcing data through streamflow prediction to flood event evaluation.

**Key features:**
- LSTM and Transformer models for multi-basin streamflow forecasting
- **FGN probabilistic models** — noise-injected LSTM/Transformer for ensemble forecasts via Functional Generative Networks (adapted from WeatherNext 2)
- CAMELS-US (671 basins) and Caravan (6,830+ global basins) dataset support
- Nash-Sutcliffe Efficiency (NSE) loss and Kling-Gupta Efficiency (KGE) metrics
- **CRPS loss and ensemble metrics** for probabilistic forecast evaluation
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
# Deterministic LSTM
python scripts/train_streamflow.py --config configs/streamflow/lstm_camels.yaml

# Probabilistic FGN-LSTM (ensemble forecasts with CRPS loss)
python scripts/train_streamflow.py --config configs/streamflow/fgn_lstm_camels.yaml
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
  |   Trainer    |  NSELoss (deterministic) or CRPSLoss (FGN)
  |              |  Metrics: NSE, KGE, CRPS, Spread-Skill
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
  losses.py              # NSELoss, CRPSLoss
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
    fgn_streamflow.py    # FGNStreamflowLSTM, FGNStreamflowTransformer (probabilistic)
  metrics/
    hydrology.py         # NSE, KGE
    flood_event.py       # CSI, POD, FAR
    ensemble.py          # CRPS, Spread-Skill, EnsembleNSE
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
  type: lstm           # or fgn_lstm, transformer, fgn_transformer
  hidden_size: 256
  num_layers: 2
  dropout: 0.2
  # FGN-specific (probabilistic models only):
  # noise_dim: 32
  # n_ensemble: 2
```

## Testing

```bash
pytest tests/ -v
```

## Probabilistic Forecasting (FGN)

The FGN (Functional Generative Network) models adapt the approach from WeatherNext 2 for streamflow forecasting. A low-dimensional noise vector `z ~ N(0, I)` is injected via conditional LayerNorm layers throughout the network. Different noise samples produce different ensemble members from the same model in a single forward pass.

- **Training**: Uses fair CRPS loss with N=2 ensemble members per sample
- **Inference**: Sample N=50 noise vectors for full ensemble prediction
- **Models**: `fgn_lstm` and `fgn_transformer` available via config

## References

- Kratzert et al. (2024). "Caravan - A global community dataset for large-sample hydrology"
- Kochkov et al. (2024). "Neural General Circulation Models for Weather and Climate"
- Price et al. (2025). "Probabilistic weather forecasting with machine learning" (FGN / WeatherNext 2)
- Newman et al. (2015). "CAMELS: Catchment Attributes and Meteorology for Large-sample Studies"

## License

MIT
