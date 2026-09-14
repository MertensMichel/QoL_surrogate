# QoL_surrogate

A surrogate model for the Accessibility needed Quality of Life (QoL) calculations of the MAAT RL Framework.

## Overview

Given a flood scenario (water depths across the road network), predicts accessibility
(cumulative reachable POIs, per TAZ zone, per POI category) as a fast learned substitute
for the full accessibility simulation - so the MAAT RL agent can query
accessibility on every step without re-running the expensive simulation.

Three transport modes - **ON_FOOT**, **CAR**, **BICYCLE** - are each served by their own
dedicated GCN-ResNet model, rather than one model jointly predicting all three.

An earlier, superseded **joint** 3-mode/21-channel pipeline (`model.py`, `data.py`,
`evaluate.py`, `baseline.py`, plus their matching `scripts/`) is kept in the repo for
reference/comparison. Every file is tagged at the top of its docstring so it's clear at
a glance which track it belongs to:
- `[LEGACY: 3-mode/21-channel pipeline]` - the old joint-model approach
- `[SHARED: ...]` - used by all three per-mode pipelines (or by both legacy and per-mode)
- `[ONFOOT]` / `[CAR]` / `[BICYCLE]` - specific to that mode's dedicated pipeline

## Quick start (inference)

```python
from qol_surrogate.inference import QoL_surrogate

surrogate = QoL_surrogate(mode="car")           # "onfoot", "car", or "bicycle"
result = surrogate.predict(raw_water_depths)    # DataFrame: taz_zoneid x POI category
```

`raw_water_depths` is a 1D array-like of edge-resolution water depths for that mode's
road network (same order as that mode's `{MODE}_edges_Copenhagen.pkl`). Load once at
setup time, then call `.predict()` repeatedly (e.g. once per RL step) - see
`inference.py`'s module docstring.

## Pipeline (per mode)

1. **Aggregate** raw hex/edge-resolution scenario data to TAZ level: `scripts/agg_{mode}.py`.
   ON_FOOT's `scripts/agg_and_static_features.py` additionally builds the static
   features/graph that CAR and BICYCLE reuse unchanged (they're mode-independent). Each
   mode also saves its own real dry (zero-flood) baseline, which the model's "absolute
   accessibility" predictions are anchored to.
2. **Train**: `scripts/train_{mode}.py` -> `models/gcn_resnet_{mode}_<timestamp>/model.pt`
   (gitignored - checkpoints aren't committed).
3. **Evaluate**: `scripts/evaluate_{mode}.py` -> `eval_results.pkl` saved into that run's folder.
4. **Compare**:
   - `notebooks/compare_modes.ipynb` - the GCN's own accuracy across all 3 modes.
   - `notebooks/compare_baseline_onfoot.ipynb` - GCN vs. Boosted Forest vs. Linear
     Regression, ON_FOOT only.
5. **Ship**: copy the trained checkpoint to
   `/mnt/raid1/MAAT/20.surrogate_data/qol_surrogate/qol_surrogate_weights_{mode}.pt` -
   `inference.py`'s `mode=` argument resolves to that path automatically.

## File structure

```
QoL_surrogate/
├── src/qol_surrogate/                    # installable package - reusable library code
│   ├── data_onfoot.py                    # [ONFOOT] aggregation + tensor-building
│   ├── data_car.py                       # [CAR]        "
│   ├── data_bicycle.py                   # [BICYCLE]    "
│   ├── model_architecture.py             # [SHARED] GCN-ResNet class, all 3 modes
│   ├── train.py                          # [SHARED] generic training loop
│   ├── inference.py                      # [SHARED] production QoL_surrogate class
│   ├── evaluate_onfoot.py                # [ONFOOT] TAZ-level R2/MAE/C-index/WAPE
│   ├── evaluate_car.py                   # [CAR]        "
│   ├── evaluate_bicycle.py               # [BICYCLE]    "
│   ├── baseline_onfoot.py                # [ONFOOT] Boosted Forest baseline
│   ├── baseline_linear_onfoot.py         # [ONFOOT] Linear Regression baseline
│   ├── model.py                          # [LEGACY] joint 3-mode/21-channel model
│   ├── data.py                           # [LEGACY]     "
│   ├── evaluate.py                       # [LEGACY]     "
│   └── baseline.py                       # [LEGACY]     "
│
├── scripts/                              # entry points - run as `python scripts/<name>.py`
│   ├── agg_and_static_features.py        # [ONFOOT + SHARED] onfoot aggregation + shared static features/graph
│   ├── agg_car.py / agg_bicycle.py       # [CAR] / [BICYCLE] aggregation only (reuse the shared static features)
│   ├── train_onfoot.py / train_car.py / train_bicycle.py
│   ├── evaluate_onfoot.py / evaluate_car.py / evaluate_bicycle.py
│   ├── train_baseline_onfoot.py / evaluate_baseline_onfoot.py             # [ONFOOT] Boosted Forest
│   ├── train_baseline_linear_onfoot.py / evaluate_baseline_linear_onfoot.py  # [ONFOOT] Linear Regression
│   └── train.py, evaluate.py, tune.py, train_data_subsets.py,             # [LEGACY]
│       compare_baseline.py, train_baseline.py, evaluate_baseline.py
│
├── notebooks/
│   ├── compare_modes.ipynb               # GCN accuracy across ON_FOOT/CAR/BICYCLE
│   ├── compare_baseline_onfoot.ipynb     # GCN vs. Boosted Forest vs. Linear Regression (ON_FOOT)
│   ├── evaluate.ipynb                    # [LEGACY] data-subset learning-curve results
│   ├── evaluate_hex.ipynb                # [LEGACY] hex-resolution evaluation
│   └── compare_baseline.ipynb            # [LEGACY] GCN vs. Boosted Forest
│
├── models/                               # trained checkpoints (gitignored, not in version control)
├── pyproject.toml
└── README.md
```

## Known limitations

- **CAR/BICYCLE train on far less data.** ~4,500 scenarios each vs. ON_FOOT's ~22,000, using
  the same untouched hyperparameters (`HIDDEN_CHANNELS=256`, `PATIENCE=12`, etc.) - not
  re-tuned for the smaller dataset. Evaluated accuracy is comparable to ON_FOOT regardless,
  but this hasn't been stress-tested.
