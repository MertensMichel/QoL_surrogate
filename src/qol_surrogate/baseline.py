"""[LEGACY: 3-mode/21-channel pipeline] Boosted Forest baseline: same TAZ-level pipeline, same target definition
(accessibility deviation from the dry baseline), same train/val/test split as
the GCN - but the graph is removed entirely, not approximated.

HistGradientBoostingRegressor is inherently single-output, and there's no
multi-output boosted-tree model - so each of the 21 mode x POI channels gets
its own MultiOutputRegressor(HistGradientBoostingRegressor). Under the hood
that's 277 independent single-output boosted-tree ensembles per channel (5817
total across all channels), each one taking the FULL flattened scenario (all
277 TAZs' 5 input features, 1385 total) as input and predicting its own TAZ's
value for that channel. No adjacency is used anywhere - any cross-TAZ signal
the trees pick up has to come from raw feature co-occurrence in the flattened
input, not an explicit graph structure.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from qol_surrogate.data import TRANSPORT_MODES
from qol_surrogate.evaluate import (
    DRY_BASELINE_DIR, HEX_PARQUET_DIR, Y_COLS,
    _load_taz_dry_baseline, _load_true_hex, _metrics_for, _summarize,
)


def flatten_x(x):
    """[n_scenarios, 277, 5] -> [n_scenarios, 1385] - one row per scenario, all
    TAZs' input features concatenated (TAZ-major, feature-minor - whatever
    torch's reshape naturally does). Same flattening must be used at train and
    inference time, which it is here since it's just a reshape of the same
    underlying tensor layout everywhere."""
    return x.reshape(x.shape[0], -1).numpy()


def build_channel_targets(y):
    """[n_scenarios, 277, 21] -> list of 21 [n_scenarios, 277] arrays, one per
    mode x POI channel, in Y_COLS order."""
    return [y[:, :, i].numpy() for i in range(y.shape[-1])]


def build_channel_model(**hgb_kwargs):
    """One [1385] -> [277] regressor for a single channel. early_stopping is
    forced on (sklearn's 'auto' default disables it below 10k samples, but we
    have ~637 training scenarios against 1385 input features - a p >> n
    regime where early stopping matters more, not less)."""
    defaults = dict(early_stopping=True, random_state=13)
    defaults.update(hgb_kwargs)
    return MultiOutputRegressor(HistGradientBoostingRegressor(**defaults), n_jobs=-1)


def train_baseline(x_train_flat, y_train_channels, **hgb_kwargs):
    """Fit one [1385] -> [277] MultiOutputRegressor per channel.
    Returns {channel_name: fitted MultiOutputRegressor}."""
    models = {}
    for col, y_col in zip(Y_COLS, y_train_channels):
        model = build_channel_model(**hgb_kwargs)
        model.fit(x_train_flat, y_col)
        models[col] = model
    return models


def predict(models, x_flat):
    """{channel: model} -> [n_scenarios, 277, 21] stacked predictions, same
    layout as the GCN's (deviation, un-normalized) output."""
    preds = [models[col].predict(x_flat) for col in Y_COLS]  # each [n_scenarios, 277]
    return np.stack(preds, axis=-1)  # [n_scenarios, 277, 21]


def evaluate_baseline(models, x_test_flat, y_test, dry_baseline_dir=DRY_BASELINE_DIR):
    """Same {"deviation": ..., "absolute": ...} structure as evaluate.evaluate()
    - no normalization step here, the trees are fit directly on raw deviation
    units, so predict() already returns real units."""
    preds_deviation = torch.from_numpy(predict(models, x_test_flat)).float()
    targets_deviation = y_test

    y_dry_taz = _load_taz_dry_baseline(dry_baseline_dir)
    preds_absolute = preds_deviation + y_dry_taz
    targets_absolute = targets_deviation + y_dry_taz

    return {
        "deviation": _summarize(preds_deviation, targets_deviation),
        "absolute": _summarize(preds_absolute, targets_absolute),
    }


def evaluate_baseline_per_taz(models, x_test_flat, y_test, taz_ids, dry_baseline_dir=DRY_BASELINE_DIR):
    """Same {metric: DataFrame(taz_id x mode)} structure as evaluate.evaluate_per_taz()."""
    preds_deviation = torch.from_numpy(predict(models, x_test_flat)).float()
    targets_deviation = y_test

    y_dry_taz = _load_taz_dry_baseline(dry_baseline_dir)
    preds_absolute = preds_deviation + y_dry_taz
    targets_absolute = targets_deviation + y_dry_taz

    metric_names = ["r2", "mae", "c_index", "wape"]
    result = {metric: {"taz_id": taz_ids} for metric in metric_names}

    for mode in TRANSPORT_MODES:
        channel_idx = [i for i, col in enumerate(Y_COLS) if col.startswith(f"cumulative_accessibility_{mode}_")]
        pred_mode = preds_absolute[:, :, channel_idx]
        true_mode = targets_absolute[:, :, channel_idx]

        per_taz = {metric: [] for metric in metric_names}
        for taz_idx in range(len(taz_ids)):
            pred_taz = pred_mode[:, taz_idx, :].numpy().reshape(-1)
            true_taz = true_mode[:, taz_idx, :].numpy().reshape(-1)
            m = _metrics_for(pred_taz, true_taz)
            for metric in metric_names:
                per_taz[metric].append(m[metric])

        for metric in metric_names:
            result[metric][mode] = per_taz[metric]

    return {metric: pd.DataFrame(result[metric]).set_index("taz_id") for metric in metric_names}


def evaluate_baseline_hex_level(models, x_test_flat, idx_test, hexes, taz_to_idx, include_file_19=True,
                                 hex_parquet_dir=HEX_PARQUET_DIR, dry_baseline_dir=DRY_BASELINE_DIR):
    """Same {channel: {r2, mae, c_index, wape}} structure as evaluate.evaluate_hex_level()."""
    preds_deviation = torch.from_numpy(predict(models, x_test_flat)).float()  # [n_test, 277, 21]

    y_dry_taz = _load_taz_dry_baseline(dry_baseline_dir)
    preds_absolute_taz = preds_deviation + y_dry_taz

    taz_idx_per_hex = hexes['taz_zoneid'].map(taz_to_idx).to_numpy()
    preds_absolute_hex = preds_absolute_taz[:, taz_idx_per_hex, :]

    true_hex = _load_true_hex(include_file_19=include_file_19, hex_parquet_dir=hex_parquet_dir)
    true_hex_test = true_hex[idx_test]

    results = {}
    for i, col in enumerate(Y_COLS):
        pred_i = preds_absolute_hex[:, :, i].numpy().reshape(-1)
        true_i = true_hex_test[:, :, i].numpy().reshape(-1)
        results[col] = _metrics_for(pred_i, true_i)

    return results


def evaluate_baseline_hex_level_per_taz(models, x_test_flat, idx_test, hexes, taz_to_idx, taz_ids, include_file_19=True,
                                         hex_parquet_dir=HEX_PARQUET_DIR, dry_baseline_dir=DRY_BASELINE_DIR):
    """Same {metric: DataFrame(taz_id x mode)} structure as
    evaluate.evaluate_hex_level_per_taz()."""
    preds_deviation = torch.from_numpy(predict(models, x_test_flat)).float()

    y_dry_taz = _load_taz_dry_baseline(dry_baseline_dir)
    preds_absolute_taz = preds_deviation + y_dry_taz

    taz_idx_per_hex = hexes['taz_zoneid'].map(taz_to_idx).to_numpy()
    preds_absolute_hex = preds_absolute_taz[:, taz_idx_per_hex, :]

    true_hex = _load_true_hex(include_file_19=include_file_19, hex_parquet_dir=hex_parquet_dir)
    true_hex_test = true_hex[idx_test]

    metric_names = ["r2", "mae", "c_index", "wape"]
    result = {metric: {"taz_id": taz_ids} for metric in metric_names}

    for mode in TRANSPORT_MODES:
        channel_idx = [i for i, col in enumerate(Y_COLS) if col.startswith(f"cumulative_accessibility_{mode}_")]
        pred_mode = preds_absolute_hex[:, :, channel_idx]
        true_mode = true_hex_test[:, :, channel_idx]

        per_taz = {metric: [] for metric in metric_names}
        for taz_position in range(len(taz_ids)):
            hex_mask = taz_idx_per_hex == taz_position
            pred_taz = pred_mode[:, hex_mask, :].numpy().reshape(-1)
            true_taz = true_mode[:, hex_mask, :].numpy().reshape(-1)
            m = _metrics_for(pred_taz, true_taz)
            for metric in metric_names:
                per_taz[metric].append(m[metric])

        for metric in metric_names:
            result[metric][mode] = per_taz[metric]

    return {metric: pd.DataFrame(result[metric]).set_index("taz_id") for metric in metric_names}
