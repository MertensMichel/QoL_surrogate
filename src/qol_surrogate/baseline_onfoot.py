"""Boosted Forest baseline for the ON_FOOT-only surrogate: same TAZ-level
pipeline (29 per-TAZ input features - see data_onfoot.STATIC_FEATURE_COLS +
the 5 dynamic water-depth stats), same target definition (accessibility
deviation from the dry baseline), same train/val/test split as
model_architecture.GCNResNet - but the graph is removed entirely, not approximated.

Ported from qol_surrogate.baseline (the old 21-channel/3-mode version) - same
architecture and reasoning, adapted to 7 output channels (POI_CATEGORIES only)
and to the onfoot data pipeline, which has no "include_file_19" split and
bundles its dry baseline as a plain tensor rather than a separate on-disk
parquet directory (see data_onfoot.load_dataset_tensors's y_dry return).

HistGradientBoostingRegressor is inherently single-output, and there's no
multi-output boosted-tree model - so each of the 7 POI-category channels gets
its own MultiOutputRegressor(HistGradientBoostingRegressor). Under the hood
that's 277 independent single-output boosted-tree ensembles per channel (1939
total across all channels), each one taking the FULL flattened scenario (all
277 TAZs' 29 input features, 8033 total) as input and predicting its own
TAZ's value for that channel. No adjacency is used anywhere - any cross-TAZ
signal the trees pick up has to come from raw feature co-occurrence in the
flattened input, not an explicit graph structure.
"""

import time

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# Duplicated from qol_surrogate.data_onfoot.POI_CATEGORIES rather than imported -
# order must match the channel order everywhere below
POI_CATEGORIES = ['cultural', 'education', 'green_space', 'health',
                   'public_spaces', 'public_transportation', 'sports']


def flatten_x(x):
    """[n_scenarios, n_taz, n_features] -> [n_scenarios, n_taz * n_features] -
    one row per scenario, all TAZs' input features concatenated (TAZ-major,
    feature-minor - whatever torch's reshape naturally does). Same flattening
    must be used at train and inference time, which it is here since it's
    just a reshape of the same underlying tensor layout everywhere."""
    return x.reshape(x.shape[0], -1).numpy()


def build_channel_targets(y):
    """[n_scenarios, n_taz, 7] -> list of 7 [n_scenarios, n_taz] arrays, one
    per POI-category channel, in POI_CATEGORIES order."""
    return [y[:, :, i].numpy() for i in range(y.shape[-1])]


def build_channel_model(**hgb_kwargs):
    """One [n_taz * n_features] -> [n_taz] regressor for a single channel.
    early_stopping is forced on (sklearn's 'auto' default disables it below
    10k samples, but we have ~15k training scenarios against 8033 input
    features - still a regime where early stopping is worth keeping on)."""
    defaults = dict(early_stopping=True, random_state=13)
    defaults.update(hgb_kwargs)
    return MultiOutputRegressor(HistGradientBoostingRegressor(**defaults), n_jobs=-1)


def train_baseline(x_train_flat, y_train_channels, **hgb_kwargs):
    """Fit one [n_taz * n_features] -> [n_taz] MultiOutputRegressor per
    channel. Returns {category: fitted MultiOutputRegressor}.

    Channels are fit sequentially (each one internally parallelized across
    all 277 per-TAZ trees via n_jobs=-1) - a print per channel is the only
    progress signal available for a job that can otherwise run for hours with
    no visible output at all."""
    models = {}
    for i, (category, y_col) in enumerate(zip(POI_CATEGORIES, y_train_channels)):
        start = time.time()
        model = build_channel_model(**hgb_kwargs)
        model.fit(x_train_flat, y_col)
        models[category] = model
        print(f'  channel [{i + 1}/{len(POI_CATEGORIES)}] "{category}" done, {time.time() - start:.1f}s', flush=True)
    return models


def predict(models, x_flat):
    """{category: model} -> [n_scenarios, n_taz, 7] stacked predictions, same
    layout as the GCN's (deviation, un-normalized) output."""
    preds = [models[category].predict(x_flat) for category in POI_CATEGORIES]  # each [n_scenarios, n_taz]
    return np.stack(preds, axis=-1)  # [n_scenarios, n_taz, 7]


def _metrics_for(pred_np, true_np):
    wape = np.sum(np.abs(true_np - pred_np)) / np.sum(np.abs(true_np))
    return {
        "r2": r2_score(true_np, pred_np),
        "mae": mean_absolute_error(true_np, pred_np),
        "c_index": concordance_index(true_np, pred_np),
        "wape": wape,
    }


def _summarize(pred_t, true_t):
    """overall (pooled) + per_channel (7 POI categories) metrics for one
    [n_scenarios, n_taz, 7] pair of prediction/target tensors."""
    overall = _metrics_for(pred_t.numpy().reshape(-1), true_t.numpy().reshape(-1))

    per_channel = {}
    for i, category in enumerate(POI_CATEGORIES):
        pred_i = pred_t[:, :, i].numpy().reshape(-1)
        true_i = true_t[:, :, i].numpy().reshape(-1)
        per_channel[category] = _metrics_for(pred_i, true_i)

    return {"overall": overall, "per_channel": per_channel}


def evaluate_baseline_onfoot(models, x_test_flat, y_test, y_dry_taz):
    """Same {"deviation": ..., "absolute": ...} structure as
    evaluate_onfoot.evaluate_onfoot() - no normalization step here, the trees
    are fit directly on raw deviation units, so predict() already returns
    real units. y_dry_taz: [n_taz, 7] (or broadcastable), e.g. the model's
    own dry_baseline buffer, or data_onfoot.load_dataset_tensors()'s
    y_dry.squeeze(0)."""
    preds_deviation = torch.from_numpy(predict(models, x_test_flat)).float()
    targets_deviation = y_test

    preds_absolute = preds_deviation + y_dry_taz
    targets_absolute = targets_deviation + y_dry_taz

    return {
        "deviation": _summarize(preds_deviation, targets_deviation),
        "absolute": _summarize(preds_absolute, targets_absolute),
    }


def evaluate_per_taz_baseline_onfoot(models, x_test_flat, y_test, taz_ids, y_dry_taz):
    """{r2, mae, c_index, wape} (absolute accessibility) *per TAZ zone*,
    pooled across test scenarios and the 7 POI categories - the baseline
    analogue of evaluate_onfoot.evaluate_per_taz_onfoot().

    Returns {metric: DataFrame(taz_id -> value, column "ON_FOOT")}."""
    preds_deviation = torch.from_numpy(predict(models, x_test_flat)).float()
    targets_deviation = y_test

    preds_absolute = preds_deviation + y_dry_taz
    targets_absolute = targets_deviation + y_dry_taz

    metric_names = ["r2", "mae", "c_index", "wape"]
    per_taz = {metric: [] for metric in metric_names}

    for taz_idx in range(len(taz_ids)):
        pred_taz = preds_absolute[:, taz_idx, :].numpy().reshape(-1)   # pooled over scenarios + POI categories
        true_taz = targets_absolute[:, taz_idx, :].numpy().reshape(-1)
        m = _metrics_for(pred_taz, true_taz)
        for metric in metric_names:
            per_taz[metric].append(m[metric])

    return {
        metric: pd.DataFrame({"ON_FOOT": per_taz[metric]}, index=pd.Index(taz_ids, name="taz_id"))
        for metric in metric_names
    }
