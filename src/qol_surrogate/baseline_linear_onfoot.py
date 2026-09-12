"""Multiple Linear Regression baseline for the ON_FOOT-only surrogate: same
TAZ-level pipeline (29 per-TAZ input features flattened across all 277 TAZ,
8033-dim input), same 7 POI-category output channels, same train/val/test
split as model_architecture.GCNResNet and baseline_onfoot's Boosted Forest - but
each per-TAZ target is now predicted by an ordinary multiple linear
regression instead of a boosted-tree ensemble. "Multiple" refers to the 8033
input features, not multiple outputs - conceptually this is still one
per-(channel, TAZ) predictor, same as the Boosted Forest baseline (1939
independent targets total: 7 channels x 277 TAZ), each a linear function of
the full flattened scenario.

Unlike the Boosted Forest, this does NOT fit 1939 independent models. Trees
can't share structure across outputs, so the Boosted Forest genuinely needs
a separate ensemble per target - but ordinary least squares over the SAME
design matrix factors per-output: the expensive part (decomposing
x_train_flat) only has to happen once, reused for every channel/TAZ target
via a single LinearRegression fit with 2D y. This gives mathematically
identical per-target coefficients to 1939 separate
LinearRegression(x_train_flat, y_col) fits, just far cheaper - and produces
one small model object instead of 1939 large ones.
"""

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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


def train_baseline_linear(x_train_flat, y_train_channels):
    """Fit all 7 x n_taz targets in ONE shared least-squares solve (see
    module docstring). Returns the fitted LinearRegression - its coef_ is
    [7 * n_taz, n_features], intercept_ is [7 * n_taz]."""
    y_stacked = np.concatenate(y_train_channels, axis=1)  # [n_train, 7 * n_taz], channel-major
    model = LinearRegression()
    model.fit(x_train_flat, y_stacked)
    return model


def predict(model, x_flat):
    """model -> [n_scenarios, n_taz, 7] stacked predictions, same layout as
    the GCN's / Boosted Forest's (deviation, un-normalized) output."""
    n_taz = model.coef_.shape[0] // len(POI_CATEGORIES)
    preds = model.predict(x_flat)                                  # [n_scenarios, 7 * n_taz]
    preds = preds.reshape(-1, len(POI_CATEGORIES), n_taz)           # [n_scenarios, 7, n_taz]
    return preds.transpose(0, 2, 1)                                  # [n_scenarios, n_taz, 7]


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


def evaluate_baseline_linear(model, x_test_flat, y_test, y_dry_taz):
    """Same {"deviation": ..., "absolute": ...} structure as
    evaluate_onfoot.evaluate_onfoot() / baseline_onfoot.evaluate_baseline_onfoot()
    - no normalization step here, the model is fit directly on raw deviation
    units, so predict() already returns real units. y_dry_taz: [n_taz, 7]
    (or broadcastable), e.g. the GCN's own dry_baseline buffer, or
    data_onfoot.load_dataset_tensors()'s y_dry.squeeze(0)."""
    preds_deviation = torch.from_numpy(predict(model, x_test_flat)).float()
    targets_deviation = y_test

    preds_absolute = preds_deviation + y_dry_taz
    targets_absolute = targets_deviation + y_dry_taz

    return {
        "deviation": _summarize(preds_deviation, targets_deviation),
        "absolute": _summarize(preds_absolute, targets_absolute),
    }


def evaluate_per_taz_baseline_linear(model, x_test_flat, y_test, taz_ids, y_dry_taz):
    """{r2, mae, c_index, wape} (absolute accessibility) *per TAZ zone*,
    pooled across test scenarios and the 7 POI categories - the linear-
    regression analogue of baseline_onfoot.evaluate_per_taz_baseline_onfoot().

    Returns {metric: DataFrame(taz_id -> value, column "ON_FOOT")}."""
    preds_deviation = torch.from_numpy(predict(model, x_test_flat)).float()
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
