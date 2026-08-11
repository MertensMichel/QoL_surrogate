"""Evaluation metrics (R², MAE, C-index, WAPE) - TAZ level and hex level."""

import os
from itertools import product

import numpy as np
import torch
from datasets import load_dataset
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error, r2_score

from qol_surrogate.data import DRY_BASELINE_DIR, HEX_PARQUET_DIR, POI_CATEGORIES, TAZ_PARQUET_DIR, TRANSPORT_MODES

Y_COLS = [f"cumulative_accessibility_{mode}_{poi}"
          for mode, poi in product(TRANSPORT_MODES, POI_CATEGORIES)]


def _load_taz_dry_baseline(dry_baseline_dir=DRY_BASELINE_DIR):
    """TAZ-level dry (no-flood) baseline accessibility, stacked in Y_COLS order.
    Shape [1, 277, 21] - the reference added back to a predicted/true deviation
    to recover absolute accessibility.
    """
    dry_baseline = load_dataset("parquet", data_files=os.path.join(dry_baseline_dir, "*.parquet"))
    dry_baseline = dry_baseline['train']
    dry_baseline.set_format(type='torch')
    dry_baseline = dry_baseline[:]
    return torch.stack([dry_baseline[c] for c in Y_COLS], dim=-1)


def _load_true_hex(hex_parquet_dir=HEX_PARQUET_DIR):
    """True hex-resolution accessibility for every sample, stacked in Y_COLS
    order. Shape [n_samples_total, n_hex, 21]."""
    ds = load_dataset("parquet", data_files=os.path.join(hex_parquet_dir, "*.parquet"))
    ds = ds['train']
    ds.set_format(type='torch')
    full = ds[:]
    return torch.stack([full[c] for c in Y_COLS], dim=-1)


def _metrics_for(pred_np, true_np):
    # WAPE: scale-independent aggregate error - unlike MAPE it doesn't blow up
    # near zero, which matters for the deviation metrics since y is often near zero
    wape = np.sum(np.abs(true_np - pred_np)) / np.sum(np.abs(true_np))
    return {
        "r2": r2_score(true_np, pred_np),
        "mae": mean_absolute_error(true_np, pred_np),
        "c_index": concordance_index(true_np, pred_np),
        "wape": wape,
    }


def _summarize(pred_t, true_t):
    """overall (pooled) + per_channel (21, mode x POI category) metrics for one
    [n_scenarios, 277, 21] pair of prediction/target tensors."""
    overall = _metrics_for(pred_t.numpy().reshape(-1), true_t.numpy().reshape(-1))

    per_channel = {}
    for i, col in enumerate(Y_COLS):
        pred_i = pred_t[:, :, i].numpy().reshape(-1)
        true_i = true_t[:, :, i].numpy().reshape(-1)
        per_channel[col] = _metrics_for(pred_i, true_i)

    return {"overall": overall, "per_channel": per_channel}


def evaluate(model, test_loader, dry_baseline_dir=DRY_BASELINE_DIR):
    """Evaluate the model on a test set, in original (un-normalized) units.

    Reports metrics in *both* spaces, since they can diverge a lot and answer
    different questions:
    - deviation: how well the model predicts the flood-induced *change* in
      accessibility - what it's actually trained to predict.
    - absolute: how well predicted deviation + TAZ-level dry baseline tracks
      true absolute accessibility - the more real-world-interpretable number,
      but one a model that just predicts "no change" could score deceptively
      well on even while being poor at deviations.

    Returns {"deviation": {"overall": ..., "per_channel": ...},
             "absolute":  {"overall": ..., "per_channel": ...}}
    """
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            preds.append(out)
            targets.append(batch.y)

    # PyG batching concatenates all graphs' nodes into one flat axis - reshape
    # back into per-scenario structure before we need to index per channel
    n_scenarios = len(test_loader.dataset)
    preds = torch.cat(preds, dim=0).reshape(n_scenarios, -1, preds[0].shape[-1])      # [n_test_scenarios, 277, 21]
    targets = torch.cat(targets, dim=0).reshape(n_scenarios, -1, targets[0].shape[-1])

    # model + targets were trained/prepared in normalized units - convert back
    # to real accessibility-deviation units before computing interpretable metrics
    preds_deviation = preds * model.y_std + model.y_mean
    targets_deviation = targets * model.y_std + model.y_mean

    y_dry_taz = _load_taz_dry_baseline(dry_baseline_dir)
    preds_absolute = preds_deviation + y_dry_taz
    targets_absolute = targets_deviation + y_dry_taz

    return {
        "deviation": _summarize(preds_deviation, targets_deviation),
        "absolute": _summarize(preds_absolute, targets_absolute),
    }


def evaluate_hex_level(model, test_loader, idx_test, hexes, taz_to_idx,
                        hex_parquet_dir=HEX_PARQUET_DIR, dry_baseline_dir=DRY_BASELINE_DIR):
    """Evaluate the model's TAZ-level predictions against true HEX-resolution
    ground truth, by broadcasting each predicted TAZ value to every hex that
    lies within it.

    This measures model error *plus* the inherent TAZ->hex resolution loss -
    the same question the old repo's "perfect surrogate ceiling" check was
    built around (how much of the remaining error is genuine model error vs.
    unavoidable resolution loss). idx_test must be the *original* scenario
    indices (0..n_samples-1, matching row order in the source hex parquet)
    that ended up in the test split - returned by split_dataset().

    Always absolute (there's no hex-resolution dry baseline to subtract - the
    old repo's attempt at building one crashed and was never completed
    either, see gnn_hex_loss/ in the old repo). Returns one {r2, mae, c_index,
    wape} dict per output channel (21 total: mode x POI category), computed at
    hex resolution, pooled over every test scenario and every hex.
    """
    # 1. model's predicted TAZ-level deviation for the test scenarios, in the
    #    same order idx_test is in (test_loader must not be shuffled)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            preds.append(out)
    # PyG batching concatenates all graphs' nodes into one flat axis - reshape
    # back into per-scenario structure before broadcasting TAZ -> hex per scenario
    n_scenarios = len(test_loader.dataset)
    preds = torch.cat(preds, dim=0).reshape(n_scenarios, -1, preds[0].shape[-1])  # [n_test, 277, 21], normalized deviation

    preds_deviation = preds * model.y_std + model.y_mean  # un-normalize

    # 2. reconstruct predicted ABSOLUTE TAZ-level accessibility: add back the
    #    (TAZ-level) dry baseline used as the deviation reference at training time
    y_dry_taz = _load_taz_dry_baseline(dry_baseline_dir)  # [1, 277, 21]
    preds_absolute_taz = preds_deviation + y_dry_taz        # [n_test, 277, 21]

    # 3. broadcast each TAZ's predicted value to every hex within that TAZ
    taz_idx_per_hex = hexes['taz_zoneid'].map(taz_to_idx).to_numpy()  # [n_hex]
    preds_absolute_hex = preds_absolute_taz[:, taz_idx_per_hex, :]    # [n_test, n_hex, 21]

    # 4. true hex-resolution accessibility, for exactly the test scenarios
    true_hex = _load_true_hex(hex_parquet_dir)  # [n_samples_total, n_hex, 21]
    true_hex_test = true_hex[idx_test]           # [n_test, n_hex, 21], same scenario order as preds

    # 5. per-channel metrics, pooled over scenarios and hexes
    results = {}
    for i, col in enumerate(Y_COLS):
        pred_i = preds_absolute_hex[:, :, i].numpy().reshape(-1)
        true_i = true_hex_test[:, :, i].numpy().reshape(-1)
        results[col] = _metrics_for(pred_i, true_i)

    return results


def evaluate_hex_ceiling(idx_test, hexes, taz_to_idx,
                          taz_parquet_dir=TAZ_PARQUET_DIR, hex_parquet_dir=HEX_PARQUET_DIR):
    """The "perfect surrogate" ceiling: broadcast the *true* TAZ-level absolute
    accessibility (not a model's prediction) to every hex within that TAZ, and
    score against the true hex-resolution data.

    No model is involved at all - any gap from a perfect score is *pure*
    TAZ->hex resolution loss: the ceiling every TAZ-resolution surrogate is
    measured against, regardless of how good the model is. Same concept as
    the old repo's evaluate_perfect_surrogate.py / perfect_surrogate_ceiling.ipynb.

    Returns one {r2, mae, c_index, wape} dict per output channel (21 total:
    mode x POI category), pooled over every test scenario and every hex.
    """
    # true TAZ-level absolute accessibility for exactly the test scenarios -
    # the raw cumulative_accessibility_* columns are already absolute values,
    # no dry baseline needed here
    ds = load_dataset("parquet", data_files=os.path.join(taz_parquet_dir, "*.parquet"))
    ds = ds['train']
    ds.set_format(type='torch')
    full_taz = ds[:]
    true_taz = torch.stack([full_taz[c] for c in Y_COLS], dim=-1)  # [n_samples_total, 277, 21]
    true_taz_test = true_taz[idx_test]                              # [n_test, 277, 21]

    # broadcast each TAZ's true value to every hex within it
    taz_idx_per_hex = hexes['taz_zoneid'].map(taz_to_idx).to_numpy()  # [n_hex]
    ceiling_hex = true_taz_test[:, taz_idx_per_hex, :]                # [n_test, n_hex, 21]

    # true hex-resolution accessibility, for exactly the test scenarios
    true_hex = _load_true_hex(hex_parquet_dir)
    true_hex_test = true_hex[idx_test]

    results = {}
    for i, col in enumerate(Y_COLS):
        pred_i = ceiling_hex[:, :, i].numpy().reshape(-1)
        true_i = true_hex_test[:, :, i].numpy().reshape(-1)
        results[col] = _metrics_for(pred_i, true_i)

    return results
