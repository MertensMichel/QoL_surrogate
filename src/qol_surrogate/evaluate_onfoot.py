"""TAZ-level evaluation metrics (R2, MAE, C-index, WAPE) for the ON_FOOT-only
surrogate (model_architecture.GCNResNet). Kept separate from qol_surrogate.evaluate,
which is hardcoded to the old 3-mode/21-channel Y_COLS and would mis-index
against this model's 7-channel (POI-category-only) output.

Unlike the old evaluate(), the dry baseline is read straight off the model's
own buffer (model.dry_baseline) instead of a separate load - the onfoot
checkpoint bundles it, so there's nothing else to fetch.
"""

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error, r2_score

# Duplicated from qol_surrogate.data_onfoot.POI_CATEGORIES rather than imported -
# order must match the model's output channel order
POI_CATEGORIES = ['cultural', 'education', 'green_space', 'health',
                   'public_spaces', 'public_transportation', 'sports']


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


def _predict_absolute(model, test_loader):
    """Shared forward pass + deviation->absolute reconstruction used by both
    evaluate_onfoot() and evaluate_per_taz_onfoot()."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            preds.append(out)
            targets.append(batch.y)

    # PyG batching concatenates all graphs' nodes into one flat axis - reshape
    # back into per-scenario structure before indexing per channel/TAZ
    n_scenarios = len(test_loader.dataset)
    preds = torch.cat(preds, dim=0).reshape(n_scenarios, -1, preds[0].shape[-1])
    targets = torch.cat(targets, dim=0).reshape(n_scenarios, -1, targets[0].shape[-1])

    preds_deviation = preds * model.y_std + model.y_mean
    targets_deviation = targets * model.y_std + model.y_mean

    preds_absolute = preds_deviation + model.dry_baseline
    targets_absolute = targets_deviation + model.dry_baseline

    return preds_deviation, targets_deviation, preds_absolute, targets_absolute


def evaluate_onfoot(model, test_loader):
    """Evaluate the ON_FOOT surrogate on a test set, in original (un-normalized)
    units - both deviation (what it's trained to predict) and absolute
    (deviation + dry baseline, the real-world-interpretable number).

    Returns {"deviation": {"overall": ..., "per_channel": ...},
             "absolute":  {"overall": ..., "per_channel": ...}}
    """
    preds_deviation, targets_deviation, preds_absolute, targets_absolute = _predict_absolute(model, test_loader)

    return {
        "deviation": _summarize(preds_deviation, targets_deviation),
        "absolute": _summarize(preds_absolute, targets_absolute),
    }


def evaluate_per_taz_onfoot(model, test_loader, taz_ids):
    """{r2, mae, c_index, wape} (absolute accessibility) *per TAZ zone*, pooled
    across test scenarios and the 7 POI categories - the onfoot analogue of
    evaluate.evaluate_per_taz(), minus the per-mode split (onfoot is the only
    mode here).

    Returns {metric: DataFrame(taz_id -> value, column "ON_FOOT")} - kept as a
    single-column DataFrame (not a Series) so it plugs directly into the same
    tazes.join(...) map-plotting pattern the old per-mode notebook cells use.
    """
    _, _, preds_absolute, targets_absolute = _predict_absolute(model, test_loader)

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
