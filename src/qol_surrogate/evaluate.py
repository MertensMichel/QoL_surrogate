"""Evaluation metrics (R², MAE, C-index, WAPE)."""

import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate(model, test_loader):
    """Evaluate the model on a test set, in original (un-normalized) units.

    Returns a dict of overall metrics computed across every TAZ node and
    every output channel (mode x POI category) pooled together.
    """
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            preds.append(out)
            targets.append(batch.y)

    preds = torch.cat(preds, dim=0)      # [n_test_scenarios * 277, 21]
    targets = torch.cat(targets, dim=0)

    # model + targets were trained/prepared in normalized units - convert back
    # to real accessibility-deviation units before computing interpretable metrics
    preds_real = preds * model.y_std + model.y_mean
    targets_real = targets * model.y_std + model.y_mean

    preds_np = preds_real.numpy().reshape(-1)
    targets_np = targets_real.numpy().reshape(-1)

    # WAPE: scale-independent aggregate error - unlike MAPE it doesn't blow up
    # near zero, which matters here since y is a deviation and often near zero
    wape = np.sum(np.abs(targets_np - preds_np)) / np.sum(np.abs(targets_np))

    return {
        "r2": r2_score(targets_np, preds_np),
        "mae": mean_absolute_error(targets_np, preds_np),
        "c_index": concordance_index(targets_np, preds_np),
        "wape": wape,
    }
