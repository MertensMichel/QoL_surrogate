"""Entry point: computes TAZ/hex/ceiling evaluation metrics for a trained run
and saves them into that run's own folder (eval_results.pkl), so
notebooks/evaluate_hex.ipynb can just load the result instead of recomputing
it. Meant to be run standalone (e.g. inside tmux), not from a notebook cell -
the hex-resolution computation is heavy enough to be worth keeping off the
notebook's kernel.

Usage:
    python scripts/evaluate.py                  # evaluates the most recently trained run
    python scripts/evaluate.py <path/to/run_dir>  # evaluates a specific run
"""

import glob
import os
import pickle
import sys

import torch
from torch_geometric.loader import DataLoader

from qol_surrogate.data import (
    ZONES_FILE, TAZ_PARQUET_DIR, TAZ_PARQUET_DIR_NO_19, DRY_BASELINE_DIR, DRY_BASELINE_DIR_NO_19,
    load_hexes, get_taz_ids, build_taz_to_idx, build_taz_geometries,
    create_edge_index, create_polygon_metrics, load_dataset_tensors, build_data_list,
)
from qol_surrogate.evaluate import evaluate, evaluate_hex_level, evaluate_hex_level_per_taz, evaluate_hex_ceiling
from qol_surrogate.model import GCNResNet

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def find_latest_run(models_dir=MODELS_DIR):
    run_dirs = sorted(d for d in glob.glob(os.path.join(models_dir, "gcn_resnet_*")) if os.path.isdir(d))
    return run_dirs[-1]


def load_model(run_dir):
    checkpoint = torch.load(os.path.join(run_dir, "model.pt"), weights_only=False)
    hp = checkpoint["hyperparameters"]
    idx_test = checkpoint["idx_test"]
    # older checkpoints saved before this flag existed - assume the with-19 default they all used
    include_file_19 = checkpoint.get("include_file_19", True)

    # placeholder buffers - load_state_dict overwrites them with the real saved values below
    model = GCNResNet(
        in_channels=hp["in_channels"], hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"], n_layers=hp["n_layers"], dropout_rate=hp["dropout_rate"],
        x_mean=torch.zeros(hp["in_channels"]), x_std=torch.ones(hp["in_channels"]),
        y_mean=torch.zeros(hp["out_channels"]), y_std=torch.ones(hp["out_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, idx_test, include_file_19


def build_test_loader(model, idx_test, include_file_19):
    # rebuild the graph + TAZ ordering exactly as build_data() does in scripts/train.py
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)
    taz_to_idx = build_taz_to_idx(taz_ids)

    tazes = build_taz_geometries(hexes)
    edge_index, edge_weight = create_edge_index(tazes, taz_to_idx)
    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    taz_parquet_dir = TAZ_PARQUET_DIR if include_file_19 else TAZ_PARQUET_DIR_NO_19
    dry_baseline_dir = DRY_BASELINE_DIR if include_file_19 else DRY_BASELINE_DIR_NO_19
    x, y = load_dataset_tensors(area, perimeter, taz_parquet_dir=taz_parquet_dir, dry_baseline_dir=dry_baseline_dir)

    # select exactly the test-set scenarios THIS run held out, normalize with this model's own stats
    x_test = (x[idx_test] - model.x_mean) / model.x_std
    y_test = (y[idx_test] - model.y_mean) / model.y_std

    test_data = build_data_list(x_test, y_test, edge_index, edge_weight)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)  # must stay unshuffled - order must match idx_test

    return test_loader, hexes, taz_to_idx, taz_ids, dry_baseline_dir, taz_parquet_dir


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Evaluating run: {run_dir}")

    model, idx_test, include_file_19 = load_model(run_dir)
    print(f"include_file_19={include_file_19}")
    test_loader, hexes, taz_to_idx, taz_ids, dry_baseline_dir, taz_parquet_dir = build_test_loader(
        model, idx_test, include_file_19
    )

    print("Computing TAZ-level metrics...")
    taz_metrics = evaluate(model, test_loader, dry_baseline_dir=dry_baseline_dir)

    print("Computing hex-level metrics...")
    hex_metrics = evaluate_hex_level(model, test_loader, idx_test, hexes, taz_to_idx,
                                      include_file_19=include_file_19, dry_baseline_dir=dry_baseline_dir)

    print("Computing hex-level metrics per TAZ...")
    hex_per_taz_metrics = evaluate_hex_level_per_taz(model, test_loader, idx_test, hexes, taz_to_idx, taz_ids,
                                                      include_file_19=include_file_19, dry_baseline_dir=dry_baseline_dir)

    print("Computing ceiling metrics (true TAZ -> hex broadcast, no model)...")
    ceiling_metrics = evaluate_hex_ceiling(idx_test, hexes, taz_to_idx,
                                            include_file_19=include_file_19, taz_parquet_dir=taz_parquet_dir)

    results = {"taz": taz_metrics, "hex": hex_metrics, "hex_per_taz": hex_per_taz_metrics, "ceiling": ceiling_metrics}

    results_path = os.path.join(run_dir, "eval_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved evaluation to {results_path}")


if __name__ == "__main__":
    main()
