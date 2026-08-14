"""Entry point: trains the Boosted Forest baseline (see
qol_surrogate.baseline for the architecture) and saves it into its own
models/baseline/ folder, kept separate from the GCN's models/gcn_resnet_*
runs.

Uses the exact same data pipeline, target definition, and train/val/test
split (same random_state -> same idx_test) as scripts/train.py, so the two
are directly comparable on identical held-out scenarios. The only thing that
changes is the architecture: no graph, no normalization - see baseline.py's
module docstring for what's actually different.
"""

import os
import pickle
from datetime import datetime

from qol_surrogate.baseline import build_channel_targets, evaluate_baseline, flatten_x, train_baseline
from qol_surrogate.data import (
    DRY_BASELINE_DIR,
    DRY_BASELINE_DIR_NO_19,
    TAZ_PARQUET_DIR,
    TAZ_PARQUET_DIR_NO_19,
    ZONES_FILE,
    aggregate_to_taz,
    build_taz_geometries,
    create_polygon_metrics,
    get_taz_ids,
    load_dataset_tensors,
    load_hexes,
    save_pseudo_dry_baseline,
    save_taz_aggregated_dataset,
    split_dataset,
)


def build_data(include_file_19=True):
    """Same TAZ-level x/y (5 inputs, 21 outputs per node) and the same
    scenario-level split as scripts/train.py's build_data() - just without
    building the graph (no edge_index/edge_weight needed here)."""
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)

    taz_parquet_dir = TAZ_PARQUET_DIR if include_file_19 else TAZ_PARQUET_DIR_NO_19
    dry_baseline_dir = DRY_BASELINE_DIR if include_file_19 else DRY_BASELINE_DIR_NO_19

    if not os.path.exists(taz_parquet_dir) or not os.path.exists(dry_baseline_dir):
        taz_agg = aggregate_to_taz(hexes, taz_ids, include_file_19=include_file_19)

        if not os.path.exists(taz_parquet_dir):
            save_taz_aggregated_dataset(taz_agg, taz_ids, output_dir=taz_parquet_dir)

        if not os.path.exists(dry_baseline_dir):
            save_pseudo_dry_baseline(taz_agg, include_file_19=include_file_19)

    tazes = build_taz_geometries(hexes)
    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    x, y = load_dataset_tensors(area, perimeter, taz_parquet_dir=taz_parquet_dir, dry_baseline_dir=dry_baseline_dir)
    x_train, x_val, x_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = split_dataset(x, y)

    return x_train, x_val, x_test, y_train, y_val, y_test, idx_test, taz_ids, dry_baseline_dir


def save_run(models, metrics, idx_test, include_file_19, models_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(models_dir, f"gbm_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump({
            "models": models,          # {channel_name: fitted MultiOutputRegressor}
            "metrics": metrics,
            "idx_test": idx_test,
            "include_file_19": include_file_19,
        }, f)

    return run_dir


include_file_19 = False  # match the GCN's current best run, for a fair comparison

# Where baseline runs get saved - kept separate from models/gcn_resnet_*.
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "baseline")


def main():
    x_train, x_val, x_test, y_train, y_val, y_test, idx_test, taz_ids, dry_baseline_dir = build_data(include_file_19)

    x_train_flat = flatten_x(x_train)
    x_test_flat = flatten_x(x_test)
    y_train_channels = build_channel_targets(y_train)

    print(f"Training {len(y_train_channels)} channel models "
          f"({x_train_flat.shape[0]} scenarios x {x_train_flat.shape[1]} features -> 277 outputs each)...")
    models = train_baseline(x_train_flat, y_train_channels)

    metrics = evaluate_baseline(models, x_test_flat, y_test, dry_baseline_dir=dry_baseline_dir)
    print(metrics)

    run_dir = save_run(models, metrics, idx_test, include_file_19, models_dir=MODELS_DIR)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
