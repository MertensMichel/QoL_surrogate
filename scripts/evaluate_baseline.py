"""Entry point: computes TAZ/hex/ceiling evaluation metrics for a trained
Boosted Forest baseline run and saves them into that run's own folder
(eval_results.pkl) - same shape as scripts/evaluate.py produces for the GCN
({taz, hex, hex_per_taz, ceiling}), so both can be loaded and compared
side-by-side in notebooks/evaluate_hex.ipynb.

Usage:
    python scripts/evaluate_baseline.py                    # evaluates the most recently trained baseline run
    python scripts/evaluate_baseline.py <path/to/run_dir>  # evaluates a specific run
"""

import glob
import os
import pickle
import sys

from qol_surrogate.baseline import (
    evaluate_baseline,
    evaluate_baseline_hex_level,
    evaluate_baseline_hex_level_per_taz,
    evaluate_baseline_per_taz,
    flatten_x,
)
from qol_surrogate.data import (
    DRY_BASELINE_DIR,
    DRY_BASELINE_DIR_NO_19,
    TAZ_PARQUET_DIR,
    TAZ_PARQUET_DIR_NO_19,
    ZONES_FILE,
    build_taz_geometries,
    build_taz_to_idx,
    create_polygon_metrics,
    get_taz_ids,
    load_dataset_tensors,
    load_hexes,
    split_dataset,
)
from qol_surrogate.evaluate import evaluate_hex_ceiling

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "baseline")


def find_latest_run(models_dir=MODELS_DIR):
    run_dirs = sorted(d for d in glob.glob(os.path.join(models_dir, "gbm_*")) if os.path.isdir(d))
    return run_dirs[-1]


def load_run(run_dir):
    with open(os.path.join(run_dir, "model.pkl"), "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint["models"], checkpoint["idx_test"], checkpoint["include_file_19"]


def build_test_set(idx_test, include_file_19):
    # rebuild the same TAZ ordering and x/y tensors build_data() in
    # scripts/train_baseline.py used - split_dataset() is deterministic
    # (fixed random_state), so this reproduces the exact same idx_test.
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)
    taz_to_idx = build_taz_to_idx(taz_ids)

    tazes = build_taz_geometries(hexes)
    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    taz_parquet_dir = TAZ_PARQUET_DIR if include_file_19 else TAZ_PARQUET_DIR_NO_19
    dry_baseline_dir = DRY_BASELINE_DIR if include_file_19 else DRY_BASELINE_DIR_NO_19
    x, y = load_dataset_tensors(area, perimeter, taz_parquet_dir=taz_parquet_dir, dry_baseline_dir=dry_baseline_dir)

    _, _, x_test, _, _, y_test, _, _, idx_test_check = split_dataset(x, y)
    assert list(idx_test_check) == list(idx_test), "split_dataset() didn't reproduce this run's idx_test"

    return x_test, y_test, hexes, taz_to_idx, taz_ids, dry_baseline_dir, taz_parquet_dir


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Evaluating baseline run: {run_dir}")

    models, idx_test, include_file_19 = load_run(run_dir)
    print(f"include_file_19={include_file_19}")

    x_test, y_test, hexes, taz_to_idx, taz_ids, dry_baseline_dir, taz_parquet_dir = build_test_set(
        idx_test, include_file_19
    )
    x_test_flat = flatten_x(x_test)

    print("Computing TAZ-level metrics...")
    taz_metrics = evaluate_baseline(models, x_test_flat, y_test, dry_baseline_dir=dry_baseline_dir)

    print("Computing per-TAZ metrics...")
    per_taz_metrics = evaluate_baseline_per_taz(models, x_test_flat, y_test, taz_ids, dry_baseline_dir=dry_baseline_dir)

    print("Computing hex-level metrics...")
    hex_metrics = evaluate_baseline_hex_level(models, x_test_flat, idx_test, hexes, taz_to_idx,
                                               include_file_19=include_file_19, dry_baseline_dir=dry_baseline_dir)

    print("Computing hex-level metrics per TAZ...")
    hex_per_taz_metrics = evaluate_baseline_hex_level_per_taz(models, x_test_flat, idx_test, hexes, taz_to_idx, taz_ids,
                                                                include_file_19=include_file_19, dry_baseline_dir=dry_baseline_dir)

    print("Computing ceiling metrics (true TAZ -> hex broadcast, no model)...")
    ceiling_metrics = evaluate_hex_ceiling(idx_test, hexes, taz_to_idx,
                                            include_file_19=include_file_19, taz_parquet_dir=taz_parquet_dir)

    results = {
        "taz": taz_metrics,
        "taz_per_taz": per_taz_metrics,
        "hex": hex_metrics,
        "hex_per_taz": hex_per_taz_metrics,
        "ceiling": ceiling_metrics,
    }

    results_path = os.path.join(run_dir, "eval_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved evaluation to {results_path}")


if __name__ == "__main__":
    main()
