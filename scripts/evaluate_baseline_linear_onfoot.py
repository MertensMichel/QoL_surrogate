"""[ONFOOT] Entry point: computes TAZ-level evaluation metrics for a trained ON_FOOT
Linear Regression baseline run and saves them into that run's own folder
(eval_results.pkl) - same shape as scripts/evaluate_onfoot.py /
scripts/evaluate_baseline_onfoot.py produce ({taz, taz_per_taz}), so all
three can be loaded and compared side-by-side without recomputing.

Usage:
    python scripts/evaluate_baseline_linear_onfoot.py                    # evaluates the most recently trained run
    python scripts/evaluate_baseline_linear_onfoot.py <path/to/run_dir>  # evaluates a specific run
"""

import glob
import os
import pickle
import sys

from qol_surrogate.baseline_linear_onfoot import (
    evaluate_baseline_linear, evaluate_per_taz_baseline_linear, flatten_x,
)
from qol_surrogate.data_onfoot import (
    DRY_BASELINE_DIR,
    STATIC_FEATURES_DIR,
    TAZ_PARQUET_DIR,
    load_dataset_tensors,
    split_dataset,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "baseline_linear_onfoot")


def find_latest_run(models_dir=MODELS_DIR):
    run_dirs = sorted(d for d in glob.glob(os.path.join(models_dir, "linear_onfoot_*")) if os.path.isdir(d))
    return run_dirs[-1]


def load_run(run_dir):
    with open(os.path.join(run_dir, "model.pkl"), "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint["model"], checkpoint["idx_test"], checkpoint["taz_ids"]


def build_test_set(idx_test):
    # rebuild the same TAZ ordering and x/y tensors build_data() in
    # scripts/train_baseline_linear_onfoot.py used - split_dataset() is
    # deterministic (fixed random_state), so this reproduces the exact same idx_test.
    with open(os.path.join(STATIC_FEATURES_DIR, "static_features.pkl"), "rb") as f:
        static_features_dict = pickle.load(f)

    x, y, y_dry = load_dataset_tensors(static_features_dict, taz_parquet_dir=TAZ_PARQUET_DIR,
                                        dry_baseline_dir=DRY_BASELINE_DIR)
    _, _, x_test, _, _, y_test, _, _, idx_test_check = split_dataset(x, y)
    assert list(idx_test_check) == list(idx_test), "split_dataset() didn't reproduce this run's idx_test"

    dry_baseline = y_dry.squeeze(0)

    return x_test, y_test, dry_baseline


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Evaluating run: {run_dir}")

    model, idx_test, taz_ids = load_run(run_dir)

    print("Building test set...")
    x_test, y_test, dry_baseline = build_test_set(idx_test)
    x_test_flat = flatten_x(x_test)

    print("Computing TAZ-level metrics...")
    taz_metrics = evaluate_baseline_linear(model, x_test_flat, y_test, dry_baseline)

    print("Computing per-TAZ metrics...")
    per_taz_metrics = evaluate_per_taz_baseline_linear(model, x_test_flat, y_test, taz_ids, dry_baseline)

    results = {"taz": taz_metrics, "taz_per_taz": per_taz_metrics}

    results_path = os.path.join(run_dir, "eval_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved evaluation to {results_path}")


if __name__ == "__main__":
    main()
