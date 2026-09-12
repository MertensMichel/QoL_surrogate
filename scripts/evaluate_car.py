"""Entry point: computes TAZ-level evaluation metrics for a trained CAR
surrogate run (model_architecture.GCNResNet, from scripts/train_car.py) and saves
them into that run's own folder (eval_results.pkl), so notebooks can just load
the result instead of recomputing it - same convention as scripts/evaluate.py
and scripts/evaluate_baseline.py use for the old 21-channel models.

Unlike scripts/evaluate.py, there's no separate hex-resolution/dry-baseline/
static-feature loading here - the car checkpoint bundles all of that as
buffers already (see qol_surrogate.model_architecture), so rebuilding the test set
only needs the dynamic dataset + the fixed scenario split.

Usage:
    python scripts/evaluate_car.py                    # evaluates the most recently trained car run
    python scripts/evaluate_car.py <path/to/run_dir>  # evaluates a specific run
"""

import glob
import os
import pickle
import sys

from torch_geometric.loader import DataLoader

from qol_surrogate.data_car import (
    DRY_BASELINE_DIR,
    STATIC_FEATURES_DIR,
    TAZ_PARQUET_DIR,
    build_data_list,
    load_dataset_tensors,
    normalize,
    split_dataset,
)
from qol_surrogate.evaluate_car import evaluate_car, evaluate_per_taz_car
from qol_surrogate.inference import load_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def find_latest_run(models_dir=MODELS_DIR):
    run_dirs = sorted(d for d in glob.glob(os.path.join(models_dir, "gcn_resnet_car_*")) if os.path.isdir(d))
    return run_dirs[-1]


def build_test_loader(model):
    """Rebuild the same test split scripts/train_car.py's build_data() produced
    (split_dataset's random_state is fixed -> same idx_test), normalized with this
    model's own saved stats. Graph/static features aren't needed here - they're
    already on the model as buffers, reused directly by build_data_list below."""
    with open(os.path.join(STATIC_FEATURES_DIR, "graph.pkl"), "rb") as f:
        graph = pickle.load(f)
    edge_index, edge_weight = graph["edge_index"], graph["edge_weight"]

    with open(os.path.join(STATIC_FEATURES_DIR, "static_features.pkl"), "rb") as f:
        static_features_dict = pickle.load(f)

    x, y, _ = load_dataset_tensors(static_features_dict, taz_parquet_dir=TAZ_PARQUET_DIR,
                                    dry_baseline_dir=DRY_BASELINE_DIR)
    _, _, x_test, _, _, y_test, _, _, idx_test = split_dataset(x, y)

    x_test = normalize(x_test, model.x_mean, model.x_std)
    y_test = normalize(y_test, model.y_mean, model.y_std)

    test_data = build_data_list(x_test, y_test, edge_index, edge_weight)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)  # must stay unshuffled

    return test_loader, idx_test


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Evaluating run: {run_dir}")

    model, taz_ids = load_model(os.path.join(run_dir, "model.pt"))

    print("Building test set...")
    test_loader, idx_test = build_test_loader(model)
    print(f"{len(test_loader.dataset)} test scenarios")

    print("Computing TAZ-level metrics...")
    taz_metrics = evaluate_car(model, test_loader)

    print("Computing per-TAZ metrics...")
    per_taz_metrics = evaluate_per_taz_car(model, test_loader, taz_ids)

    results = {"taz": taz_metrics, "taz_per_taz": per_taz_metrics}

    results_path = os.path.join(run_dir, "eval_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved evaluation to {results_path}")


if __name__ == "__main__":
    main()
