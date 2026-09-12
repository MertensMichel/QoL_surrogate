"""[ONFOOT] Entry point: trains the ON_FOOT Multiple Linear Regression baseline (see
qol_surrogate.baseline_linear_onfoot for the architecture) and saves it into
its own models/baseline_linear_onfoot/ folder, kept separate from the GCN's
models/gcn_resnet_onfoot_* runs and from models/baseline_onfoot/ (Boosted
Forest).

Uses the exact same cached data (scripts/agg_and_static_features.py's output)
and the same train/val/test split (same random_state -> same idx_test) as
scripts/train_onfoot.py and scripts/train_baseline_onfoot.py, so all three
are directly comparable on identical held-out scenarios.
"""

import os
import pickle
import time
from datetime import datetime

from qol_surrogate.baseline_linear_onfoot import (
    build_channel_targets, evaluate_baseline_linear, flatten_x, train_baseline_linear,
)
from qol_surrogate.data_onfoot import (
    DRY_BASELINE_DIR,
    STATIC_FEATURES_DIR,
    TAZ_PARQUET_DIR,
    load_dataset_tensors,
    split_dataset,
)


def build_data():
    """Same TAZ-level x/y (29 inputs, 7 outputs per node) and the same
    scenario-level split as scripts/train_onfoot.py's / train_baseline_onfoot.py's
    build_data() - just without building the graph (no edge_index/edge_weight
    needed here)."""
    with open(os.path.join(STATIC_FEATURES_DIR, "static_features.pkl"), "rb") as f:
        static_features_dict = pickle.load(f)

    with open(os.path.join(STATIC_FEATURES_DIR, "graph.pkl"), "rb") as f:
        graph = pickle.load(f)
    taz_ids = graph["taz_ids"]

    x, y, y_dry = load_dataset_tensors(static_features_dict, taz_parquet_dir=TAZ_PARQUET_DIR,
                                        dry_baseline_dir=DRY_BASELINE_DIR)
    x_train, x_val, x_test, y_train, y_val, y_test, _, _, idx_test = split_dataset(x, y)

    dry_baseline = y_dry.squeeze(0)  # [1, 277, 7] -> [277, 7], broadcasts against any [n_scenarios, 277, 7]

    return x_train, x_val, x_test, y_train, y_val, y_test, idx_test, taz_ids, dry_baseline


def save_run(model, metrics, idx_test, taz_ids, models_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(models_dir, f"linear_onfoot_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump({
            "model": model,          # single sklearn LinearRegression, all 7*277 targets
            "metrics": metrics,
            "idx_test": idx_test,    # which original scenarios were held out
            "taz_ids": taz_ids,      # real taz_zoneid for each position (0..num_taz-1)
        }, f)

    return run_dir


# Where baseline runs get saved - kept separate from models/gcn_resnet_onfoot_* and models/baseline_onfoot/.
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "baseline_linear_onfoot")


def main():
    x_train, x_val, x_test, y_train, y_val, y_test, idx_test, taz_ids, dry_baseline = build_data()

    x_train_flat = flatten_x(x_train)
    x_test_flat = flatten_x(x_test)
    y_train_channels = build_channel_targets(y_train)

    print(f"Fitting one shared least-squares solve "
          f"({x_train_flat.shape[0]} scenarios x {x_train_flat.shape[1]} features -> "
          f"{len(taz_ids) * len(y_train_channels)} targets)...", flush=True)
    start = time.time()
    model = train_baseline_linear(x_train_flat, y_train_channels)
    print(f"Done, {time.time() - start:.1f}s", flush=True)

    print("Evaluating on the test set...", flush=True)
    metrics = evaluate_baseline_linear(model, x_test_flat, y_test, dry_baseline)
    print(metrics)

    run_dir = save_run(model, metrics, idx_test, taz_ids, models_dir=MODELS_DIR)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
