"""[BICYCLE] Entry point: loads the pre-cached TAZ-level BICYCLE dataset, builds the model, trains,
and saves a fully self-contained checkpoint (weights + norm stats + static features +
graph + dry baseline, all as buffers - see qol_surrogate.model_architecture.GCNResNet).

Unlike scripts/train.py, this does NOT lazily aggregate raw data on first run - it
assumes scripts/agg_bicycle.py has already been run to produce the cached dynamic
parquet and dry-baseline parquet. static_features.pkl/graph.pkl are NOT mode-specific
(the street-network/POI/geometry static features are identical regardless of transport
mode) and are reused as-is from scripts/agg_and_static_features.py's onfoot output.
"""

import os
import pickle
from datetime import datetime

import torch

from qol_surrogate.model_architecture import GCNResNet
from qol_surrogate.train import train

from qol_surrogate.data_bicycle import (
    build_data_list,
    build_loaders,
    build_taz_to_idx,
    compute_normalization_stats,
    load_dataset_tensors,
    load_edge_taz_ids,
    normalize,
    split_dataset,
    stack_static_features,
    DRY_BASELINE_DIR,
    STATIC_FEATURE_COLS,
    STATIC_FEATURES_DIR,
    TAZ_PARQUET_DIR,
)


def build_data():
    """Load the pre-cached static features + graph + dynamic/dry-baseline parquet and
    build train/val/test loaders, exactly like train_onfoot.py's build_data() but
    sourced from scripts/agg_bicycle.py's cached output.
    """
    with open(os.path.join(STATIC_FEATURES_DIR, "static_features.pkl"), "rb") as f:
        static_features_dict = pickle.load(f)

    with open(os.path.join(STATIC_FEATURES_DIR, "graph.pkl"), "rb") as f:
        graph = pickle.load(f)
    edge_index, edge_weight, taz_ids = graph["edge_index"], graph["edge_weight"], graph["taz_ids"]

    edge_taz_ids = load_edge_taz_ids(build_taz_to_idx(taz_ids))

    x, y, y_dry = load_dataset_tensors(static_features_dict, taz_parquet_dir=TAZ_PARQUET_DIR,
                                        dry_baseline_dir=DRY_BASELINE_DIR)
    x_train, x_val, x_test, y_train, y_val, y_test, _, _, idx_test = split_dataset(x, y)

    x_mean, x_std, y_mean, y_std = compute_normalization_stats(x_train, y_train)

    x_train = normalize(x_train, x_mean, x_std)
    x_val   = normalize(x_val, x_mean, x_std)
    x_test  = normalize(x_test, x_mean, x_std)

    y_train = normalize(y_train, y_mean, y_std)
    y_val   = normalize(y_val, y_mean, y_std)
    y_test  = normalize(y_test, y_mean, y_std)

    train_data = build_data_list(x_train, y_train, edge_index, edge_weight)
    val_data   = build_data_list(x_val, y_val, edge_index, edge_weight)
    test_data  = build_data_list(x_test, y_test, edge_index, edge_weight)

    train_loader, val_loader, test_loader = build_loaders(train_data, val_data, test_data)

    static_features = stack_static_features(static_features_dict)  # [277, n_static], raw/unnormalized
    dry_baseline = y_dry.squeeze(0)  # [1, 277, 7] -> [277, 7], same shape as a model prediction

    return (train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std, idx_test,
            static_features, edge_index, edge_weight, dry_baseline, edge_taz_ids, taz_ids)


def build_model(x_mean, x_std, y_mean, y_std, static_features, edge_index, edge_weight, dry_baseline, edge_taz_ids):
    """Instantiate GCNResNet with the training-set normalization stats AND the
    scenario-independent static features/graph/dry baseline/edge_taz_ids wired in as
    buffers, so a saved checkpoint is fully self-contained for inference."""
    model = GCNResNet(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        n_layers=N_LAYERS,
        dropout_rate=DROPOUT_RATE,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        static_features=static_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        dry_baseline=dry_baseline,
        edge_taz_ids=edge_taz_ids,
    )
    return model


def save_run(model, train_losses, val_losses, idx_test, taz_ids, models_dir):
    """Save the model weights + full per-epoch loss history into a dedicated,
    timestamped run folder - same convention as scripts/train_onfoot.py's save_run()."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(models_dir, f"gcn_resnet_bicycle_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_path = os.path.join(run_dir, "model.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "idx_test": idx_test,  # which original scenarios were held out
        "static_feature_cols": STATIC_FEATURE_COLS,  # column order for the static_features
                                                       # buffer - needed to interpret it later
        "taz_ids": taz_ids,  # real taz_zoneid for each position (0..num_taz-1) in every
                              # buffer - without this, model output is unlabeled positions
        "hyperparameters": {
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "hidden_channels": HIDDEN_CHANNELS,
            "n_layers": N_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            # shapes needed to build placeholder buffers before load_state_dict(),
            # same pattern scripts/evaluate.py uses for x_mean/y_mean today
            "num_taz": model.static_features.shape[0],
            "n_static_features": model.static_features.shape[1],
            "n_edges": model.graph_edge_index.shape[1],
            "n_edge_taz_ids": model.edge_taz_ids.shape[0],
        },
    }, checkpoint_path)

    return run_dir


# x/y feature counts are fixed by the data (5 dynamic water-depth stats + 24 static
# features in, 7 BICYCLE POI-category accessibility-deviation outputs) - not tunable.
IN_CHANNELS = 29
OUT_CHANNELS = 7

# Same starting-point hyperparameters as scripts/train_onfoot.py - not yet tuned for
# this dataset specifically (4.5k BICYCLE scenarios vs. onfoot's ~22k).
HIDDEN_CHANNELS = 256
N_LAYERS = 4
DROPOUT_RATE = 0.1
NUM_EPOCHS = 500
PATIENCE = 12  # stop early if validation loss hasn't improved in this many epochs

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def main():
    (train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std, idx_test,
     static_features, edge_index, edge_weight, dry_baseline, edge_taz_ids, taz_ids) = build_data()

    model = build_model(x_mean, x_std, y_mean, y_std, static_features, edge_index, edge_weight,
                         dry_baseline, edge_taz_ids)

    model, train_losses, val_losses = train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE)

    # NOTE: no BICYCLE-specific evaluate function exists yet (mirrors evaluate_onfoot.py's
    # relationship to ON_FOOT - qol_surrogate.evaluate.evaluate() is hardcoded to the old
    # 3-mode/21-channel Y_COLS and would KeyError here). Test-set R2/MAE/WAPE reporting
    # needs an evaluate_bicycle.py before this can report metrics beyond train/val loss curves.
    run_dir = save_run(model, train_losses, val_losses, idx_test, taz_ids, models_dir=MODELS_DIR)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
