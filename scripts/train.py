"""Entry point: loads data, builds the model, trains, and evaluates.

Thin script - imports from qol_surrogate.*, defines main(), guarded by __main__.
"""

import os
from datetime import datetime

import torch

from qol_surrogate.model import GCNResNet
from qol_surrogate.train import train
from qol_surrogate.evaluate import evaluate

from qol_surrogate.data import (
    aggregate_to_taz,
    build_data_list,
    build_loaders,
    build_taz_geometries,
    build_taz_to_idx,
    compute_normalization_stats,
    create_edge_index,
    create_polygon_metrics,
    load_dataset_tensors,
    load_hexes,
    get_taz_ids,
    normalize,
    save_pseudo_dry_baseline,
    save_taz_aggregated_dataset,
    split_dataset,
    DRY_BASELINE_DIR,
    NETWORK_DIR,
    TAZ_PARQUET_DIR,
    ZONES_FILE,
)

def build_data():
    """Train a surrogate model on TAZ-level data."""
    # Load hexes and build TAZ mapping
    hexes = load_hexes(ZONES_FILE)

    taz_ids = get_taz_ids(hexes)
    taz_to_idx = build_taz_to_idx(taz_ids)

    if not os.path.exists(TAZ_PARQUET_DIR) or not os.path.exists(DRY_BASELINE_DIR):
        taz_agg = aggregate_to_taz(hexes, taz_ids)

        if not os.path.exists(TAZ_PARQUET_DIR):
            save_taz_aggregated_dataset(taz_agg, taz_ids, TAZ_PARQUET_DIR)

        if not os.path.exists(DRY_BASELINE_DIR):
            save_pseudo_dry_baseline(taz_agg, DRY_BASELINE_DIR)

    # Create Graph
    tazes = build_taz_geometries(hexes)
    edge_index, edge_weight = create_edge_index(tazes, taz_to_idx)

    # Load dataset tensors
    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    x, y = load_dataset_tensors(area, perimeter, taz_parquet_dir=TAZ_PARQUET_DIR, dry_baseline_dir=DRY_BASELINE_DIR)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y)

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

    return train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std


# x/y feature counts are fixed by the data (3 dynamic + 2 static water-depth/geometry
# features in, 21 mode x POI-category accessibility-deviation outputs) - not tunable.
IN_CHANNELS = 5
OUT_CHANNELS = 21

# Actual hyperparameters - starting points matching the old repo's GCN-POI-ResNet baseline.
HIDDEN_CHANNELS = 128
N_LAYERS = 6
DROPOUT_RATE = 0.1

# Where trained models get saved - one file per run, timestamped so runs never overwrite each other.
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def build_model(x_mean, x_std, y_mean, y_std):
    """Instantiate the GCNResNet model, with the training-set normalization
    stats wired in as buffers so they travel with the model afterward."""
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
    )
    return model


def save_run(model, train_losses, val_losses, metrics, models_dir=MODELS_DIR):
    """Save the model weights + full per-epoch loss history + test metrics to a
    single timestamped checkpoint file, so a later run can plot loss vs. epoch
    or reload the model without needing anything else alongside it."""
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(models_dir, f"gcn_resnet_{timestamp}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "metrics": metrics,
        "hyperparameters": {
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "hidden_channels": HIDDEN_CHANNELS,
            "n_layers": N_LAYERS,
            "dropout_rate": DROPOUT_RATE,
        },
    }, checkpoint_path)

    return checkpoint_path


def main():
    train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std = build_data()
    model = build_model(x_mean, x_std, y_mean, y_std)

    model, train_losses, val_losses = train(model, train_loader, val_loader)

    metrics = evaluate(model, test_loader)
    print(metrics)

    checkpoint_path = save_run(model, train_losses, val_losses, metrics)
    print(f"Saved run to {checkpoint_path}")


if __name__ == "__main__":
    main()
