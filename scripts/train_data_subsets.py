"""[LEGACY: 3-mode/21-channel pipeline] Learning-curve experiment: trains on
increasing fractions of the (small, ~637-scenario) legacy training set to see
how accuracy scales with data size. Not applicable to the newer per-mode
(onfoot/car/bicycle) pipeline or its much larger datasets."""

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
    DRY_BASELINE_DIR_NO_19,
    NETWORK_DIR,
    TAZ_PARQUET_DIR,
    TAZ_PARQUET_DIR_NO_19,
    ZONES_FILE,
)

def build_data_subset(include_file_19=True, subset_size=None):
    """Train a surrogate model on TAZ-level data."""
    # Load hexes and build TAZ mapping
    hexes = load_hexes(ZONES_FILE)

    taz_ids = get_taz_ids(hexes)
    taz_to_idx = build_taz_to_idx(taz_ids)

    taz_parquet_dir = TAZ_PARQUET_DIR if include_file_19 else TAZ_PARQUET_DIR_NO_19
    dry_baseline_dir = DRY_BASELINE_DIR if include_file_19 else DRY_BASELINE_DIR_NO_19

    if not os.path.exists(taz_parquet_dir) or not os.path.exists(dry_baseline_dir):
        taz_agg = aggregate_to_taz(hexes, taz_ids, include_file_19=include_file_19)

        if not os.path.exists(taz_parquet_dir):
            save_taz_aggregated_dataset(taz_agg, taz_ids, output_dir=taz_parquet_dir)

        if not os.path.exists(dry_baseline_dir):
            save_pseudo_dry_baseline(taz_agg, include_file_19=include_file_19)

    # Create Graph
    tazes = build_taz_geometries(hexes)
    edge_index, edge_weight = create_edge_index(tazes, taz_to_idx)

    # Load dataset tensors
    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    x, y = load_dataset_tensors(area, perimeter, taz_parquet_dir=taz_parquet_dir, dry_baseline_dir=dry_baseline_dir)
    x_train, x_val, x_test, y_train, y_val, y_test, _, _, idx_test = split_dataset(x, y)

    perm = torch.randperm(x_train.shape[0], generator=torch.Generator().manual_seed(42))
    subset_perm = perm[:int(subset_size * len(perm))]

    x_train_subset = x_train[subset_perm]
    y_train_subset = y_train[subset_perm]

    x_mean, x_std, y_mean, y_std = compute_normalization_stats(x_train_subset, y_train_subset)

    x_train = normalize(x_train_subset, x_mean, x_std)
    x_val   = normalize(x_val, x_mean, x_std)
    x_test  = normalize(x_test, x_mean, x_std)

    y_train = normalize(y_train_subset, y_mean, y_std)
    y_val   = normalize(y_val, y_mean, y_std)
    y_test  = normalize(y_test, y_mean, y_std)

    train_data = build_data_list(x_train, y_train, edge_index, edge_weight)
    val_data   = build_data_list(x_val, y_val, edge_index, edge_weight)
    test_data  = build_data_list(x_test, y_test, edge_index, edge_weight)


    train_loader, val_loader, test_loader = build_loaders(train_data, val_data, test_data)

    return train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std, idx_test


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


def save_run(model, train_losses, val_losses, metrics, idx_test, include_file_19, subset_size, models_dir):
    """Save the model weights + full per-epoch loss history + test metrics into
    a dedicated folder for this run (one folder per run, timestamped so runs
    never collide) - other artifacts for this run (e.g. hex-level evaluation
    results) get saved alongside model.pt in that same folder later.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(models_dir, f"gcn_resnet_{subset_size}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_path = os.path.join(run_dir, "model.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "metrics": metrics,
        "idx_test": idx_test,  # which original scenarios were held out - needed to
                                # go back to hex-resolution ground truth for this run
        "include_file_19": include_file_19,  # which dataset variant this run was trained on -
                                              # scripts/evaluate.py needs this to load matching data
        "hyperparameters": {
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "hidden_channels": HIDDEN_CHANNELS,
            "n_layers": N_LAYERS,
            "dropout_rate": DROPOUT_RATE,
        },
    }, checkpoint_path)

    return run_dir



# x/y feature counts are fixed by the data (3 dynamic + 2 static water-depth/geometry
# features in, 21 mode x POI-category accessibility-deviation outputs) - not tunable.
IN_CHANNELS = 5
OUT_CHANNELS = 21

include_file_19 = False  # whether to include the Copenhagen_19.parquet file in the TAZ-level dataset

# Actual hyperparameters - starting points matching the old repo's GCN-POI-ResNet baseline.
HIDDEN_CHANNELS = 256
N_LAYERS = 4
DROPOUT_RATE = 0.1
NUM_EPOCHS = 500
PATIENCE = 25  # stop early if validation loss hasn't improved in this many epochs

# Where trained models get saved - one file per run, timestamped so runs never overwrite each other.
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/data_subsets/")


def main():
    for subset_size in [0.1, 0.25, 0.5, 0.75, 0.90, 1.0]:
        SUBSET_SIZE = subset_size

        train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std, idx_test = build_data_subset(include_file_19, subset_size=SUBSET_SIZE)
        model = build_model(x_mean, x_std, y_mean, y_std)

        model, train_losses, val_losses = train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE)

        dry_baseline_dir = DRY_BASELINE_DIR if include_file_19 else DRY_BASELINE_DIR_NO_19
        metrics = evaluate(model, test_loader, dry_baseline_dir=dry_baseline_dir)
        print(metrics)

        run_dir = save_run(model, train_losses, val_losses, metrics, idx_test, include_file_19, SUBSET_SIZE, models_dir=MODELS_DIR)
        print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()

