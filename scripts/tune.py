"""Hyperparameter tuning for GCNResNet via Optuna (TPE sampler + median pruning).

Standalone script - does not import from or modify scripts/train.py or
scripts/evaluate.py. The TAZ graph/tensors are built once (independent of any
hyperparameter) and reused across all trials; each trial gets its own model
and a short training loop (kept local to this script, not
qol_surrogate.train.train(), so per-epoch validation loss can be reported to
Optuna for pruning - train() doesn't expose a hook for that).

Tunes: hidden_channels, n_layers, dropout_rate, lr, batch_size.
Fixed: everything else matches scripts/train.py's current settings
(include_file_19=False, Adam, ReduceLROnPlateau(factor=0.5, patience=10), MSE loss).

Usage:
    python scripts/tune.py --n-trials 50
    python scripts/tune.py --n-trials 50 --timeout 7200
"""

import argparse
import json
import os

import optuna
import torch

from qol_surrogate.model import GCNResNet

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
    TAZ_PARQUET_DIR,
    TAZ_PARQUET_DIR_NO_19,
    ZONES_FILE,
)

# x/y feature counts are fixed by the data, not tunable.
IN_CHANNELS = 5
OUT_CHANNELS = 21

INCLUDE_FILE_19 = False  # matches scripts/train.py's current setting

MAX_EPOCHS = 60
PATIENCE = 8  # early-stopping patience within a trial - shorter than scripts/train.py's
               # 500/25 since many trials need to run; the best trial can be re-trained
               # longer afterward with scripts/train.py using its winning hyperparameters

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "tuning")


def build_data(include_file_19=INCLUDE_FILE_19):
    """Build the TAZ graph + normalized train/val Data lists once, shared by
    every trial. Mirrors scripts/train.py's build_data(), but stops short of
    building loaders since batch_size is itself a tuned hyperparameter."""
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

    tazes = build_taz_geometries(hexes)
    edge_index, edge_weight = create_edge_index(tazes, taz_to_idx)

    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    x, y = load_dataset_tensors(area, perimeter, taz_parquet_dir=taz_parquet_dir, dry_baseline_dir=dry_baseline_dir)
    x_train, x_val, _, y_train, y_val, _, _, _, _ = split_dataset(x, y)

    x_mean, x_std, y_mean, y_std = compute_normalization_stats(x_train, y_train)

    x_train = normalize(x_train, x_mean, x_std)
    x_val   = normalize(x_val, x_mean, x_std)
    y_train = normalize(y_train, y_mean, y_std)
    y_val   = normalize(y_val, y_mean, y_std)

    train_data = build_data_list(x_train, y_train, edge_index, edge_weight)
    val_data   = build_data_list(x_val, y_val, edge_index, edge_weight)

    return train_data, val_data, x_mean, x_std, y_mean, y_std


def objective(trial, train_data, val_data, x_mean, x_std, y_mean, y_std):
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    train_loader, val_loader, _ = build_loaders(train_data, val_data, val_data, batch_size=batch_size)

    model = GCNResNet(
        in_channels=IN_CHANNELS, hidden_channels=hidden_channels, out_channels=OUT_CHANNELS,
        n_layers=n_layers, dropout_rate=dropout_rate,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.edge_weight)
                val_loss += criterion(out, batch.y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="wall-clock budget in seconds")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Building dataset (once, shared across all trials)...")
    train_data, val_data, x_mean, x_std, y_mean, y_std = build_data()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=13),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
        storage=f"sqlite:///{os.path.join(RESULTS_DIR, 'tuning.db')}",
        study_name="qol_surrogate_gcn",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, x_mean, x_std, y_mean, y_std),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print(f"Finished: {len(study.trials)} trials total.")
    print(f"Best value (val MSE): {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    best_path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(best_path, "w") as f:
        json.dump({"value": study.best_trial.value, "max_train_epochs": MAX_EPOCHS, "params": study.best_trial.params}, f, indent=2)
    print(f"Saved best params to {best_path}")
    print("To train a full run with these, copy them into the hyperparameter constants "
          "in scripts/train.py.")


if __name__ == "__main__":
    main()
