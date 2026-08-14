"""Entry point: compares the GCN and the Boosted Forest baseline on identical
held-out scenarios, TAZ-resolution only (model prediction vs. true TAZ value -
no hex-resolution data needed, so this is fast for both models).

For each of the three metrics (R², WAPE, C-index), produces a boxplot of the
per-TAZ distribution (277 zones), grouped by transport mode, GCN vs. Boosted
Forest side by side.

Usage:
    python scripts/compare_baseline.py
    python scripts/compare_baseline.py <gcn_run_dir> <baseline_run_dir>
"""

import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from qol_surrogate.baseline import evaluate_baseline_per_taz, flatten_x
from qol_surrogate.data import (
    DRY_BASELINE_DIR,
    DRY_BASELINE_DIR_NO_19,
    TAZ_PARQUET_DIR,
    TAZ_PARQUET_DIR_NO_19,
    TRANSPORT_MODES,
    ZONES_FILE,
    build_data_list,
    build_taz_geometries,
    build_taz_to_idx,
    create_edge_index,
    create_polygon_metrics,
    get_taz_ids,
    load_dataset_tensors,
    load_hexes,
)
from qol_surrogate.evaluate import evaluate_per_taz
from qol_surrogate.model import GCNResNet

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GCN_MODELS_DIR = os.path.join(REPO_ROOT, "models")
BASELINE_MODELS_DIR = os.path.join(REPO_ROOT, "models", "baseline")


def find_latest(models_dir, pattern):
    run_dirs = sorted(d for d in glob.glob(os.path.join(models_dir, pattern)) if os.path.isdir(d))
    if not run_dirs:
        raise FileNotFoundError(f"no runs matching {pattern} found in {models_dir}")
    return run_dirs[-1]


def _taz_tensors(include_file_19):
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)
    taz_to_idx = build_taz_to_idx(taz_ids)
    tazes = build_taz_geometries(hexes)
    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    taz_parquet_dir = TAZ_PARQUET_DIR if include_file_19 else TAZ_PARQUET_DIR_NO_19
    dry_baseline_dir = DRY_BASELINE_DIR if include_file_19 else DRY_BASELINE_DIR_NO_19
    x, y = load_dataset_tensors(area, perimeter, taz_parquet_dir=taz_parquet_dir, dry_baseline_dir=dry_baseline_dir)

    return hexes, taz_ids, taz_to_idx, tazes, x, y, dry_baseline_dir


def load_gcn_per_taz(run_dir):
    checkpoint = torch.load(os.path.join(run_dir, "model.pt"), weights_only=False)
    hp = checkpoint["hyperparameters"]
    idx_test = checkpoint["idx_test"]
    include_file_19 = checkpoint.get("include_file_19", True)

    model = GCNResNet(
        in_channels=hp["in_channels"], hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"], n_layers=hp["n_layers"], dropout_rate=hp["dropout_rate"],
        x_mean=torch.zeros(hp["in_channels"]), x_std=torch.ones(hp["in_channels"]),
        y_mean=torch.zeros(hp["out_channels"]), y_std=torch.ones(hp["out_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    hexes, taz_ids, taz_to_idx, tazes, x, y, dry_baseline_dir = _taz_tensors(include_file_19)
    edge_index, edge_weight = create_edge_index(tazes, taz_to_idx)

    x_test = (x[idx_test] - model.x_mean) / model.x_std
    y_test = (y[idx_test] - model.y_mean) / model.y_std

    test_data = build_data_list(x_test, y_test, edge_index, edge_weight)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    per_taz = evaluate_per_taz(model, test_loader, taz_ids, dry_baseline_dir=dry_baseline_dir)
    return per_taz, idx_test, include_file_19


def load_baseline_per_taz(run_dir):
    with open(os.path.join(run_dir, "model.pkl"), "rb") as f:
        checkpoint = pickle.load(f)
    models = checkpoint["models"]
    idx_test = checkpoint["idx_test"]
    include_file_19 = checkpoint["include_file_19"]

    _, taz_ids, _, _, x, y, dry_baseline_dir = _taz_tensors(include_file_19)

    x_test_flat = flatten_x(x[idx_test])
    y_test = y[idx_test]

    per_taz = evaluate_baseline_per_taz(models, x_test_flat, y_test, taz_ids, dry_baseline_dir=dry_baseline_dir)
    return per_taz, idx_test, include_file_19


METRICS = ["r2", "wape", "c_index"]
METRIC_LABELS = {"r2": "R²", "wape": "WAPE", "c_index": "C-index"}

# fixed categorical order (slots 1 & 2 of the validated default palette) -
# color always means the same model across every panel
COLOR_GCN = "#2a78d6"       # blue
COLOR_BASELINE = "#eb6834"  # orange
INK_MUTED = "#898781"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
BOX_WIDTH = 0.32


def plot_comparison(gcn_per_taz, baseline_per_taz, out_path):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 5))
    x = np.arange(len(TRANSPORT_MODES))

    for ax, metric in zip(axes, METRICS):
        gcn_data = [gcn_per_taz[metric][mode].dropna().to_numpy() for mode in TRANSPORT_MODES]
        baseline_data = [baseline_per_taz[metric][mode].dropna().to_numpy() for mode in TRANSPORT_MODES]

        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        def _style(bp, color):
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_edgecolor(color)
                box.set_alpha(0.55)
                box.set_linewidth(1.2)
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color(INK_MUTED)
                    line.set_linewidth(1.0)
            for median in bp["medians"]:
                median.set_color(INK_SECONDARY)
                median.set_linewidth(1.4)
            for flier in bp["fliers"]:
                flier.set(marker="o", markersize=3, markerfacecolor=color,
                          markeredgecolor="none", alpha=0.4)

        bp_gcn = ax.boxplot(gcn_data, positions=x - BOX_WIDTH / 2 - 0.02, widths=BOX_WIDTH,
                             patch_artist=True, zorder=2)
        bp_baseline = ax.boxplot(baseline_data, positions=x + BOX_WIDTH / 2 + 0.02, widths=BOX_WIDTH,
                                  patch_artist=True, zorder=2)
        _style(bp_gcn, COLOR_GCN)
        _style(bp_baseline, COLOR_BASELINE)

        ax.set_xlim(-0.6, len(TRANSPORT_MODES) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(TRANSPORT_MODES, color=INK_SECONDARY)
        ax.set_title(METRIC_LABELS[metric], color="#0b0b0b")
        ax.tick_params(axis="y", colors=INK_MUTED)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_GCN, alpha=0.55, edgecolor=COLOR_GCN, linewidth=1.2),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_BASELINE, alpha=0.55, edgecolor=COLOR_BASELINE, linewidth=1.2),
    ]
    fig.legend(handles, ["GCN", "Boosted Forest"], loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False)
    fig.suptitle("TAZ-resolution accuracy per zone, by mode — GCN vs. Boosted Forest baseline (n=277 TAZs)", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")


def main():
    gcn_run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest(GCN_MODELS_DIR, "gcn_resnet_*")
    baseline_run_dir = sys.argv[2] if len(sys.argv) > 2 else find_latest(BASELINE_MODELS_DIR, "gbm_*")
    print(f"GCN run:      {gcn_run_dir}")
    print(f"Baseline run: {baseline_run_dir}")

    print("Computing GCN per-TAZ metrics...")
    gcn_per_taz, gcn_idx_test, gcn_include_19 = load_gcn_per_taz(gcn_run_dir)

    print("Computing Boosted Forest per-TAZ metrics...")
    baseline_per_taz, baseline_idx_test, baseline_include_19 = load_baseline_per_taz(baseline_run_dir)

    if gcn_include_19 != baseline_include_19 or list(gcn_idx_test) != list(baseline_idx_test):
        raise ValueError(
            "GCN and baseline runs were evaluated on different data/test scenarios - "
            "not directly comparable. Make sure both were trained with the same "
            "include_file_19 setting (so split_dataset() reproduces the same idx_test)."
        )

    out_path = os.path.join(baseline_run_dir, "comparison_taz.png")
    plot_comparison(gcn_per_taz, baseline_per_taz, out_path)


if __name__ == "__main__":
    main()
