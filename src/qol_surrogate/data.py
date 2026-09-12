"""[LEGACY: 3-mode/21-channel pipeline] Data aggregation, TAZ graph construction, Data-object building, and normalization.

Ported from: notebooks/aggregation_to_taz_level.ipynb, notebooks/build_graph.ipynb
Each function below is a faithful port of a notebook cell/section - logic and
variable names are unchanged, only wrapped into functions with explicit
parameters (defaulting to the same hardcoded paths the notebooks used) instead
of relying on notebook-global state.
"""

import os
import pickle
from itertools import product
import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

TRANSPORT_MODES = ['CAR', 'BICYCLE', 'ON_FOOT']
POI_CATEGORIES = ['cultural', 'education', 'green_space', 'health',
                   'public_spaces', 'public_transportation', 'sports']

ZONES_FILE = "/mnt/raid1/MAAT/08.accessibility/Copenhagen/zones_Copenhagen.pkl"
NETWORK_DIR = "/mnt/raid1/MAAT/network"
HEX_PARQUET_DIR = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_withassignment/parquet"
TAZ_PARQUET_DIR = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_withassignment/parquet_taz"
TAZ_PARQUET_DIR_NO_19 = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_withassignment/parquet_taz_without_19"
DRY_BASELINE_DIR = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_withassignment/parquet_taz_baseline"
DRY_BASELINE_DIR_NO_19 = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_withassignment/parquet_taz_baseline_without_19"


def load_hexes(zones_file=ZONES_FILE):
    with open(zones_file, 'rb') as f:
        return pickle.load(f)


def get_taz_ids(hexes):
    """Canonical TAZ ordering used for every TAZ-level list everywhere downstream."""
    return sorted(hexes['taz_zoneid'].unique())


def build_taz_to_idx(taz_ids):
    return {taz_id: i for i, taz_id in enumerate(taz_ids)}


# --- aggregation_to_taz_level.ipynb: raw hex/edge-level parquet -> TAZ-level dataset ---

def aggregate_to_taz(hexes, taz_ids, include_file_19=True, hex_parquet_dir=HEX_PARQUET_DIR, network_dir=NETWORK_DIR):
    """Aggregate raw per-sample hex/edge-level accessibility data to TAZ level.

    Water depths: mean per TAZ, over the edges that lie in that TAZ.
    Accessibility: mean per TAZ, over the hexes that lie in that TAZ.
    Output has the same 24 columns as the source parquet, but every list is
    length-277 (TAZ-level) instead of edge/hex-level.
    """

    parquet_files = sorted(glob.glob(hex_parquet_dir + "/*.parquet"))
    if not include_file_19:
        parquet_files = [f for f in parquet_files if not f.endswith("Copenhagen_19.parquet")]

    ds = load_dataset("parquet", data_files=parquet_files)
    ds.set_format("pandas")

    edges = {
        mode: pickle.load(open(f"{network_dir}/{mode}_edges_Copenhagen.pkl", "rb"))
        for mode in TRANSPORT_MODES
    }
    edge_taz_ids = {mode: edges[mode]['region_id'].to_numpy() for mode in TRANSPORT_MODES}

    cols = list(product(TRANSPORT_MODES, POI_CATEGORIES))
    aligned_acc = hexes[['hex_id', 'taz_zoneid']].copy()

    full_df = ds["train"].to_pandas()

    sample_rows = []
    for sample_id, row in full_df.iterrows():  # looping through all samples
        sample_row = {}

        for mode in TRANSPORT_MODES:
            wd_col = f'water_depths_{mode}'
            wd_taz = pd.Series(row[wd_col]).groupby(edge_taz_ids[mode]).mean().reindex(taz_ids)
            sample_row[wd_col] = wd_taz.tolist()

        for mode, category in cols:
            col_name = f'cumulative_accessibility_{mode}_{category}'
            aligned_acc[col_name] = row[col_name]

        agg = aligned_acc.groupby('taz_zoneid', as_index=False).mean(numeric_only=True)
        agg = agg.set_index('taz_zoneid').reindex(taz_ids)

        for mode, category in cols:
            col_name = f'cumulative_accessibility_{mode}_{category}'
            sample_row[col_name] = agg[col_name].tolist()

        sample_rows.append(sample_row)

    taz_acc = pd.DataFrame(sample_rows)
    taz_acc = taz_acc[list(ds["train"].column_names)]  # match the source parquet's column order exactly

    return taz_acc


def save_taz_aggregated_dataset(taz_acc, taz_ids, output_dir=TAZ_PARQUET_DIR):

    os.makedirs(output_dir, exist_ok=True)

    taz_acc.to_parquet(os.path.join(output_dir, "Copenhagen_taz.parquet"))

    # The order of every list above is implicit (matches the source convention) -
    # save the TAZ id it corresponds to at each position, right next to the data.

    with open(os.path.join(output_dir, "taz_ids.pkl"), "wb") as f:
        pickle.dump(taz_ids, f)

    return output_dir


def save_pseudo_dry_baseline(taz_acc, include_file_19=True, output_dir=None):
    """Save the scenario with the lowest total CAR water depth as a stand-in dry
    baseline, in its own folder so it never gets glob'd in with the real samples.

    output_dir defaults based on include_file_19 (DRY_BASELINE_DIR vs
    DRY_BASELINE_DIR_NO_19) - pass it explicitly to override.
    """
    if output_dir is None:
        output_dir = DRY_BASELINE_DIR if include_file_19 else DRY_BASELINE_DIR_NO_19

    os.makedirs(output_dir, exist_ok=True)

    min_row = float('inf')
    min_id = -1
    for id, row in enumerate(taz_acc['water_depths_CAR']):
        row_sum = sum(row)
        if row_sum < min_row:
            min_row = row_sum
            min_id = id

    taz_acc.iloc[[min_id]].to_parquet(os.path.join(output_dir, "Copenhagen_taz_pseudo_dry_baseline.parquet"))

    return output_dir


# --- build_graph.ipynb: TAZ graph + training tensors ---

def build_taz_geometries(hexes):
    tazes = hexes[['taz_zoneid', 'geometry']].dissolve(by='taz_zoneid')
    tazes = tazes.to_crs(epsg=25832)
    return tazes


def create_edge_index(tazes, taz_to_idx):
    tazes_touching = gpd.sjoin(tazes, tazes, predicate='touches')  # only keeps left geometry
    tazes_touching = tazes_touching.reset_index()

    tazes_touching['geometry_right']  = tazes_touching['taz_zoneid_right'].map(tazes['geometry'])
    tazes_touching['shared_boundary'] = tazes_touching['geometry'].intersection(tazes_touching['geometry_right'])
    tazes_touching['boundary_weight'] = tazes_touching['shared_boundary'].length

    src = tazes_touching['taz_zoneid_left'].map(taz_to_idx)
    dst = tazes_touching['taz_zoneid_right'].map(taz_to_idx)

    edge_index_np = np.array([src.to_numpy(), dst.to_numpy()])  # combine into one ndarray first - torch.tensor() on a list of ndarrays is slow
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weight = torch.tensor(tazes_touching['boundary_weight'].to_numpy(), dtype=torch.float32)

    return edge_index, edge_weight


def create_polygon_metrics(tazes, taz_ids):
    area      = torch.tensor(tazes.geometry.area.reindex(taz_ids).to_numpy(), dtype=torch.float32)
    perimeter = torch.tensor(tazes.geometry.length.reindex(taz_ids).to_numpy(), dtype=torch.float32)

    return area, perimeter


def load_dataset_tensors(area, perimeter, taz_parquet_dir=TAZ_PARQUET_DIR, dry_baseline_dir=DRY_BASELINE_DIR):
    """Load the TAZ-aggregated dataset + dry baseline, build x (water depth per
    mode + static area/perimeter) and y (accessibility deviation from dry
    baseline) tensors.

    x: [n_samples, 277, 5]   y: [n_samples, 277, 21]
    """
    x_cols = [f"water_depths_{mode}" for mode in TRANSPORT_MODES]
    y_cols = [f"cumulative_accessibility_{mode}_{poi}"
              for mode, poi in product(TRANSPORT_MODES, POI_CATEGORIES)]

    ds = load_dataset("parquet", data_files=os.path.join(taz_parquet_dir, "*.parquet"))
    ds = ds['train']
    ds.set_format(type='torch')  # no columns= restriction — ALL columns become tensors
    ds = ds[:]

    dry_baseline = load_dataset("parquet", data_files=os.path.join(dry_baseline_dir, "*.parquet"))
    dry_baseline = dry_baseline['train']
    dry_baseline.set_format(type='torch')
    dry_baseline = dry_baseline[:]

    x_dynamic = torch.stack([ds[c] for c in x_cols], dim=-1)  # [n_samples, 277, 3]

    x_static = torch.stack([area, perimeter], dim=-1)
    x_static = x_static.unsqueeze(dim=0)
    x_static = x_static.expand(x_dynamic.shape[0], -1, -1)

    x = torch.cat((x_dynamic, x_static), dim=-1)

    y_absolute = torch.stack([ds[c] for c in y_cols], dim=-1)  # [n_samples, 277, 21]
    y_dry = torch.stack([dry_baseline[c] for c in y_cols], dim=-1)

    # DECISION: currently predicting the deviation from the dry (no-flood) baseline,
    # not the absolute accessibility value. To switch back to predicting the absolute
    # value later, just use y = y_absolute instead of the line below.
    y = y_absolute - y_dry

    return x, y


def split_dataset(x, y, test_size=0.2, val_size=0.125, random_state=13):
    """Scenario-level train/val/test split (never split at the node level - see
    notes from the build_graph session on why that would leak scenarios).
    val_size is relative to the *remaining* data after the test split
    (0.125 of the remaining 80% = 10% of the original total).

    Also returns which *original* scenario index (0..n_samples-1, matching row
    order in the source hex-level parquet) ended up in each split - needed to
    go back to hex-resolution ground truth for exactly the test-set scenarios.
    """
    idx = np.arange(x.shape[0])

    x_tmp, x_test, y_tmp, y_test, idx_tmp, idx_test = train_test_split(
        x, y, idx, test_size=test_size, random_state=random_state
    )
    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        x_tmp, y_tmp, idx_tmp, test_size=val_size, random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test, idx_train, idx_val, idx_test


def compute_normalization_stats(x_train, y_train):
    """Mean/std per feature, pooled over the scenario and TAZ-node axes.
    Must only ever be computed from the training split.
    """
    x_mean = x_train.mean(dim=(0, 1))
    x_std  = x_train.std(dim=(0, 1)).clamp(1e-6)

    y_mean = y_train.mean(dim=(0, 1))
    y_std  = y_train.std(dim=(0, 1)).clamp(1e-6)

    return x_mean, x_std, y_mean, y_std


def normalize(tensor, mean, std):
    return (tensor - mean) / std


def build_data_list(x, y, edge_index, edge_weight):
    """One torch_geometric Data object per scenario, all sharing the same fixed
    TAZ-adjacency graph (edge_index/edge_weight)."""
    return [
        Data(x=x[i], y=y[i], edge_index=edge_index, edge_weight=edge_weight)
        for i in range(x.shape[0])
    ]


def build_loaders(train_data, val_data, test_data, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
