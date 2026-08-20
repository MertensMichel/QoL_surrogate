"""Data aggregation, TAZ graph construction, Data-object building, and normalization.

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
POI_FILE = "/mnt/raid1/MAAT/07.osm_pois/Copenhagen/pois_clean_Copenhagen.pkl"
HEX_PARQUET_DIR = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_foot/parquet"
TAZ_PARQUET_DIR = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_foot/parquet_taz"
TAZ_PARQUET_DIR_NO_19 = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_foot/parquet_taz_without_19"
DRY_BASELINE_DIR = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_foot/parquet_taz_baseline"
DRY_BASELINE_DIR_NO_19 = "/mnt/raid1/MAAT/20.surrogate_data/cph/accessibility_foot/parquet_taz_baseline_without_19"
BASELINE_FILE = "/mnt/raid1/MAAT/20.surrogate_data/qol_surrogate/Copenhagen_acc_raindist_samples_BASELINE.pkl"
STATIC_FEATURES_DIR = "/mnt/raid1/MAAT/20.surrogate_data/qol_surrogate"

# Canonical order for the static-feature block of x - the SAME list must be used
# wherever static_features.pkl's dict gets turned into a tensor (training assembly
# here, and later whatever builds the model's buffered static-feature tensor for
# inference), or features end up silently misaligned between training and inference.
STATIC_FEATURE_COLS = (
    [f"num_streets_{mode}" for mode in TRANSPORT_MODES]
    + [f"street_length_{mode}_{stat}" for mode in TRANSPORT_MODES for stat in ["sum", "median", "std"]]
    + ["node_degree", "shared_boundary_fraction", "mean_neighbor_centroid_distance"]
    + [f"poi_{category}" for category in POI_CATEGORIES]
    + ["area", "perimeter"]
)

# 19 source files that are exact, row-for-row duplicates of other source files (confirmed via
# full water_depths_ON_FOOT content hash, same order) - each pair below is (kept, dropped).
# Copenhagen_0 <-> Copenhagen_1, and Copenhagen_{83..100} <-> Copenhagen_{101..118} (offset of 18).
DUPLICATE_FILES = ["Copenhagen_1.parquet"] + [f"Copenhagen_{n}.parquet" for n in range(101, 119)]


def load_hexes(zones_file=ZONES_FILE):
    with open(zones_file, 'rb') as f:
        return pickle.load(f)


def get_taz_ids(hexes):
    """Canonical TAZ ordering used for every TAZ-level list everywhere downstream."""
    return sorted(hexes['taz_zoneid'].unique())


def build_taz_to_idx(taz_ids):
    return {taz_id: i for i, taz_id in enumerate(taz_ids)}


# --- aggregation_to_taz_level.ipynb: raw hex/edge-level parquet -> TAZ-level dataset ---

def aggregate_to_taz(hexes, taz_ids, exclude_duplicates=True,
                      hex_parquet_dir=HEX_PARQUET_DIR, network_dir=NETWORK_DIR):
    """Aggregate raw per-sample hex/edge-level accessibility data to TAZ level.

    Water depths: mean + p25/p50/p75/p90 per TAZ, over the ON_FOOT edges that lie
    in that TAZ. Accessibility: mean per TAZ, over the hexes that lie in that TAZ.

    Processes one source file at a time (~100-700 scenarios, up to ~1.4MB per
    water-depth array) rather than loading the whole ~30k-scenario raw dataset
    into memory at once - the full dataset would be ~59GB as a single in-memory
    DataFrame, which doesn't fit. Each file's raw arrays are aggregated down to
    TAZ level (~11KB/scenario) and discarded before moving to the next file.
    """

    parquet_files = sorted(glob.glob(hex_parquet_dir + "/*.parquet"))
    if exclude_duplicates:
        parquet_files = [f for f in parquet_files if os.path.basename(f) not in DUPLICATE_FILES]

    edges = {
        mode: pickle.load(open(f"{network_dir}/{mode}_edges_Copenhagen.pkl", "rb"))
        for mode in TRANSPORT_MODES
    }
    edge_taz_ids = {mode: edges[mode]['region_id'].to_numpy() for mode in TRANSPORT_MODES}

    cols = list(product(["ON_FOOT"], POI_CATEGORIES))
    aligned_acc = hexes[['hex_id', 'taz_zoneid']].copy()

    acc_cols = [f'cumulative_accessibility_{mode}_{category}' for mode, category in cols]
    needed_cols = ['water_depths_ON_FOOT'] + acc_cols

    sample_rows = []
    for file_num, pf in enumerate(parquet_files):
        print(f"Aggregating {os.path.basename(pf)} ({file_num + 1}/{len(parquet_files)})...")
        df = pd.read_parquet(pf, columns=needed_cols)

        for _, row in df.iterrows():  # looping through this file's samples only
            sample_row = {}

            for mode in ["ON_FOOT"]:
                wd_col = f'water_depths_{mode}'
                wd_taz = pd.Series(row[wd_col]).groupby(edge_taz_ids[mode]).mean().reindex(taz_ids)
                sample_row[wd_col] = wd_taz.tolist()

                for percentile in [0.25, 0.5, 0.75, 0.90]:
                    col_name = f'water_depths_{mode}_p{percentile*100:.0f}'
                    wd_perc = pd.Series(row[wd_col]).groupby(edge_taz_ids[mode]).quantile(percentile).reindex(taz_ids)
                    sample_row[col_name] = wd_perc.tolist()

            for mode, category in cols:
                col_name = f'cumulative_accessibility_{mode}_{category}'
                aligned_acc[col_name] = row[col_name]

            agg = aligned_acc.groupby('taz_zoneid', as_index=False).mean(numeric_only=True)
            agg = agg.set_index('taz_zoneid').reindex(taz_ids)

            for mode, category in cols:
                col_name = f'cumulative_accessibility_{mode}_{category}'
                sample_row[col_name] = agg[col_name].tolist()

            sample_rows.append(sample_row)

        del df  # free this file's raw per-edge/per-hex arrays before loading the next one

    taz_acc = pd.DataFrame(sample_rows)
    list_cols = ["water_depths_ON_FOOT", "water_depths_ON_FOOT_p25", "water_depths_ON_FOOT_p50",
                 "water_depths_ON_FOOT_p75", "water_depths_ON_FOOT_p90"] + acc_cols
    taz_acc = taz_acc[list(list_cols)]  # match the source parquet's column order exactly

    return taz_acc


def save_taz_aggregated_dataset(taz_acc, taz_ids, output_dir=TAZ_PARQUET_DIR):

    os.makedirs(output_dir, exist_ok=True)

    taz_acc.to_parquet(os.path.join(output_dir, "Copenhagen_taz.parquet"))

    # The order of every list above is implicit (matches the source convention) -
    # save the TAZ id it corresponds to at each position, right next to the data.

    with open(os.path.join(output_dir, "taz_ids.pkl"), "wb") as f:
        pickle.dump(taz_ids, f)

    return output_dir


def save_dry_baseline(hexes, taz_ids, baseline_file=BASELINE_FILE, output_dir=DRY_BASELINE_DIR,
                       network_dir=NETWORK_DIR):
    """Aggregate the true dry (zero-flood, event_intensity=0) baseline scenario to TAZ
    level and save it - replaces the old min-water-depth "pseudo" stand-in now that a
    real dry-run sample exists.

    Reuses aggregate_to_taz()'s exact per-scenario aggregation logic rather than
    duplicating it: the baseline pickle is the same raw single-scenario schema as one
    row of the main dataset, so it's written out as a one-file "dataset" of its own
    and run through the normal aggregation path.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(baseline_file, 'rb') as f:
        baseline_df = pickle.load(f)

    acc_cols = [f'cumulative_accessibility_ON_FOOT_{category}' for category in POI_CATEGORIES]
    needed_cols = ['water_depths_ON_FOOT'] + acc_cols
    baseline_df = baseline_df[needed_cols]  # drop event_intensity/alpha_per_zone - unneeded,
                                             # and alpha_per_zone isn't parquet-serializable as-is

    raw_dir = os.path.join(output_dir, "_raw_baseline")
    os.makedirs(raw_dir, exist_ok=True)
    baseline_df.to_parquet(os.path.join(raw_dir, "Copenhagen_baseline.parquet"))

    taz_baseline = aggregate_to_taz(hexes, taz_ids, exclude_duplicates=False,
                                     hex_parquet_dir=raw_dir, network_dir=network_dir)
    save_taz_aggregated_dataset(taz_baseline, taz_ids, output_dir=output_dir)

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


def load_edge_taz_ids(taz_to_idx, network_dir=NETWORK_DIR):
    """Per-ON_FOOT-edge TAZ assignment, in *positional* index space (0..len(taz_ids)-1,
    via taz_to_idx) - needed at inference time to aggregate a raw edge-resolution sample
    (same format as one row of the training data) into the 5 TAZ-level dynamic stats,
    the same way aggregate_to_taz() does internally for training. Kept in positional
    space (not raw taz_zoneid) for consistency with every other buffered tensor.

    Some ON_FOOT edges belong to a TAZ with no hexes (e.g. 103143 - has road edges but
    was never assigned a polygon, so it's absent from taz_to_idx entirely). Those edges
    get mapped to a "dustbin" index (len(taz_to_idx), one past the last valid TAZ) rather
    than left unmapped - scatter/groupby into a (len(taz_to_idx) + 1)-sized target and
    slice off the last row to discard them, instead of needing a separate mask.
    """
    edges = pickle.load(open(f"{network_dir}/ON_FOOT_edges_Copenhagen.pkl", "rb"))
    dustbin = len(taz_to_idx)
    edge_taz_ids = edges['region_id'].map(taz_to_idx).fillna(dustbin).astype(int).to_numpy()
    return torch.tensor(edge_taz_ids, dtype=torch.long)


def create_polygon_metrics(tazes, taz_ids):
    area      = torch.tensor(tazes.geometry.area.reindex(taz_ids).to_numpy(), dtype=torch.float32)
    perimeter = torch.tensor(tazes.geometry.length.reindex(taz_ids).to_numpy(), dtype=torch.float32)

    return area, perimeter


def create_graph_features(taz_ids, tazes, edge_index, edge_weights, perimeter, network_dir=NETWORK_DIR):
    """Static (scenario-independent) per-TAZ features. edge_index/edge_weights come from
    create_edge_index() and perimeter from create_polygon_metrics() - node_degree,
    shared_boundary_fraction, and mean_neighbor_centroid_distance all reuse that same
    touches-adjacency graph instead of running their own separate spatial joins.

    edge_index holds *positional* TAZ indices (0..len(taz_ids)-1, via taz_to_idx), which is
    already the same order taz_ids/perimeter are in - no reindexing needed for those three.
    """

    edges = {
            mode: pickle.load(open(f"{network_dir}/{mode}_edges_Copenhagen.pkl", "rb"))
            for mode in TRANSPORT_MODES
        }
    
    edge_taz_ids = {mode: edges[mode]['region_id'].to_numpy() for mode in TRANSPORT_MODES}

    static_features = {}
    for mode in TRANSPORT_MODES:
        feature_name = f"num_streets_{mode}"
        static_features[feature_name] = edges[mode].groupby(edge_taz_ids[mode]).size().reindex(taz_ids)

        feature_name = f"street_length_{mode}_sum"
        static_features[feature_name] = edges[mode].groupby(edge_taz_ids[mode])['length'].sum().reindex(taz_ids)

        feature_name = f"street_length_{mode}_median"
        static_features[feature_name] = edges[mode].groupby(edge_taz_ids[mode])['length'].median().reindex(taz_ids)

        feature_name = f"street_length_{mode}_std"
        static_features[feature_name] = edges[mode].groupby(edge_taz_ids[mode])['length'].std().reindex(taz_ids)

    src, dst = edge_index[0], edge_index[1]

    node_degree = torch.bincount(src, minlength=len(taz_ids))
    static_features["node_degree"] = node_degree

    boundary_sum = torch.zeros(len(taz_ids)).scatter_add_(0, src, edge_weights)
    static_features["shared_boundary_fraction"] = boundary_sum / perimeter

    centroids = tazes.geometry.centroid.reindex(taz_ids)
    centroid_xy = torch.tensor(
        np.stack([centroids.x.to_numpy(), centroids.y.to_numpy()], axis=1), dtype=torch.float32
    )
    edge_dist = (centroid_xy[src] - centroid_xy[dst]).norm(dim=1)
    dist_sum = torch.zeros(len(taz_ids)).scatter_add_(0, src, edge_dist)
    mean_neighbor_centroid_distance = dist_sum / node_degree

    # A handful of TAZs touch no other TAZ (node_degree == 0) -> 0/0 = NaN. Median-fill
    # from the TAZs that do have neighbors, rather than leaving NaN to poison training.
    isolated = node_degree == 0
    if isolated.any():
        median_dist = mean_neighbor_centroid_distance[~isolated].median()
        mean_neighbor_centroid_distance[isolated] = median_dist

    static_features["mean_neighbor_centroid_distance"] = mean_neighbor_centroid_distance

    return static_features



def create_poi_features(taz_ids, hexes, poi_file=POI_FILE):
    """Raw OSM POI count per TAZ per category (7 features) - how many POIs of each
    POI_CATEGORIES type fall within each TAZ polygon (point-in-polygon, not density).

    Builds TAZ polygons directly from hexes in their native CRS (EPSG:4326, same as
    the POI source) rather than via build_taz_geometries()'s EPSG:25832 reprojection -
    this is a containment check, not a length/area/distance measurement, and staying
    unprojected avoids a CRS mismatch on the spatial join. POI geometries here are
    already Points (verified across all 7 categories), so no centroid step is needed.
    """
    tazes = hexes[['taz_zoneid', 'geometry']].dissolve(by='taz_zoneid')

    with open(poi_file, 'rb') as f:
        pois = pickle.load(f)

    poi_features = {}
    for category in POI_CATEGORIES:
        joined = gpd.sjoin(pois[category][['geometry']], tazes, predicate='within')
        counts = joined.groupby('taz_zoneid').size().reindex(taz_ids, fill_value=0)
        poi_features[f"poi_{category}"] = counts

    return poi_features



def stack_static_features(static_features, cols=STATIC_FEATURE_COLS):
    """Combine the static_features.pkl dict (mixed pandas Series / torch tensors, one
    entry per feature, each already in taz_ids order) into a single ordered
    [n_taz, len(cols)] tensor, using `cols` as the canonical column order.

    Must be the same function/column order used to build whatever buffered static
    feature tensor ships with the trained model for inference - see STATIC_FEATURE_COLS.
    """
    columns = []
    for col in cols:
        value = static_features[col]
        if isinstance(value, torch.Tensor):
            columns.append(value.float())
        else:
            columns.append(torch.tensor(value.to_numpy(), dtype=torch.float32))
    return torch.stack(columns, dim=-1)


def load_dataset_tensors(static_features, taz_parquet_dir=TAZ_PARQUET_DIR, dry_baseline_dir=DRY_BASELINE_DIR):
    """Load the TAZ-aggregated dynamic dataset + dry baseline, combine with the
    (scenario-independent) static features into x, and build y (accessibility
    deviation from dry baseline).

    static_features: the dict loaded from static_features.pkl (create_graph_features +
    create_poi_features + area/perimeter, flattened) - identical for every scenario,
    broadcast across all of them here.

    x: [n_samples, 277, 5 + len(STATIC_FEATURE_COLS)]   y: [n_samples, 277, 7]
    """
    # CAUTION: this order (mean, p25, p50, p75, p90) must exactly match
    # inference.QoL_surrogate.predict()'s dynamic_stats order - it's currently
    # duplicated by hand in both places, not derived from one shared source, so a
    # change on one side won't error on the other, it'll just silently mislabel
    # which column means what.
    dynamic_cols = ["water_depths_ON_FOOT", "water_depths_ON_FOOT_p25", "water_depths_ON_FOOT_p50",
                     "water_depths_ON_FOOT_p75", "water_depths_ON_FOOT_p90"]
    y_cols = [f"cumulative_accessibility_ON_FOOT_{poi}" for poi in POI_CATEGORIES]

    ds = load_dataset("parquet", data_files=os.path.join(taz_parquet_dir, "*.parquet"))
    ds = ds['train']
    ds.set_format(type='torch')
    ds = ds[:]

    dry_baseline = load_dataset("parquet", data_files=os.path.join(dry_baseline_dir, "*.parquet"))
    dry_baseline = dry_baseline['train']
    dry_baseline.set_format(type='torch')
    dry_baseline = dry_baseline[:]

    x_dynamic = torch.stack([ds[c] for c in dynamic_cols], dim=-1)  # [n_samples, 277, 5]

    x_static = stack_static_features(static_features)  # [277, n_static]
    x_static = x_static.unsqueeze(dim=0).expand(x_dynamic.shape[0], -1, -1)  # [n_samples, 277, n_static]

    x = torch.cat((x_dynamic, x_static), dim=-1)

    y_absolute = torch.stack([ds[c] for c in y_cols], dim=-1)  # [n_samples, 277, 7]
    y_dry = torch.stack([dry_baseline[c] for c in y_cols], dim=-1)

    # DECISION: currently predicting the deviation from the dry (no-flood) baseline,
    # not the absolute accessibility value. To switch back to predicting the absolute
    # value later, just use y = y_absolute instead of the line below.
    y = y_absolute - y_dry

    return x, y, y_dry


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
