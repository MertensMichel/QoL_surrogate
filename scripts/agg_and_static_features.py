import os
import pickle

from qol_surrogate.data_onfoot import (
    aggregate_to_taz,
    build_taz_geometries,
    build_taz_to_idx,
    create_edge_index,
    create_poi_features,
    create_polygon_metrics,
    create_graph_features,
    load_hexes,
    get_taz_ids,
    save_dry_baseline,
    save_taz_aggregated_dataset,
    TAZ_PARQUET_DIR,
    ZONES_FILE,
)

# Static features + graph are scenario-independent - saved separately from the
# dynamic per-scenario data (which stays under TAZ_PARQUET_DIR).
STATIC_FEATURES_DIR = "/mnt/raid1/MAAT/20.surrogate_data/qol_surrogate"

def aggregate_dynamic_to_taz_and_save():
    """Aggregate raw per-sample hex/edge-level accessibility data to TAZ level and save the result."""
    # Load hexes and build TAZ mapping
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)
    taz_to_idx = build_taz_to_idx(taz_ids)

    # Aggregate to TAZ level
    taz_agg = aggregate_to_taz(hexes, taz_ids)

    # Save the aggregated dataset
    save_taz_aggregated_dataset(taz_agg, taz_ids, output_dir=TAZ_PARQUET_DIR)

def create_static_features_and_save():
    """Create static (scenario-independent) per-TAZ features + the TAZ graph, and save both."""
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)
    taz_to_idx = build_taz_to_idx(taz_ids)
    tazes = build_taz_geometries(hexes)

    edge_index, edge_weights = create_edge_index(tazes, taz_to_idx)
    area, perimeter = create_polygon_metrics(tazes, taz_ids)

    graph_features = create_graph_features(taz_ids, tazes, edge_index, edge_weights, perimeter)
    poi_features = create_poi_features(taz_ids, hexes)

    # Flattened - no more separate 'graph_features'/'poi_features' sub-dicts, every
    # feature is a top-level key so it can be looked up/stacked directly downstream.
    static_features = {
        **graph_features,
        **poi_features,
        "area": area,
        "perimeter": perimeter,
    }

    os.makedirs(STATIC_FEATURES_DIR, exist_ok=True)

    with open(os.path.join(STATIC_FEATURES_DIR, "static_features.pkl"), "wb") as f:
        pickle.dump(static_features, f)

    # edge_index/edge_weights are positional (via taz_to_idx) - taz_ids travels alongside
    # so that position i can always be mapped back to the TAZ it actually represents.
    with open(os.path.join(STATIC_FEATURES_DIR, "graph.pkl"), "wb") as f:
        pickle.dump({"edge_index": edge_index, "edge_weight": edge_weights, "taz_ids": taz_ids}, f)


def aggregate_dry_baseline_and_save():
    """Aggregate the true dry (zero-flood) baseline scenario to TAZ level and save it."""
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)

    save_dry_baseline(hexes, taz_ids)


if __name__ == "__main__":
    aggregate_dynamic_to_taz_and_save()
    create_static_features_and_save()
    aggregate_dry_baseline_and_save()