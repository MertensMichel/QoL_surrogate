"""[BICYCLE] Aggregate raw BICYCLE hex/edge-level accessibility data to TAZ level and save
both the dynamic dataset and the real dry baseline.

Unlike scripts/agg_and_static_features.py, this does NOT (re)compute static
features/graph - those are mode-agnostic and already sit under
STATIC_FEATURES_DIR from the onfoot run, so the CAR/BICYCLE pipelines reuse
them as-is.

Uses save_dry_baseline() - a real dry-run BICYCLE scenario is now available
(Copenhagen_acc_raindist_samples_BASELINE_BICYCLE.pkl), so the old min-total-water-
depth pseudo stand-in (save_pseudo_dry_baseline) is no longer needed.
"""

from qol_surrogate.data_bicycle import (
    DRY_BASELINE_DIR,
    TAZ_PARQUET_DIR,
    ZONES_FILE,
    aggregate_to_taz,
    get_taz_ids,
    load_hexes,
    save_dry_baseline,
    save_taz_aggregated_dataset,
)


def aggregate_dynamic_to_taz_and_save():
    """Aggregate raw per-sample hex/edge-level accessibility data to TAZ level,
    save the result, and save the real dry baseline."""
    hexes = load_hexes(ZONES_FILE)
    taz_ids = get_taz_ids(hexes)

    taz_agg = aggregate_to_taz(hexes, taz_ids)

    save_taz_aggregated_dataset(taz_agg, taz_ids, output_dir=TAZ_PARQUET_DIR)
    save_dry_baseline(hexes, taz_ids, output_dir=DRY_BASELINE_DIR)


if __name__ == "__main__":
    aggregate_dynamic_to_taz_and_save()
