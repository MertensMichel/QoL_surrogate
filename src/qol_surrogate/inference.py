"""[SHARED: all 3 modes, via mode=] Inference entry point for the QoL surrogate - load a trained checkpoint once
(by mode - "onfoot", "car", or "bicycle"), then call .predict() as many times as
needed (e.g. once per RL step) without reloading anything.

Usage:
    surrogate = QoL_surrogate(mode="car")                   # load once, at setup time
    result = surrogate.predict(raw_water_depths)            # call repeatedly

    # raw_water_depths: 1D array-like, one water-depth value per edge of that mode's
    #   road network, same order as {MODE}_edges_Copenhagen.pkl (length ==
    #   surrogate.model.edge_taz_ids.shape[0], e.g. 175342 for onfoot)
    # result: DataFrame indexed by real taz_zoneid, columns = POI category names -
    #   absolute (not deviation) accessibility for that mode, ready to use directly.
"""

import os

import pandas as pd
import torch

from qol_surrogate.model_architecture import GCNResNet

# Duplicated from qol_surrogate.data_onfoot.POI_CATEGORIES rather than imported
# Order must match the model's output channel order
POI_CATEGORIES = ['cultural', 'education', 'green_space', 'health',
                   'public_spaces', 'public_transportation', 'sports']

# Where staged production checkpoints live, one per mode - see
# QoL_surrogate.__init__'s `mode` argument.
STATIC_FEATURES_DIR = "/mnt/raid1/MAAT/20.surrogate_data/qol_surrogate"


def resolve_model_path(mode, static_features_dir=STATIC_FEATURES_DIR):
    """mode ("onfoot", "car", or "bicycle", case-insensitive) -> the conventioned
    checkpoint path for that mode: qol_surrogate_weights_{mode}.pt."""
    return os.path.join(static_features_dir, f"qol_surrogate_weights_{mode.lower()}.pt")


def load_model(model_path):
    """Reconstruct GCNResNet from a checkpoint - build a placeholder with dummy
    tensors of the right shape for every buffer, then load_state_dict() overwrites
    them with the real saved values. Shapes come from checkpoint["hyperparameters"],
    which save_run() populated for exactly this purpose.
    """
    checkpoint = torch.load(model_path, weights_only=False)
    hp = checkpoint["hyperparameters"]

    model = GCNResNet(
        in_channels=hp["in_channels"], hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"], n_layers=hp["n_layers"], dropout_rate=hp["dropout_rate"],
        x_mean=torch.zeros(hp["in_channels"]), x_std=torch.ones(hp["in_channels"]),
        y_mean=torch.zeros(hp["out_channels"]), y_std=torch.ones(hp["out_channels"]),
        static_features=torch.zeros(hp["num_taz"], hp["n_static_features"]),
        edge_index=torch.zeros(2, hp["n_edges"], dtype=torch.long),
        edge_weight=torch.zeros(hp["n_edges"]),
        dry_baseline=torch.zeros(hp["num_taz"], hp["out_channels"]),
        edge_taz_ids=torch.zeros(hp["n_edge_taz_ids"], dtype=torch.long),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["taz_ids"]


class QoL_surrogate:
    def __init__(self, mode=None, model_path=None):
        """Load a trained checkpoint - pass exactly one of `mode` or `model_path`.

        mode: "onfoot", "car", or "bicycle" (case-insensitive) - resolves to the
        staged production checkpoint for that mode (qol_surrogate_weights_{mode}.pt
        under STATIC_FEATURES_DIR). This is the normal way to construct this class.

        model_path: an explicit checkpoint path, for loading a specific file directly
        (e.g. a run-directory checkpoint during development) instead of the staged
        production one.
        """
        if (mode is None) == (model_path is None):
            raise ValueError("QoL_surrogate: pass exactly one of `mode` or `model_path`")

        if mode is not None:
            model_path = resolve_model_path(mode)

        self.mode = mode
        self.model, self.taz_ids = load_model(model_path)

    def predict(self, input_sample):
        """Predict absolute accessibility (for this instance's mode) per TAZ per POI
        category from a raw, edge-resolution water-depth sample (same format/order
        as one row of the training data - length == number of edges in this mode's
        road network).

        Returns: a DataFrame indexed by real taz_zoneid (self.taz_ids), columns
        named by POI category - self-labeling, so no separate lookup is needed to
        know which row/column means what.
        """
        num_taz = self.model.static_features.shape[0]

        # input_sample: 1D array-like (list / numpy array / torch tensor) of length
        # self.model.edge_taz_ids.shape[0] (e.g. 175342 for onfoot in Copenhagen) -
        # one water-depth value per edge of this mode's road network, in the SAME
        # ORDER as that mode's {MODE}_edges_Copenhagen.pkl (i.e. positionally aligned
        # with edge_taz_ids, since that's what it gets grouped against right below).


        # CAUTION: this order (mean, p25, p50, p75, p90) must exactly match
        # data_{mode}.load_dataset_tensors()'s dynamic_cols order used at training
        # time (data_onfoot/data_car/data_bicycle each define their own, identically
        # ordered) - it's currently duplicated by hand in all these places, not
        # derived from one shared source, so a change on one side won't error on the
        # other, it'll just silently mislabel which column means what.
        grouped = pd.Series(input_sample).groupby(self.model.edge_taz_ids.numpy())
        dynamic_stats = [grouped.mean()] + [grouped.quantile(q) for q in [0.25, 0.5, 0.75, 0.9]]
        dynamic = torch.stack([
            torch.tensor(stat.reindex(range(num_taz)).to_numpy(), dtype=torch.float32)
            for stat in dynamic_stats
        ], dim=-1)  # [num_taz, 5]

        x = torch.cat([dynamic, self.model.static_features], dim=-1)  # [num_taz, in_channels]
        x_norm = (x - self.model.x_mean) / self.model.x_std

        self.model.eval()
        with torch.no_grad():
            out = self.model(x_norm, self.model.graph_edge_index, self.model.graph_edge_weight)

        # Denormalize the predicted deviation, then add back the dry baseline to
        # recover absolute accessibility - the number actually usable downstream.
        absolute = out * self.model.y_std + self.model.y_mean + self.model.dry_baseline

        return pd.DataFrame(absolute.numpy(), index=self.taz_ids, columns=POI_CATEGORIES)






