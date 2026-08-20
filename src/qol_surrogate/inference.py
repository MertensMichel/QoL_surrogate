"""Inference entry point for the ON_FOOT QoL surrogate - load a trained checkpoint
once, then call .predict() as many times as needed (e.g. once per RL step) without
reloading anything.

Usage:
    surrogate = QoL_surrogate("path/to/model.pt")          # load once, at setup time
    result = surrogate.predict(raw_water_depths)            # call repeatedly

    # raw_water_depths: 1D array-like, one ON_FOOT water-depth value per edge, same
    #   order as ON_FOOT_edges_Copenhagen.pkl (length == n_onfoot_edges, e.g. 175342)
    # result: DataFrame indexed by real taz_zoneid, columns = POI category names -
    #   absolute (not deviation) ON_FOOT accessibility, ready to use directly.
"""

import pandas as pd
import torch

from qol_surrogate.model_onfoot import GCNResNet

# Duplicated from qol_surrogate.data_onfoot.POI_CATEGORIES rather than imported
# Order must match the model's output channel order
POI_CATEGORIES = ['cultural', 'education', 'green_space', 'health',
                   'public_spaces', 'public_transportation', 'sports']


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
        edge_taz_ids=torch.zeros(hp["n_onfoot_edges"], dtype=torch.long),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["taz_ids"]


class QoL_surrogate:
    def __init__(self, model_path):

        self.model, self.taz_ids = load_model(model_path)

    def predict(self, input_sample):
        """Predict absolute ON_FOOT accessibility per TAZ per POI category from a raw,
        edge-resolution water-depth sample (same format/order as one row of the
        training data - length == number of ON_FOOT edges).

        Returns: a DataFrame indexed by real taz_zoneid (self.taz_ids), columns
        named by POI category - self-labeling, so no separate lookup is needed to
        know which row/column means what.
        """
        num_taz = self.model.static_features.shape[0]

        # input_sample: 1D array-like (list / numpy array / torch tensor) of length
        # self.model.edge_taz_ids.shape[0] (n_onfoot_edges, e.g. 175342 for Copenhagen)
        # - one ON_FOOT water-depth value per edge, in the SAME ORDER as
        # ON_FOOT_edges_Copenhagen.pkl (i.e. positionally aligned with edge_taz_ids,
        # since that's what it gets grouped against right below).


        # CAUTION: this order (mean, p25, p50, p75, p90) must exactly match
        # data_onfoot.load_dataset_tensors()'s dynamic_cols order used at training
        # time - it's currently duplicated by hand in both places, not derived from
        # one shared source, so a change on one side won't error on the other, it'll
        # just silently mislabel which column means what.
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






