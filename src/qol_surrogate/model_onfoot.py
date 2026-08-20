"""The GCNResNet model class.

Ported from: notebooks/gcn.ipynb
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCNResNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout_rate,
                 x_mean, y_mean, x_std, y_std,
                 static_features, edge_index, edge_weight, dry_baseline, edge_taz_ids):
        super().__init__()
        self.encoder      = nn.Linear(in_channels, hidden_channels)
        self.convs        = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(n_layers)])
        self.bns          = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(n_layers)])
        self.dropout_rate = dropout_rate
        self.decoder      = nn.Linear(hidden_channels, out_channels)

        self.register_buffer('x_mean', x_mean)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_std', y_std)

        # Deployment-context data bundled into the checkpoint so it's fully
        # self-contained for inference (see qol_surrogate.inference) - not used by
        # forward() itself. graph_edge_index/graph_edge_weight are kept separate from
        # forward()'s own edge_index/edge_weight arguments deliberately: training
        # batches multiple graphs into one block-diagonal edge_index per minibatch
        # (via PyG's DataLoader), which is NOT the same tensor as the single canonical
        # 277-node graph stored here - these buffers are only valid for single-sample
        # inference, forward() must keep taking its graph as an explicit argument.
        self.register_buffer('static_features', static_features)
        self.register_buffer('graph_edge_index', edge_index)
        self.register_buffer('graph_edge_weight', edge_weight)
        self.register_buffer('dry_baseline', dry_baseline)
        # Per-ON_FOOT-edge -> TAZ mapping (positional, dustbin index = len(taz_ids) for
        # edges belonging to a TAZ with no hexes) - see data_onfoot.load_edge_taz_ids().
        # Needed to aggregate a raw edge-resolution inference sample into TAZ-level stats.
        self.register_buffer('edge_taz_ids', edge_taz_ids)


    def forward(self, x, edge_index, edge_weight=None):

        h = self.encoder(x)  # by default nn.Linear only acts on the last dimension

        h_res = h
        for conv, bns in zip(self.convs, self.bns):
            h = conv(h_res, edge_index, edge_weight)
            h = bns(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout_rate, training=self.training)

            h_res = h_res + h

        x_out = self.decoder(h_res)

        return x_out