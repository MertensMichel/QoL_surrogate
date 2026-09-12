"""[LEGACY: 3-mode/21-channel pipeline] The GCNResNet model class.

Ported from: notebooks/gcn.ipynb
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCNResNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout_rate, x_mean, y_mean, x_std, y_std):
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
