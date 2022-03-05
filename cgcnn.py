import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic.typing import Literal
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from pydantic import BaseSettings as PydanticBaseSettings

from models.utils import RBFExpansion


class CGCNNConfig(PydanticBaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["cgcnn"]
    conv_layers: int = 1
    atom_input_features: int = 92
    edge_features: int = 32
    fc_features: int = 128
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"

class CGCNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(self, fc_features, **kwargs):
        super(CGCNNConv, self).__init__(aggr="add", node_dim=0)
        self.fc_features = fc_features
        self.atoms = nn.Linear(fc_features, fc_features)
        self.bn = nn.BatchNorm1d(fc_features)

        self.edge_interaction = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.BatchNorm1d(fc_features),
            nn.Sigmoid(),
        )

        self.edge_updating = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.BatchNorm1d(fc_features),
            nn.Softplus(),
        )

    def forward(self, data, x, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        out = self.propagate(
            data.edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        out = self.atoms(x + self.bn(out))
        return out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = torch.cat((x_i, x_j, edge_attr), dim=1)
        return self.edge_interaction(z) * self.edge_updating(z)


class CGCNN(nn.Module):
    """CGCNN dgl implementation."""

    def __init__(self, config: CGCNNConfig = CGCNNConfig(name="cgcnn")):
        """Set up CGCNN modules."""
        super().__init__()
        self.classification = config.classification
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.fc_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.fc_features)
        )

        self.conv_layers = nn.ModuleList(
            [
                CGCNNConv(config.fc_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.fc_features, config.fc_features), nn.SiLU()
        )

        if config.zero_inflated:
            # add latent Bernoulli variable model to zero out
            # predictions in non-negative regression model
            self.zero_inflated = True
            self.fc_nonzero = nn.Linear(config.fc_features, 1)
            self.fc_scale = nn.Linear(config.fc_features, 1)
            # self.fc_shape = nn.Linear(config.fc_features, 1)
            self.fc_scale.bias.data = torch.tensor(
                # np.log(2.1), dtype=torch.float
                2.1,
                dtype=torch.float,
            )
            if self.classification:
                raise ValueError(
                    "Classification not implemented for zero_inflated"
                )
        else:
            self.zero_inflated = False
            if self.classification:
                self.fc_out = nn.Linear(config.fc_features, 2)
                self.softmax = nn.LogSoftmax(dim=1)
            else:
                self.fc_out = nn.Linear(
                    config.fc_features, config.output_features
                )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, data):
        """CGCNN function mapping graph to outputs."""
        # fixed edge features: RBF-expanded bondlengths
        # bondlength = torch.norm(data.edge_attr, dim=1)
        edge_features = self.edge_embedding(data.edge_attr)
        # initial node features: atom feature network...
        node_features = self.atom_embedding(data.x)

        # CGCNN-Conv block: update node features
        for conv_layer in self.conv_layers:
            node_features, edge_features = conv_layer(data, node_features, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")
        features = self.fc(features)

        if self.zero_inflated:
            logit_p = self.fc_nonzero(features)
            log_scale = self.fc_scale(features)
            # log_shape = self.fc_shape(features)

            # pred = (torch.sigmoid(logit_p)
            #         * torch.exp(log_scale)
            #         * torch.exp(log_shape))
            # out = torch.where(p < 0.5, torch.zeros_like(out), out)
            return (
                torch.squeeze(logit_p),
                torch.squeeze(log_scale),
                # torch.squeeze(log_shape),
            )

        else:
            out = self.fc_out(features)
            if self.link:
                out = self.link(out)
        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)

        return torch.squeeze(out)
