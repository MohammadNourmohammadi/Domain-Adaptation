"""Shared GCN encoder f_theta used by the FGW prototype-graph pipeline.

Same architecture as the encoder inside `CausalGNN_DANN` (BoW linear
projection followed by two GCN layers) but kept standalone so the FGW
method does not import from the existing model. Returns node embeddings
h_v in R^d, where d = `hidden_dim`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SharedGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, proj_dim)
        self.conv1 = GCNConv(proj_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return h
