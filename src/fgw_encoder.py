"""Shared GCN encoder f_theta used by the FGW prototype-graph pipeline.

Same architecture as the encoder inside `CausalGNN_DANN` (BoW linear
projection followed by two GCN layers) but kept standalone so the FGW
method does not import from the existing model. Returns node embeddings
h_v in R^d, where d = `hidden_dim`.

A final LayerNorm puts every embedding on a unit-ish scale. This matters
for the FGW geometry: it keeps the ego features comparable to the
prototype embeddings (which are initialised at the same scale) so the
feature half of the FGW cost reflects direction, not magnitude offsets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SharedGCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        hidden_dim: int,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, proj_dim)
        self.conv1 = GCNConv(proj_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return self.norm(h)
