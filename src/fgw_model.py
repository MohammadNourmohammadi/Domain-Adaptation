"""Top-level model for the FGW prototype-graph DA method.

This module is intentionally thin: it just bundles the shared GCN
encoder and the learnable PrototypeBank so they share an optimizer and
a `state_dict`. The ego-graph caches (one per data graph) live outside
the module because they are derived from data and would otherwise
pollute checkpoints.
"""

import torch
import torch.nn as nn

from .fgw_encoder import SharedGCNEncoder
from .fgw_prototypes import PrototypeBank


class FGWPrototypeDA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_protos: int,
        proto_size: int,
        anchor_weight: float,
        adjacency_temp: float,
    ):
        super().__init__()
        self.encoder = SharedGCNEncoder(in_dim, proj_dim, hidden_dim)
        self.prototypes = PrototypeBank(
            num_classes=num_classes,
            num_protos=num_protos,
            proto_size=proto_size,
            embed_dim=hidden_dim,
            anchor_weight=anchor_weight,
            adjacency_temp=adjacency_temp,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)
