"""Top-level model for the FGW prototype-graph DA method.

This module bundles three pieces that share an optimizer and a
`state_dict`:

  * the shared GCN encoder f_theta;
  * a parametric `ClassifierHead` g_psi that maps a node embedding to
    class logits — this is what produces predictions and is trained with
    the supervised source cross-entropy. Decoupling prediction from the
    FGW distances is what stops the alignment objective from being able
    to trivially flatten the classifier (the ln-2 collapse);
  * the learnable `PrototypeBank`, now used for *transfer* only: FGW
    distances to the prototypes drive the target-alignment loss and an
    auxiliary source term that keeps the prototypes class-meaningful.

The ego-graph caches (one per data graph) live outside the module
because they are derived from data and would otherwise pollute
checkpoints.
"""

import torch
import torch.nn as nn

from .fgw_encoder import SharedGCNEncoder
from .fgw_prototypes import PrototypeBank


class ClassifierHead(nn.Module):
    """Parametric 2-layer MLP head on top of the encoder embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


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
        head_hidden: int = 32,
        head_dropout: float = 0.5,
        use_layernorm: bool = True,
        embed_init_scale: float = 1.0,
    ):
        super().__init__()
        self.encoder = SharedGCNEncoder(
            in_dim, proj_dim, hidden_dim, use_layernorm=use_layernorm,
        )
        self.head = ClassifierHead(
            in_dim=hidden_dim,
            hidden_dim=head_hidden,
            num_classes=num_classes,
            dropout=head_dropout,
        )
        self.prototypes = PrototypeBank(
            num_classes=num_classes,
            num_protos=num_protos,
            proto_size=proto_size,
            embed_dim=hidden_dim,
            anchor_weight=anchor_weight,
            adjacency_temp=adjacency_temp,
            embed_init_scale=embed_init_scale,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def classify(self, emb: torch.Tensor) -> torch.Tensor:
        """Class logits from node embeddings via the parametric head."""
        return self.head(emb)
