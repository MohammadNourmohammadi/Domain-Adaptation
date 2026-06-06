"""Learnable bank of M prototype graphs per class.

For every (class c, slot m) we store a free node-embedding matrix and a
symmetric soft-adjacency logit matrix. The prototype anchor is always
node 0; its anchor indicator coordinate is appended at every forward so
gradients only flow through the learnable embedding dimensions.

We expose a soft adjacency A in [0, 1] and a structure matrix C^P
derived from it as `1 - A` (off-diagonal). This proxy is fully
differentiable and on the same [0, 1] scale as the ego-graph's
normalised shortest-path distances, which is what FGW compares.
"""

import torch
import torch.nn as nn


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_protos: int,
        proto_size: int,
        embed_dim: int,
        anchor_weight: float,
        adjacency_temp: float,
    ):
        super().__init__()
        self.C = num_classes
        self.M = num_protos
        self.n_p = proto_size
        self.d = embed_dim
        self.anchor_weight = anchor_weight
        self.temp = adjacency_temp

        Z = torch.randn(num_classes, num_protos, proto_size, embed_dim) * 0.1
        self.Z = nn.Parameter(Z)

        E = torch.randn(num_classes, num_protos, proto_size, proto_size) * 0.1
        self.E_logits = nn.Parameter(E)

    def features(self) -> torch.Tensor:
        """(C, M, n_p, d + 1) features with the anchor indicator on node 0."""
        device = self.Z.device
        anchor = torch.zeros(self.C, self.M, self.n_p, 1, device=device)
        anchor[:, :, 0, 0] = self.anchor_weight
        return torch.cat([self.Z, anchor], dim=-1)

    def soft_adjacency(self) -> torch.Tensor:
        """Symmetric soft adjacency in [0, 1] with a zero diagonal."""
        E = 0.5 * (self.E_logits + self.E_logits.transpose(-1, -2))
        A = torch.sigmoid(E / self.temp)
        eye = torch.eye(self.n_p, device=A.device)
        return A * (1.0 - eye)

    def structure_matrices(self) -> torch.Tensor:
        """(C, M, n_p, n_p) soft distance matrix, derived from A."""
        A = self.soft_adjacency()
        eye = torch.eye(self.n_p, device=A.device)
        return (1.0 - A) * (1.0 - eye)

    def uniform_mass(self, device) -> torch.Tensor:
        return torch.full((self.n_p,), 1.0 / self.n_p, device=device)
