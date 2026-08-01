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
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_protos: int,
        proto_size: int,
        embed_dim: int,
        anchor_weight: float,
        adjacency_temp: float,
        embed_init_scale: float = 1.0,
    ):
        super().__init__()
        self.C = num_classes
        self.M = num_protos
        self.n_p = proto_size
        self.d = embed_dim
        self.anchor_weight = anchor_weight
        self.temp = adjacency_temp

        # Init at the encoder's (LayerNorm'd) embedding scale so the FGW
        # feature cost isn't dominated by a constant magnitude offset.
        Z = torch.randn(num_classes, num_protos, proto_size, embed_dim) * embed_init_scale
        self.Z = nn.Parameter(Z)

        # Wide init so sigmoid(E/temp) spreads C_p across roughly [0.12, 0.88]
        # rather than collapsing at ~0.5. This matches the range of the ego
        # graph's diameter-normalised shortest-path distances, without which the
        # FGW structural half is a class-independent constant offset.
        E = torch.randn(num_classes, num_protos, proto_size, proto_size) * 2.0
        self.E_logits = nn.Parameter(E)

    def embeddings(self) -> torch.Tensor:
        """(C, M, n_p, d) prototype embeddings, on the encoder's scale.

        The raw parameter `Z` is unconstrained in magnitude, and the FGW
        feature cost `||F_e - F_p||^2 / D` is quadratic in it: any pressure
        that inflates `Z` inflates the distances, which then saturates
        `fgw_logits` and sends very large gradients back into the shared
        encoder through `L_proto`. LayerNorm here is the same normalisation
        the encoder applies to its own output, so the two sides of the FGW
        comparison stay on one scale for the whole run and the prototypes can
        only move in *direction*, which is the part that carries class
        information.
        """
        return F.layer_norm(self.Z, (self.d,))

    def features(self) -> torch.Tensor:
        """(C, M, n_p, d + 1) features with the anchor indicator on node 0."""
        device = self.Z.device
        anchor = torch.zeros(self.C, self.M, self.n_p, 1, device=device)
        anchor[:, :, 0, 0] = self.anchor_weight
        return torch.cat([self.embeddings(), anchor], dim=-1)

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
