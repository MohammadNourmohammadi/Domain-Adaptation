"""Shared GCN encoder f_theta used by the FGW prototype-graph pipeline.

Same architecture as the encoder inside `CausalGNN_DANN` (BoW projection
followed by two GCN layers) but kept standalone so the FGW method does not
import from the existing model. Returns node embeddings h_v in R^d, where
d = `hidden_dim`.

The input projection can be either

  * a learned ``nn.Linear(in_dim, proj_dim)`` (the original), or
  * a **frozen** basis installed via `set_projection` (``frozen_proj=True``).

The frozen path exists because the learned one dominates the parameter
count and overfits almost immediately: with a 3170-dim BoW input and a
10% label budget, ``Linear(3170, 64)`` is 202,880 parameters against a
couple of thousand labeled nodes — ~100 parameters per labeled example,
and >90% of the model. `svd_projection` replaces it with the top
`proj_dim` right singular vectors of the *unlabeled* feature matrix
(every source and target graph stacked), which is fitted without touching
a single label and leaves nothing to memorise: 202,880 trainable
parameters become 0 in that layer.

A final LayerNorm puts every embedding on a unit-ish scale. This matters
for the FGW geometry: it keeps the ego features comparable to the
prototype embeddings (which are initialised at the same scale) so the
feature half of the FGW cost reflects direction, not magnitude offsets.
"""

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


@torch.no_grad()
def svd_projection(
    features: Iterable[torch.Tensor], proj_dim: int, niter: int = 4,
) -> torch.Tensor:
    """Top-`proj_dim` right singular vectors of the stacked feature matrix.

    ``features`` is every graph's ``x`` — sources *and* target. This is a
    purely unsupervised fit (no labels, no train/val distinction), so using
    the target features here does not leak anything a transductive DA setup
    does not already assume: the target graph is available at training time,
    only its labels are hidden.

    The basis is rescaled so the projected features have unit overall
    standard deviation. Raw singular directions carry wildly different
    magnitudes (S_1 can be orders of magnitude above S_128), and feeding
    that straight into a GCN would make the first component drown out the
    rest; folding one global scalar into the basis fixes the input scale
    while keeping the *relative* component weighting, which is informative.
    """
    X = torch.cat([f.detach().to(device="cpu", dtype=torch.float32) for f in features])
    in_dim = X.size(1)
    q = min(proj_dim, *X.shape)
    _, _, V = torch.svd_lowrank(X, q=q, niter=niter)      # V: (in_dim, q)

    basis = X.new_zeros(in_dim, proj_dim)
    basis[:, :q] = V                                      # pad if rank < proj_dim
    std = (X @ basis[:, :q]).std()
    if std > 0:
        basis /= std
    return basis


class SharedGCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        hidden_dim: int,
        use_layernorm: bool = True,
        frozen_proj: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.frozen_proj = frozen_proj
        if frozen_proj:
            self.input_proj = None
            # Buffers, not parameters: they ride along in the state_dict (so a
            # restored checkpoint keeps its basis) but never see a gradient.
            self.register_buffer("proj_basis", torch.zeros(in_dim, proj_dim))
            self.register_buffer("proj_ready", torch.zeros((), dtype=torch.bool))
        else:
            self.input_proj = nn.Linear(in_dim, proj_dim)
        self.conv1 = GCNConv(proj_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.out_dim = hidden_dim

    @torch.no_grad()
    def set_projection(self, basis: torch.Tensor) -> None:
        """Install the frozen input basis (see `svd_projection`)."""
        if not self.frozen_proj:
            raise RuntimeError(
                "set_projection() requires the encoder to be built with "
                "frozen_proj=True; this one has a learned input_proj."
            )
        if tuple(basis.shape) != (self.in_dim, self.proj_dim):
            raise ValueError(
                f"projection basis has shape {tuple(basis.shape)}, expected "
                f"{(self.in_dim, self.proj_dim)}"
            )
        self.proj_basis.copy_(basis.to(self.proj_basis))
        self.proj_ready.fill_(True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.frozen_proj:
            if not bool(self.proj_ready):
                raise RuntimeError(
                    "frozen input projection was never set; call "
                    "encoder.set_projection(svd_projection(...)) first."
                )
            # No ReLU here: a fixed orthonormal basis produces signed
            # coordinates, and rectifying them would throw away half of every
            # component. The GCN layers below still supply the nonlinearity.
            h = x @ self.proj_basis
        else:
            h = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return self.norm(h)
