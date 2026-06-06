"""Batched Fused Gromov-Wasserstein distance.

Thin wrapper around POT. For each (ego-graph, class, prototype) triplet
we build the feature cost matrix M_ij = ||F_e[i] - F_p[j]||^2 and call
either the conditional-gradient FGW solver (`fused_gromov_wasserstein2`,
default) or its entropic Sinkhorn variant when `epsilon` is set.

Returns a tensor of shape (B, C, M) carrying gradients to:
  - F_e (and therefore the encoder weights, raw features, and inputs);
  - F_p (the prototype embeddings);
  - C_p (the prototype structure matrix, via its soft adjacency).
"""

from typing import Optional

import ot
import torch


def pairwise_fgw_distances(
    F_e: torch.Tensor,        # (B, k_e, d+1)
    C_e: torch.Tensor,        # (B, k_e, k_e)
    h_e: torch.Tensor,        # (B, k_e)
    F_p: torch.Tensor,        # (C, M, n_p, d+1)
    C_p: torch.Tensor,        # (C, M, n_p, n_p)
    q: torch.Tensor,          # (n_p,)
    alpha: float = 0.5,
    epsilon: Optional[float] = None,
    max_iter: int = 50,
) -> torch.Tensor:
    """Returns FGW distances of shape (B, C, M).

    The solve runs in-place on whatever device the inputs already live on
    (e.g. CUDA), so no host/device transfers are needed.
    """
    B = F_e.shape[0]
    C_cls, M = F_p.shape[:2]
    out = F_e.new_zeros(B, C_cls, M)

    use_entropic = epsilon is not None and epsilon > 0
    for b in range(B):
        Fb, Cb, hb = F_e[b], C_e[b], h_e[b]
        for c in range(C_cls):
            for m in range(M):
                Fp = F_p[c, m]
                Cp = C_p[c, m]
                Mcost = torch.cdist(Fb, Fp) ** 2
                if use_entropic:
                    d = ot.gromov.entropic_fused_gromov_wasserstein2(
                        Mcost, Cb, Cp, hb, q,
                        loss_fun="square_loss",
                        alpha=alpha,
                        epsilon=epsilon,
                        max_iter=max_iter,
                    )
                else:
                    d = ot.gromov.fused_gromov_wasserstein2(
                        Mcost, Cb, Cp, hb, q,
                        loss_fun="square_loss",
                        alpha=alpha,
                        max_iter=max_iter,
                    )
                out[b, c, m] = d

    return out
