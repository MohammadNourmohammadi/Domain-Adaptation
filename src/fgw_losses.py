"""Loss components for the FGW prototype-graph DA pipeline.

Each function is a small, self-contained piece. The training loop in
`fgw_train.py` is responsible for combining them with the configured
weights and ramp schedule. Keeping the losses isolated also makes it
trivial to ablate any single term.

Symbols follow the method note:
    L_cls      supervised cross-entropy on the sources
    L_sep      inter-class prototype margin + intra-class decorrelation
    L_pl       confidence-thresholded pseudo-label cross-entropy
    L_vrex     variance of per-source risks
    L_struct   L1 penalty on prototype soft adjacencies

The target-alignment (L_align) and information-maximisation (L_ent) terms
have been removed: the only target signal left is the confidence-thresholded
pseudo-label CE used in the refine phase.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F


_EPS = 1e-8


# --------------------------------------------------------------------- 1
def cls_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


# --------------------------------------------------------------------- 2
def separation_loss(
    proto_F: torch.Tensor,        # (C, M, n_p, d+1)
    proto_C: torch.Tensor,        # (C, M, n_p, n_p)
    proto_q: torch.Tensor,        # (n_p,)
    alpha: float,
    epsilon: Optional[float],
    max_iter: int,
    margin: float,
    pairwise_fn: Callable,
    intra_margin: float = 0.5,
) -> torch.Tensor:
    """Push inter-class prototypes apart; keep within-class ones diverse.

    Reuses the same `pairwise_fgw_distances` machinery so the geometry
    of "prototype-vs-prototype" matches "ego-vs-prototype".
    """
    C, M, n_p, d_plus = proto_F.shape

    F_flat = proto_F.reshape(C * M, n_p, d_plus)
    C_flat = proto_C.reshape(C * M, n_p, n_p)
    h_flat = proto_q.unsqueeze(0).expand(C * M, n_p)

    dists = pairwise_fn(
        F_flat, C_flat, h_flat,
        proto_F, proto_C, proto_q,
        alpha=alpha, epsilon=epsilon, max_iter=max_iter,
    )  # (C*M, C, M)

    device = dists.device
    inter_mask = torch.ones(C * M, C, M, device=device)
    intra_mask = torch.zeros(C * M, C, M, device=device)
    for c1 in range(C):
        for m1 in range(M):
            i = c1 * M + m1
            inter_mask[i, c1, :] = 0.0
            intra_mask[i, c1, :] = 1.0
            intra_mask[i, c1, m1] = 0.0  # exclude self-pair

    inter_sum = inter_mask.sum().clamp_min(1.0)
    intra_sum = intra_mask.sum().clamp_min(1.0)
    inter = (dists * inter_mask).sum() / inter_sum
    intra = (dists * intra_mask).sum() / intra_sum

    # Two bounded hinges: push inter-class prototypes apart up to `margin`,
    # and penalise within-class prototypes only while they are *closer*
    # than `intra_margin`. The old `- intra` term was unbounded below and
    # perversely rewarded spreading within-class prototypes without limit.
    return F.relu(margin - inter) + F.relu(intra_margin - intra)


# --------------------------------------------------------------------- 3
def pseudo_label_loss(
    logits: torch.Tensor, p_bc: torch.Tensor, threshold: float,
) -> torch.Tensor:
    conf, pred = p_bc.max(dim=1)
    mask = conf >= threshold
    if mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask], pred[mask])


# --------------------------------------------------------------------- 4
def vrex_loss(per_source_losses: torch.Tensor) -> torch.Tensor:
    return per_source_losses.var(unbiased=False)


# --------------------------------------------------------------------- 5
def struct_l1_loss(A: torch.Tensor) -> torch.Tensor:
    return A.abs().mean()
