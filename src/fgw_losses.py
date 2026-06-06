"""Loss components for the FGW prototype-graph DA pipeline.

Each function is a small, self-contained piece. The training loop in
`fgw_train.py` is responsible for combining them with the configured
weights and ramp schedule. Keeping the losses isolated also makes it
trivial to ablate any single term.

Symbols follow the method note:
    L_cls      supervised cross-entropy on the sources
    L_align    target alignment to prototype manifolds (DEC-style)
    L_ent      information maximisation on the target
    L_sep      inter-class prototype margin + intra-class decorrelation
    L_pl       confidence-thresholded pseudo-label cross-entropy
    L_vrex     variance of per-source risks
    L_struct   L1 penalty on prototype soft adjacencies
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F


_EPS = 1e-8


# --------------------------------------------------------------------- 1
def cls_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


# --------------------------------------------------------------------- 2
def align_loss(d_bc: torch.Tensor, p_bc: torch.Tensor) -> torch.Tensor:
    """L_align = mean_v sum_c q(c|v) * d_c(v) with a sharpened, detached q.

    DEC-style soft assignment: q_ic ∝ p_ic^2 / sum_i p_ic, then row-norm.
    The square sharpens the assignment and the per-class division
    self-balances classes, which guards against the trivial collapse.
    """
    with torch.no_grad():
        f_c = p_bc.sum(dim=0).clamp_min(_EPS)
        q = (p_bc ** 2) / f_c
        q = q / q.sum(dim=1, keepdim=True).clamp_min(_EPS)
    return (q * d_bc).sum(dim=1).mean()


# --------------------------------------------------------------------- 3
def im_loss(p_bc: torch.Tensor, prior: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Information maximisation: sharpen individual preds, diversify mean.

    L_ent = mean_v H(p(.|v)) - H(mean_v p(.|v))
    If `prior` is given it replaces the uniform implicit prior in the
    second term, useful under heavy class imbalance.
    """
    ent_individual = -(p_bc * (p_bc + _EPS).log()).sum(dim=1).mean()
    p_mean = p_bc.mean(dim=0)
    if prior is None:
        ent_mean = -(p_mean * (p_mean + _EPS).log()).sum()
    else:
        ent_mean = -(p_mean * (prior + _EPS).log()).sum()
    return ent_individual - ent_mean


# --------------------------------------------------------------------- 4
def separation_loss(
    proto_F: torch.Tensor,        # (C, M, n_p, d+1)
    proto_C: torch.Tensor,        # (C, M, n_p, n_p)
    proto_q: torch.Tensor,        # (n_p,)
    alpha: float,
    epsilon: Optional[float],
    max_iter: int,
    margin: float,
    pairwise_fn: Callable,
) -> torch.Tensor:
    """Push inter-class prototypes apart; decorrelate within a class.

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

    # Margin hinge on inter (push apart up to a margin), minus intra
    # (encourage diversity within a class).
    return F.relu(margin - inter) - intra


# --------------------------------------------------------------------- 5
def pseudo_label_loss(
    logits: torch.Tensor, p_bc: torch.Tensor, threshold: float,
) -> torch.Tensor:
    conf, pred = p_bc.max(dim=1)
    mask = conf >= threshold
    if mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask], pred[mask])


# --------------------------------------------------------------------- 6
def vrex_loss(per_source_losses: torch.Tensor) -> torch.Tensor:
    return per_source_losses.var(unbiased=False)


# --------------------------------------------------------------------- 7
def struct_l1_loss(A: torch.Tensor) -> torch.Tensor:
    return A.abs().mean()
