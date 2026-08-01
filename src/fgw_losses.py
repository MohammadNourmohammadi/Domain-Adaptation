"""Loss components for the FGW prototype-graph DA pipeline.

Each function is a small, self-contained piece. The training loop in
`fgw_train.py` is responsible for combining them with the configured
weights and ramp schedule. Keeping the losses isolated also makes it
trivial to ablate any single term.

Symbols follow the method note:
    L_cls      supervised cross-entropy on the sources
    L_align    target alignment to prototype manifolds (DEC-style)
    L_ent      information maximisation on the target
    L_sep      cosine repulsion between different classes' prototypes
    L_pl       confidence-thresholded pseudo-label cross-entropy
    L_vrex     variance of per-source risks
    L_struct   L1 penalty on prototype soft adjacencies
"""

import torch
import torch.nn.functional as F


_EPS = 1e-8


# --------------------------------------------------------------------- 1
def cls_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y)


# --------------------------------------------------------------------- 2
def align_loss(d_bc: torch.Tensor, p_bc: torch.Tensor, tau: float) -> torch.Tensor:
    """L_align = mean_v KL(q(.|v) || softmax(-d(v)/tau)), up to a constant.

    DEC-style soft assignment: q_ic ∝ p_ic^2 / sum_i p_ic, then row-norm.
    The square sharpens the assignment and the per-class division
    self-balances classes.

    The score is the *relative* ordering of the class distances, not their
    magnitude. The earlier form `sum_c q_c * d_c` was minimised by shrinking
    every d_c alike: its gradient w.r.t. every distance was non-negative, so
    nothing was ever pushed apart and the cheapest descent direction was to
    park all prototypes on the centroid of the embedding cloud, at which
    point d_c(v) carries no class information at all. `log_softmax(-d/tau)`
    is unchanged by d -> d + const, so uniform shrinkage now buys nothing;
    the only way down is d_ĉ(v) < d_{c≠ĉ}(v).

    `tau` is the same soft-min temperature used by `fgw_logits`, so this
    scores the target exactly as `L_proto` scores the sources.
    """
    with torch.no_grad():
        f_c = p_bc.sum(dim=0).clamp_min(_EPS)
        q = (p_bc ** 2) / f_c
        q = q / q.sum(dim=1, keepdim=True).clamp_min(_EPS)
    log_post = F.log_softmax(-d_bc / tau, dim=1)
    return -(q * log_post).sum(dim=1).mean()


# --------------------------------------------------------------------- 3
def im_loss(p_bc: torch.Tensor) -> torch.Tensor:
    """Conditional-entropy minimisation: sharpen each node's prediction.

    L_ent = mean_v H(p(.|v))

    Only the individual (per-node) entropy term is used: minimising it makes
    every target prediction confident. The marginal / class-balance term has
    been removed, so this loss imposes no constraint on the overall predicted
    class mix.
    """
    return -(p_bc * (p_bc + _EPS).log()).sum(dim=1).mean()


# --------------------------------------------------------------------- 4
def separation_loss(proto_Z: torch.Tensor) -> torch.Tensor:
    """Cosine repulsion between the mean embeddings of different classes.

    `proto_Z` is (C, M, n_p, d): the prototype node embeddings. Each (class,
    slot) is summarised by its mean node embedding, L2-normalised, and the loss
    is the mean positive cosine similarity over every *cross-class* pair.

    This replaces an FGW-based hinge, `relu(margin - inter_fgw_distance)` with
    `margin = 1.0`, for two reasons:

    * **It was unsatisfiable.** Inter-prototype FGW distance is bounded by the
      embedding scale (LayerNorm'd, so O(1)) and the structure scale (C_p in
      [0, 1]), while `L_align` and `L_proto` both pull every prototype onto the
      same data manifold. The hinge therefore never reached zero and its value
      grew roughly linearly with the epoch count — ~0.00095 x epoch, i.e. a
      clock rather than a loss, reaching 58% of the total objective at epoch
      1000 while the supervised term was 1.6% of it. Worse, the only way the
      optimiser could reduce it was to inflate prototype magnitudes, which
      saturated `fgw_logits` and periodically detonated the run through
      `L_proto`'s gradient into the shared encoder.
    * **It cost a (C*M) x (C*M) FGW solve every step**, which dominated the
      per-step compute for a term that measured nothing.

    The cosine form is bounded in [0, 1], is exactly satisfiable at 0 (any
    arrangement where distinct classes' prototype means are orthogonal or
    opposed), and is ~1000x cheaper. Within-class slots are deliberately left
    unconstrained: their diversity is what the M-way soft-min in
    `fgw_class_distances` is for, and the old `intra_margin` hinge only ever
    fought the class-level signal.
    """
    C, M = proto_Z.shape[:2]
    mu = F.normalize(proto_Z.mean(dim=2), dim=-1)                  # (C, M, d)
    S = torch.einsum("cmd,end->cmen", mu, mu)                      # (C, M, C, M)
    same = (
        torch.eye(C, device=proto_Z.device, dtype=torch.bool)
        .view(C, 1, C, 1)
        .expand_as(S)
    )
    inter = S[~same]
    if inter.numel() == 0:                 # single class: nothing to separate
        return proto_Z.new_zeros(())
    return F.relu(inter).mean()


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
