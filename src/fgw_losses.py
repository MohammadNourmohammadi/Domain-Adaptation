"""Loss components for the FGW prototype-graph DA pipeline.

Each function is a small, self-contained piece. The training loop in
`fgw_train.py` is responsible for combining them with the configured
weights and ramp schedule. Keeping the losses isolated also makes it
trivial to ablate any single term.

Symbols follow the method note:
    L_cls      supervised cross-entropy on the sources
    L_align    target alignment to prototype manifolds (DEC-style)
    L_ent      information maximisation on the target (entropy + prior match)
    L_margin   hinge pushing each source ego away from wrong-class prototypes
    L_sep      cosine repulsion between different classes' prototypes
    L_pl       confidence-thresholded pseudo-label cross-entropy
    L_vrex     variance of per-source risks
    L_struct   prototype soft-adjacency density penalty
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


_EPS = 1e-8


# --------------------------------------------------------------------- 1
def cls_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy, optionally with a per-example weight.

    ``weights`` carries the node-level transferability score s_local (see
    `fgw_select.transferability`): source nodes whose ego-graphs sit far
    from the target manifold contribute less. The weights are renormalised to
    mean 1 so turning selection on does not silently rescale the loss (and
    therefore the effective learning rate) relative to the unweighted run.
    """
    if weights is None:
        return F.cross_entropy(logits, y)
    per = F.cross_entropy(logits, y, reduction="none")
    w = weights / weights.mean().clamp_min(_EPS)
    return (per * w).mean()


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
def balance_scale(prior: torch.Tensor) -> float:
    """Multiplier that makes total collapse cost at least what it buys.

    Collapsing every target prediction onto the majority class frees `ln C` of
    conditional entropy and costs `KL(onehot || prior) = -log(max_c prior_c)`
    of balance penalty. Those are equal only when the prior is uniform — which
    is why the textbook IM weight of 1.0 works in the balanced setting and
    quietly fails everywhere else. Under Twitch's estimated target prior
    [0.844, 0.156] the collapse frees 0.693 and costs 0.170, a net *gain* of
    0.52, so the run walks straight back to the all-negative predictor even
    with the balance term present. That is exactly what the first full run did.

    Scaling by `ln C / -log(max prior)` restores the balance for any prior and
    is identically 1 for a uniform one, so this changes nothing in the
    balanced case. Capped, because a near-degenerate prior would otherwise
    send the weight to infinity.
    """
    p_max = float(prior.max().clamp(1e-6, 1 - 1e-6))
    C = prior.numel()
    return min(math.log(C) / -math.log(p_max), 20.0)


def im_loss(
    p_bc: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
    balance_weight: float = 1.0,
) -> torch.Tensor:
    """Information maximisation: sharpen each prediction, hold the class mix.

        L_ent = mean_v H(p(.|v))  +  w * KL(p̄ || prior)
        w     = balance_weight * balance_scale(prior)

    **The second term is not optional.** Without it this is pure conditional-
    entropy minimisation, whose global optimum is "predict one class with
    probability 1 for every target node" — there is nothing else in the
    objective opposing it. Dropping it is what collapsed the Twitch runs onto
    the all-negative predictor: `ent` fell 0.67 -> 0.01 while target macro-F1
    walked down to 0.434, which is exactly the score of predicting the
    majority class on a graph that is 24.5% positive.

    ``prior`` is the assumed target class distribution. With ``None`` it is
    uniform, and `KL(p̄ || u) = log C - H(p̄)`, recovering the textbook IM
    objective up to an additive constant. Uniform is the *wrong* assumption
    under label shift, though: pooled over the five Twitch sources the
    positive rate is 0.473 while RU is 0.245, so a uniform marginal actively
    pulls towards a 50/50 split. Pass the estimated target prior instead —
    see `fgw_select.estimate_target_prior` — and note that doing so *requires*
    the `balance_scale` correction to stay collapse-proof.

    Stated as a KL rather than as `-H(p̄)` so that both cases are one code
    path and the non-uniform case is a proper divergence (>= 0, zero exactly
    when the predicted mix matches the prior).
    """
    ent_individual = -(p_bc * (p_bc + _EPS).log()).sum(dim=1).mean()
    p_mean = p_bc.mean(dim=0)
    if prior is None:
        prior = p_mean.new_full(p_mean.shape, 1.0 / p_mean.numel())
    div = (p_mean * ((p_mean + _EPS).log() - (prior + _EPS).log())).sum()
    return ent_individual + balance_weight * balance_scale(prior) * div


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
      1000 while the supervised term was 1.6% of it.
    * **It cost a (C*M) x (C*M) FGW solve every step**, which dominated the
      per-step compute for a term that measured nothing.

    The cosine form is bounded in [0, 1], is exactly satisfiable at 0, and is
    ~1000x cheaper. It only constrains a summary statistic, though, which is
    not on its own enough to make the FGW distances class-discriminative —
    `fgw_margin_loss` is the term that does that job directly.
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
def fgw_margin_loss(
    d_bc: torch.Tensor,
    y: torch.Tensor,
    margin: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hinge on the *source-conditional* FGW class distances.

        mean_v relu( margin + d_y(v) - min_{c != y} d_c(v) )

    `L_proto` (cross-entropy on `-d/tau`) is supposed to make the prototypes
    class-meaningful, but on its own it never did: across a 350-epoch run
    `L_proto` sat at 0.687 against ln 2 = 0.693, i.e. d_0(v) ≈ d_1(v) for
    every ego graph, so the FGW branch carried no class information at all and
    `L_align` was pulling the target toward prototypes that could not tell the
    classes apart. The reason is that softmax cross-entropy has a vanishing
    gradient exactly where the run is stuck: at d_0 = d_1 the logit difference
    is 0, and the gradient it produces is proportional to that difference.

    A hinge does not vanish there. Below the margin it applies a constant-
    magnitude force separating the correct-class distance from the nearest
    wrong-class one, which is precisely the geometry `L_proto` needs and
    cannot bootstrap by itself. It is applied to the sources only, where the
    labels are real — this is not a pseudo-label term.

    ``weights`` is the optional per-node s_local (see `cls_loss`).
    """
    idx = y.view(-1, 1)
    d_true = d_bc.gather(1, idx).squeeze(1)                       # (B,)
    d_other = d_bc.scatter(1, idx, float("inf")).min(dim=1).values
    viol = F.relu(margin + d_true - d_other)
    if weights is None:
        return viol.mean()
    w = weights / weights.mean().clamp_min(_EPS)
    return (viol * w).mean()


# --------------------------------------------------------------------- 6
def pseudo_label_loss(
    logits: torch.Tensor, p_bc: torch.Tensor, threshold: float,
) -> torch.Tensor:
    conf, pred = p_bc.max(dim=1)
    mask = conf >= threshold
    if mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask], pred[mask])


# --------------------------------------------------------------------- 7
def vrex_loss(per_source_losses: torch.Tensor) -> torch.Tensor:
    return per_source_losses.var(unbiased=False)


# --------------------------------------------------------------------- 8
def struct_density_loss(A: torch.Tensor, target_density: float) -> torch.Tensor:
    """Pull each prototype's edge density towards `target_density`.

    The previous term was `A.abs().mean()`, plain L1 on the soft adjacency.
    Since the structure matrix the FGW solver actually sees is `C_p = 1 - A`,
    driving A -> 0 drives C_p -> all-ones: **a constant matrix, carrying zero
    structural information**. Weight decay was pushing the same way from the
    other side — it pulls `E_logits` -> 0, hence `sigmoid(0) = 0.5`, hence
    `C_p` -> 0.5 everywhere, also constant. Between them the two regularisers
    were erasing exactly the structure the method is built on, which is a
    large part of why the FGW distances never separated the classes.

    Targeting a density instead keeps the penalty (prototypes stay simple and
    do not saturate) while leaving the *pattern* of A free, which is the part
    that carries information. `run_training` calibrates `target_density` from
    the data so that `mean(C_p)` matches `mean(C_ego)`; putting both structure
    matrices on one scale is the precondition for the FGW structural term to
    mean anything.
    """
    n_p = A.size(-1)
    off_diag = n_p * (n_p - 1)
    if off_diag == 0:
        return A.new_zeros(())
    dens = A.flatten(start_dim=-2).sum(dim=-1) / off_diag          # (C, M)
    return (dens - target_density).abs().mean()
