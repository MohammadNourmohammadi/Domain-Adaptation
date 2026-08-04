"""Label-shift estimation and FGW-native source selection.

Two label-free estimators live here. Neither ever touches a target label.

`estimate_target_prior`
    Black-Box Shift Estimation (Lipton, Wang & Smola, ICML 2018) on the
    held-out source split. Twitch has a hard label shift — the five sources
    pool to a 0.473 positive rate against RU's 0.245 — and both the IM
    balance term and the inference-time prior correction need a number for it.

`transferability`
    Per-source-domain (`s_global`) and per-source-node (`s_local`) weights,
    read directly off the FGW geometry.

    This is the part that answers SelMAG's central claim. Their §4.2 estimates
    the same two quantities by pretraining three self-supervised pretext tasks
    (masked-attribute, edge and context prediction) on every graph, feeding
    the transferred losses into a learned selector network, and training that
    selector through a MAML bilevel loop over virtual source/target splits.

    The observation here is that an FGW pipeline already carries a metric
    between ego-graphs, so the same two quantities come out of one batched
    distance matrix with no pretext tasks, no selector parameters and no outer
    loop. For each source j we compute the FGW distances D_j between its
    labeled ego-graphs and a fixed sample of target ego-graphs, then read off

        s_global(j) ∝ exp( -mean_t min_s D_j[s, t] / temp )   -- directed
                                                                 Chamfer:
            how well source j's ego-graphs *cover* the target's. A source
            containing no neighbourhood that looks like anything in the target
            gets a large score here and is down-weighted.

        s_local(v) ∝ exp( -min_t D_j[v, t] / temp )
            how close one source node's own neighbourhood sits to the target
            manifold. This is SelMAG's within-domain variation term.

    Both are then applied the way SelMAG applies theirs (their Eq. 13, which
    scales the transport cost by s_global * s_local): here they weight the
    supervised source terms that shape the shared prototype bank, which is the
    channel through which source data reaches the target in this architecture.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as Fnn
from torch_geometric.data import Data

from .fgw_distance import pairwise_fgw_distances
from .fgw_ego import build_ego_batch_from_cache


# --------------------------------------------------------------- label shift
@torch.no_grad()
def class_prior(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Empirical class distribution of `labels`, with no zero entries."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    return (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-6)


@torch.no_grad()
def estimate_target_prior(
    model,
    sources: List[Data],
    src_val: List[torch.Tensor],
    target: Data,
    num_classes: int,
    source_prior: torch.Tensor,
    min_prob: float = 0.01,
    max_cond: float = 4.0,
) -> Tuple[Optional[torch.Tensor], str]:
    """BBSE estimate of the target class prior. Returns (prior, note).

    Uses hard predictions, which is the standard BBSE-hard variant and is far
    better conditioned than the soft one when the classifier is only mediocre:

        Ĉ[i, j] = P̂(ŷ = i, y = j)   on the held-out *source* split
        μ̂[i]    = P̂(ŷ = i)          on the *target* graph
        solve Ĉ w = μ̂     then      q_j = w_j * p_src(j)

    `w_j = q(j)/p_src(j)` are the label-shift importance weights. The identity
    holds under the label-shift assumption `p(x | y)` is domain-invariant,
    which is exactly the assumption a class-conditional alignment method is
    already making.

    Returns `(None, reason)` when the estimate cannot be trusted — a
    near-singular confusion matrix, or a solution that lands outside the
    simplex by more than the clamp can absorb. Callers fall back to the source
    prior, which is a strictly better guess than uniform.
    """
    if not any(v.numel() for v in src_val):
        return None, "no held-out source split"

    model.eval()
    pred_list, true_list = [], []
    for s, val in zip(sources, src_val):
        if val.numel() == 0:
            continue
        logits = model.classify(model.encode(s.x, s.edge_index))[val]
        pred_list.append(logits.argmax(dim=1))
        true_list.append(s.y[val])
    pred = torch.cat(pred_list)
    true = torch.cat(true_list)

    n = pred.numel()
    joint = torch.zeros(num_classes, num_classes, device=pred.device)
    joint.index_put_((pred, true), torch.ones(n, device=pred.device), accumulate=True)
    joint /= n

    tgt_pred = model.classify(model.encode(target.x, target.edge_index)).argmax(dim=1)
    mu = torch.bincount(tgt_pred, minlength=num_classes).float() / tgt_pred.numel()

    # The solve runs on CPU in float64: it is a C x C system (free), the extra
    # precision matters because the inverse amplifies estimation noise, and MPS
    # supports neither float64 nor `linalg.lstsq`.
    joint64 = joint.detach().cpu().double()
    mu64 = mu.detach().cpu().double()

    # **Reliability gate.** BBSE inverts the confusion matrix, so its variance
    # blows up as the classifier approaches chance — and the failure is not
    # symmetric: it systematically over-states the majority class, which is the
    # one direction that actively encourages the collapse the prior is supposed
    # to prevent. On Twitch (target AUROC ~0.55) it returned [0.844, 0.156]
    # against a truth of [0.755, 0.245], and feeding that to the IM balance term
    # pushed the model further toward the majority than no correction at all.
    #
    # The gate is on the *conditional* confusion P(pred | y), whose conditioning
    # is a direct read on class separability (for two classes cond = 1/Youden's
    # J). Simulated sweeps put reliable estimates near cond 1.7 and unusable
    # ones at 8+, so the default cutoff of 4 (J >= 0.25) admits classifiers that
    # genuinely separate and rejects the rest. Rejection is not a failure mode:
    # the caller falls back to the pooled source prior, which is both a decent
    # guess and biased in the safe direction.
    col = joint64.sum(dim=0).clamp_min(1e-9)
    conditional = joint64 / col.unsqueeze(0)
    try:
        cond = torch.linalg.cond(conditional).item()
    except Exception:
        return None, "confusion matrix not invertible"
    if not (cond == cond) or cond > max_cond:
        return None, (f"classifier too close to chance for BBSE "
                      f"(cond {cond:.2f} > {max_cond:.2f})")

    w = torch.linalg.lstsq(joint64, mu64.unsqueeze(1)).solution.squeeze(1)
    q = (w * source_prior.detach().cpu().double()).float().to(source_prior.device)

    if not torch.isfinite(q).all():
        return None, "non-finite BBSE solution"
    # Small negatives are ordinary estimation noise; a large one means the
    # linear system was solved outside the simplex and the estimate is junk.
    if q.min() < -0.25:
        return None, f"BBSE solution off-simplex (min {q.min().item():.2f})"

    q = q.clamp_min(min_prob)
    return q / q.sum(), "ok"


@torch.no_grad()
def prior_corrected_logits(
    logits: torch.Tensor,
    source_prior: torch.Tensor,
    target_prior: Optional[torch.Tensor],
) -> torch.Tensor:
    """Re-base source-trained logits onto the target prior.

    Under label shift `p(x | y)` is shared, so

        q(y = c | x) ∝ p(y = c | x) * q(c) / p(c)

    i.e. add `log q(c) - log p(c)` to the logits. Argmax on the raw logits is
    a decision rule calibrated to the *source* prior, which on Twitch means a
    rule tuned for 47% positives being applied to a graph with 24.5% — and
    macro-F1 at that imbalance is dominated by where the decision boundary
    sits. This is the cheapest real gain available, and it uses no target
    labels: `target_prior` comes from BBSE.
    """
    if target_prior is None:
        return logits
    shift = (target_prior.clamp_min(1e-8).log()
             - source_prior.clamp_min(1e-8).log())
    return logits + shift.view(1, -1)


# ------------------------------------------------------- FGW transferability
@torch.no_grad()
def _ego_fgw_matrix(
    src_F, src_C, src_h, tgt_F, tgt_C, tgt_h, cfg, chunk: int,
) -> torch.Tensor:
    """FGW distances between two sets of ego-graphs -> (n_src, n_tgt).

    Reuses the batched prototype solver by presenting the target ego-graphs as
    a single "class" of `n_tgt` prototype slots. Their mass vectors are all
    the same function of `ego_size` (uniform plus the anchor boost), so one
    shared `q` is exact rather than an approximation.
    """
    rows = []
    q = tgt_h[0]
    Fp = tgt_F.unsqueeze(0)                      # (1, n_tgt, k, d+1)
    Cp = tgt_C.unsqueeze(0)                      # (1, n_tgt, k, k)
    for start in range(0, src_F.size(0), chunk):
        end = min(start + chunk, src_F.size(0))
        d = pairwise_fgw_distances(
            src_F[start:end], src_C[start:end], src_h[start:end],
            Fp, Cp, q,
            alpha=cfg.fgw_alpha, epsilon=cfg.fgw_epsilon,
            max_iter=cfg.fgw_max_iter,
        )                                        # (chunk, 1, n_tgt)
        rows.append(d.squeeze(1))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def transferability(
    model,
    sources: List[Data],
    src_caches: list,
    src_labeled: List[torch.Tensor],
    target: Data,
    tgt_cache,
    tgt_sample: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """(s_global (K,), s_local [(n_j,) per source], raw domain distances (K,)).

    `s_global` is normalised to mean 1 over the K sources and `s_local` to
    mean 1 within each source, so enabling selection reweights the *relative*
    contributions without changing the overall loss scale.

    `tgt_sample` should be a fixed set of target nodes so the numbers stay
    comparable between refreshes.
    """
    model.eval()
    device = target.x.device

    tgt_emb = model.encode(target.x, target.edge_index)
    tF, tC, th = build_ego_batch_from_cache(
        tgt_cache, tgt_emb, tgt_sample,
        anchor_weight=cfg.anchor_weight, anchor_mass_extra=cfg.anchor_mass_extra,
    )

    chunk = max(int(getattr(cfg, "select_chunk", 256)), 1)
    temp_g = float(getattr(cfg, "select_temp_global", 0.1))
    temp_l = float(getattr(cfg, "select_temp_local", 0.25))
    cap = float(getattr(cfg, "select_clip", 5.0))

    # A directed Chamfer distance shrinks monotonically with the number of
    # source ego-graphs it may choose from, so comparing raw values across
    # domains of different size measures *size*, not transferability. The
    # labeled counts here span 5x (DE 760 vs PTBR 153) and the unmatched
    # version duly ranked the domains almost exactly by node count. Every
    # domain therefore contributes the same number of rows to `s_global`.
    n_match = min(int(lab.numel()) for lab in src_labeled)

    domain_d, node_d = [], []
    for s, cache, labeled in zip(sources, src_caches, src_labeled):
        emb = model.encode(s.x, s.edge_index)
        sF, sC, sh = build_ego_batch_from_cache(
            cache, emb, labeled,
            anchor_weight=cfg.anchor_weight,
            anchor_mass_extra=cfg.anchor_mass_extra,
        )
        D = _ego_fgw_matrix(sF, sC, sh, tF, tC, th, cfg, chunk)   # (n_j, n_tgt)
        # per source node: distance to its nearest target neighbourhood.
        # Uses every row — this one is a per-node quantity, so it has no
        # size bias to correct for.
        node_d.append(D.min(dim=1).values)
        # per domain: how well this source covers the target (directed
        # Chamfer), over a size-matched row sample.
        rows = (torch.randperm(D.size(0), device=D.device)[:n_match]
                if D.size(0) > n_match else slice(None))
        domain_d.append(D[rows].min(dim=0).values.mean())

    domain_d = torch.stack(domain_d)                              # (K,)
    K = domain_d.numel()
    # Centre before the exponential so the scores depend on the *spread* of the
    # domain distances, not on their absolute size (which drifts as the encoder
    # trains and would otherwise make the temperature meaningless).
    s_global = torch.softmax(-(domain_d - domain_d.mean()) / max(temp_g, 1e-6), dim=0)
    s_global = (s_global * K).clamp(1.0 / cap, cap)

    pooled = torch.cat(node_d)
    centre, scale = pooled.median(), pooled.std().clamp_min(1e-6)
    s_local = [
        torch.exp(-(d - centre) / (scale * max(temp_l, 1e-6) / 0.25)).clamp(1.0 / cap, cap)
        for d in node_d
    ]
    s_local = [w / w.mean().clamp_min(1e-8) for w in s_local]
    return s_global.to(device), [w.to(device) for w in s_local], domain_d


@torch.no_grad()
def mean_ego_structure(caches: list) -> float:
    """Mean off-diagonal entry of the cached ego structure matrices.

    Used to calibrate the prototype edge density so that `mean(C_p)` matches
    `mean(C_ego)`. Without that calibration the FGW structural term compares
    two matrices living on different scales and degenerates into a
    class-independent constant offset.
    """
    tot, n = 0.0, 0
    for c in caches:
        if c is None or c._struct_dev is None:
            continue
        C = c._struct_dev
        k = C.size(-1)
        if k < 2:
            continue
        off = (C.sum(dim=(-1, -2))) / (k * (k - 1))
        tot += float(off.sum())
        n += off.numel()
    return tot / n if n else 0.5
