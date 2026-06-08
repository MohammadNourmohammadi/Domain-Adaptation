"""Training loop for the FGW prototype-graph DA method.

Prediction is parametric: a 2-layer MLP head on the encoder produces the
class logits used for the supervised source loss, evaluation, IM and
pseudo-labels. The FGW prototype machinery is used for *transfer* only
(target alignment + an auxiliary source term that keeps the prototypes
class-meaningful). Decoupling the prediction head from the FGW distances
is what prevents the alignment objective from collapsing the classifier
to the uniform ln-2 fixed point.

Three phases (controlled by `cfg.warmup_frac`, `cfg.refine_frac`):

  1. WARM-UP    – L_cls + L_proto (+ L_sep + L_vrex + L_struct). The head
                  and the prototypes must become meaningful before any
                  target signal is introduced; aligning to random
                  prototypes just injects noise.
  2. ADAPT      – ramp lambda_align and lambda_ent from 0 to their full
                  values with the same sigmoid schedule used for GRL.
  3. REFINE     – additionally enable confidence-thresholded pseudo-label
                  cross-entropy on the target.

With `cfg.no_da` the prototypes and every target term are switched off,
leaving a pure encoder+head source-supervised baseline (the diagnostic
for reading the achievable in-domain ceiling).

Mini-batching is done over nodes, not graphs: every step encodes each
full graph once (cheap) and then samples `cfg.nodes_per_step` seeds
from each source / target to form FGW problems.
"""

import math
from typing import List

import torch
import torch.nn.functional as Fnn
from torch_geometric.data import Data

from .fgw_classifier import fgw_class_distances, fgw_logits
from .fgw_config import FGWConfig
from .fgw_distance import pairwise_fgw_distances
from .fgw_ego import EgoGraphCache, build_ego_batch_from_cache
from .fgw_losses import (
    align_loss,
    cls_loss,
    im_loss,
    pseudo_label_loss,
    separation_loss,
    struct_l1_loss,
    vrex_loss,
)
from .fgw_model import FGWPrototypeDA
from .utils import compute_metrics


# ---------------------------------------------------------------------- schedule
def _ramp(epoch: int, ramp_epochs: int) -> float:
    p = min(epoch / max(ramp_epochs, 1), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def _phase(epoch: int, total: int, warmup_frac: float, refine_frac: float) -> str:
    if epoch <= int(warmup_frac * total):
        return "warmup"
    if epoch <= int(refine_frac * total):
        return "adapt"
    return "refine"


# ---------------------------------------------------------------------- caches
def _make_cache(g: Data, cfg: FGWConfig, device: str) -> EgoGraphCache:
    cache = EgoGraphCache(
        edge_index=g.edge_index,
        num_nodes=g.num_nodes,
        ppr_alpha=cfg.ppr_alpha,
        ppr_iters=cfg.ppr_iters,
        ego_size=cfg.ego_size,
    )
    cache.precompute_all(batch_size=512, device=device)
    return cache


def _proto_tensors(model: FGWPrototypeDA, device):
    Fp = model.prototypes.features()             # (C, M, n_p, d+1)
    Cp = model.prototypes.structure_matrices()   # (C, M, n_p, n_p)
    q = model.prototypes.uniform_mass(device)
    return Fp, Cp, q


# ----------------------------------------------------------------- train step
def train_step(
    model: FGWPrototypeDA,
    sources: List[Data],
    target: Data,
    src_caches: List[EgoGraphCache],
    tgt_cache: EgoGraphCache,
    optimizer: torch.optim.Optimizer,
    cfg: FGWConfig,
    epoch: int,
) -> dict:
    model.train()
    device = target.x.device
    da = not cfg.no_da
    phase = _phase(epoch, cfg.epochs, cfg.warmup_frac, cfg.refine_frac) if da else "srconly"
    ramp = _ramp(epoch, cfg.ramp_epochs)
    align_w = cfg.lambda_align * ramp if (da and phase != "warmup") else 0.0
    ent_w = cfg.lambda_ent * ramp if (da and phase != "warmup") else 0.0
    pl_w = cfg.lambda_pl if (da and phase == "refine") else 0.0
    proto_w = cfg.lambda_proto if da else 0.0

    src_emb = [model.encode(s.x, s.edge_index) for s in sources]

    Fp = Cp = q = None
    if da:
        Fp, Cp, q = _proto_tensors(model, device)

    # ------------------------------------------------------ supervised sources
    # Prediction/CE go through the parametric head; the FGW distances feed an
    # auxiliary term (L_proto) that keeps the prototypes class-meaningful.
    per_src_losses = []          # head CE per source (drives L_cls and V-REx)
    per_src_proto = []           # FGW-prototype CE per source (anchoring)
    per_src_metrics = []
    for s, emb, cache in zip(sources, src_emb, src_caches):
        n = min(cfg.nodes_per_step, s.num_nodes)
        seeds = torch.randperm(s.num_nodes, device=device)[:n]
        y = s.y[seeds]
        head_logits = model.classify(emb[seeds])
        per_src_losses.append(cls_loss(head_logits, y))
        per_src_metrics.append(compute_metrics(head_logits, y))
        if proto_w > 0:
            Fe, Ce, he = build_ego_batch_from_cache(
                cache, emb, seeds,
                anchor_weight=cfg.anchor_weight,
                anchor_mass_extra=cfg.anchor_mass_extra,
            )
            d_bcm = pairwise_fgw_distances(
                Fe, Ce, he, Fp, Cp, q,
                alpha=cfg.fgw_alpha, epsilon=cfg.fgw_epsilon,
                max_iter=cfg.fgw_max_iter,
            )
            per_src_proto.append(cls_loss(fgw_logits(d_bcm, cfg.tau), y))
    L_cls = torch.stack(per_src_losses).mean()
    L_vrex = vrex_loss(torch.stack(per_src_losses))
    L_proto = (
        torch.stack(per_src_proto).mean() if per_src_proto
        else L_cls.new_zeros(())
    )

    # -------------------------------------------------- target align + IM (+ PL)
    zero = L_cls.new_zeros(())
    L_align = L_ent = L_pl = zero
    if da and (align_w > 0 or ent_w > 0 or pl_w > 0):
        tgt_emb = model.encode(target.x, target.edge_index)
        n_t = min(cfg.nodes_per_step, target.num_nodes)
        seeds_t = torch.randperm(target.num_nodes, device=device)[:n_t]
        head_logits_t = model.classify(tgt_emb[seeds_t])
        p_t = Fnn.softmax(head_logits_t, dim=1)

        prior = (
            torch.tensor(cfg.target_class_prior, device=device, dtype=p_t.dtype)
            if cfg.target_class_prior is not None else None
        )
        if ent_w > 0:
            L_ent = im_loss(p_t, prior)
        if pl_w > 0:
            L_pl = pseudo_label_loss(head_logits_t, p_t.detach(), cfg.pl_threshold)
        if align_w > 0:
            Fe_t, Ce_t, he_t = build_ego_batch_from_cache(
                tgt_cache, tgt_emb, seeds_t,
                anchor_weight=cfg.anchor_weight,
                anchor_mass_extra=cfg.anchor_mass_extra,
            )
            d_bcm_t = pairwise_fgw_distances(
                Fe_t, Ce_t, he_t, Fp, Cp, q,
                alpha=cfg.fgw_alpha, epsilon=cfg.fgw_epsilon,
                max_iter=cfg.fgw_max_iter,
            )
            d_bc_t = fgw_class_distances(d_bcm_t, cfg.tau)
            # DEC assignment driven by the *head*'s prediction: pull each
            # target ego toward the prototype of its predicted class.
            L_align = align_loss(d_bc_t, p_t)

    # ---------------------------------------------------- prototype regularisers
    L_sep = L_struct = zero
    if proto_w > 0:
        L_sep = separation_loss(
            Fp, Cp, q,
            alpha=cfg.fgw_alpha, epsilon=cfg.fgw_epsilon, max_iter=cfg.fgw_max_iter,
            margin=cfg.sep_margin, pairwise_fn=pairwise_fgw_distances,
            intra_margin=cfg.sep_intra_margin,
        )
        L_struct = struct_l1_loss(model.prototypes.soft_adjacency())

    loss = (
        L_cls
        + proto_w * L_proto
        + align_w * L_align
        + ent_w * L_ent
        + cfg.lambda_sep * L_sep
        + cfg.lambda_vrex * L_vrex
        + cfg.lambda_struct * L_struct
        + pl_w * L_pl
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "L_cls": L_cls.item(),
        "L_proto": L_proto.item(),
        "L_align": L_align.item(),
        "L_ent": L_ent.item(),
        "L_sep": L_sep.item(),
        "L_vrex": L_vrex.item(),
        "L_struct": L_struct.item(),
        "L_pl": L_pl.item(),
        "phase": phase,
        "align_w": align_w,
        "ent_w": ent_w,
        "src_f1_mean": sum(m["f1"] for m in per_src_metrics) / len(per_src_metrics),
    }


# --------------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(model: FGWPrototypeDA, data: Data) -> dict:
    """Predictions come from the parametric head, so evaluation no longer
    needs ego-graphs or FGW solves (one cheap encode + MLP pass)."""
    model.eval()
    emb = model.encode(data.x, data.edge_index)
    logits = model.classify(emb)
    metrics = compute_metrics(logits, data.y)
    metrics["loss"] = Fnn.cross_entropy(logits, data.y).item()
    return metrics


# ------------------------------------------------------------------ orchestrator
def run_training(
    model: FGWPrototypeDA,
    sources: List[Data],
    target: Data,
    cfg: FGWConfig,
) -> FGWPrototypeDA:
    device = cfg.device
    sources = [s.to(device) for s in sources]
    target = target.to(device)
    model = model.to(device)

    if cfg.no_da:
        # Pure source-supervised baseline: the head never touches the FGW
        # machinery, so there is no need to precompute any ego-graphs.
        src_caches = [None for _ in sources]
        tgt_cache = None
    else:
        print("Precomputing PPR ego-graphs ...")
        src_caches = [_make_cache(s, cfg, device) for s in sources]
        tgt_cache = _make_cache(target, cfg, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        stats = train_step(
            model, sources, target, src_caches, tgt_cache,
            optimizer, cfg, epoch,
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            tgt_stats = evaluate(model, target)
            print(
                f"Epoch {epoch:3d} [{stats['phase']:>7}] "
                f"loss {stats['loss']:.4f}  "
                f"cls {stats['L_cls']:.4f}  "
                f"pr {stats['L_proto']:.4f}  "
                f"al(w={stats['align_w']:.2f}) {stats['L_align']:.4f}  "
                f"ent {stats['L_ent']:.4f}  "
                f"sep {stats['L_sep']:.4f}  "
                f"vx {stats['L_vrex']:.4f}  "
                f"pl {stats['L_pl']:.4f} | "
                f"src_f1 {stats['src_f1_mean']:.4f}  "
                f"tgt_f1 {tgt_stats['f1']:.4f}  tgt_auc {tgt_stats['auc']:.4f}"
            )
            if tgt_stats["f1"] > best_f1:
                best_f1 = tgt_stats["f1"]
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model
