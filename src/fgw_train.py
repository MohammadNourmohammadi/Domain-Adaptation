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

Model selection uses a held-out slice of the source label budget
(`cfg.source_val_frac`) — never target labels. The end-of-run table also
reports what the label-free criteria (SND, prediction entropy) would have
picked and how far each lands from the target-oracle epoch.
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
def _make_cache(
    g: Data, cfg: FGWConfig, device: str, seed_nodes: torch.Tensor = None,
) -> EgoGraphCache:
    cache = EgoGraphCache(
        edge_index=g.edge_index,
        num_nodes=g.num_nodes,
        ppr_alpha=cfg.ppr_alpha,
        ppr_iters=cfg.ppr_iters,
        ego_size=cfg.ego_size,
    )
    # `seed_nodes=None` precomputes every node (the target). Source graphs pass
    # their labeled subset: those are the only nodes ever seeded into an FGW
    # problem, so precomputing the rest would waste the (expensive) PPR +
    # shortest-path work on ego-graphs that are never read.
    cache.precompute_all(batch_size=512, device=device, seed_nodes=seed_nodes)
    return cache


def _proto_tensors(model: FGWPrototypeDA, device):
    Fp = model.prototypes.features()             # (C, M, n_p, d+1)
    Cp = model.prototypes.structure_matrices()   # (C, M, n_p, n_p)
    q = model.prototypes.uniform_mass(device)
    return Fp, Cp, q


# --------------------------------------------------------- supervision budget
def _labeled_indices(g: Data, ratio: float, stratified: bool) -> torch.Tensor:
    """Node indices whose labels are revealed to the supervised source loss.

    Implements the "X% of nodes labeled" supervision budget: only the returned
    ``ratio`` fraction of source nodes ever contribute to the CE loss — the other
    ``1 - ratio`` still flow through message passing but their labels are hidden.
    The subset is drawn once (a random split, re-drawn when the global seed
    changes), so V̄ᵢ is fixed across epochs as the paper's setting requires.

    With ``stratified`` the draw is per-class (each class contributes ~``ratio``
    of its nodes, min 1) — safer for imbalanced label sets; otherwise it is a
    pure random draw over all nodes. ``ratio >= 1`` returns every node.
    """
    num_nodes = g.num_nodes
    device = g.y.device
    if ratio >= 1.0:
        return torch.arange(num_nodes, device=device)
    if stratified:
        parts = []
        for c in torch.unique(g.y):
            cls_idx = (g.y == c).nonzero(as_tuple=False).view(-1)
            k = max(1, int(round(ratio * cls_idx.numel())))
            perm = torch.randperm(cls_idx.numel(), device=device)[:k]
            parts.append(cls_idx[perm])
        return torch.cat(parts)
    k = max(1, int(round(ratio * num_nodes)))
    return torch.randperm(num_nodes, device=device)[:k]


def _labeled_split(
    g: Data, ratio: float, stratified: bool, val_frac: float,
) -> tuple:
    """Split the revealed-label budget into a train part and a *held-out* part.

    The CE loss only ever sees the train part; the val part is scored but never
    backpropped, which is what makes it a legitimate model-selection signal.
    Without it the only source number available is macro-F1 on the exact seeds
    just optimised, which measures memorisation rather than generalisation.

    The draw mirrors ``_labeled_indices``: per-class when ``stratified`` (so the
    val split keeps the class mix), a pure random split otherwise. At least one
    node always stays in train, so tiny budgets degrade gracefully to an empty
    val split rather than an empty train split.
    """
    idx = _labeled_indices(g, ratio, stratified)
    empty = idx.new_empty((0,))
    if val_frac <= 0.0 or idx.numel() < 2:
        return idx, empty

    def _cut(pool: torch.Tensor) -> tuple:
        perm = pool[torch.randperm(pool.numel(), device=pool.device)]
        n_val = min(max(1, int(round(val_frac * perm.numel()))), perm.numel() - 1)
        return perm[n_val:], perm[:n_val]

    if stratified:
        train_parts, val_parts = [], []
        y_idx = g.y[idx]
        for c in torch.unique(y_idx):
            cls_pool = idx[y_idx == c]
            if cls_pool.numel() < 2:
                train_parts.append(cls_pool)
                continue
            tr, va = _cut(cls_pool)
            train_parts.append(tr)
            val_parts.append(va)
        val = torch.cat(val_parts) if val_parts else empty
        return torch.cat(train_parts), val
    return _cut(idx)


# ----------------------------------------------------------------- train step
def train_step(
    model: FGWPrototypeDA,
    sources: List[Data],
    target: Data,
    src_caches: List[EgoGraphCache],
    src_labeled: List[torch.Tensor],
    tgt_cache: EgoGraphCache,
    optimizer: torch.optim.Optimizer,
    cfg: FGWConfig,
    epoch: int,
) -> dict:
    model.train()
    device = target.x.device
    da = not cfg.no_da
    phase = _phase(epoch, cfg.epochs, cfg.warmup_frac, cfg.refine_frac) if da else "srconly"
    # The ramp must be measured from the *end of warmup*, not from epoch 0:
    # alignment is off during warmup, so counting absolute epochs would leave the
    # ramp already saturated (~1.0) on the first adapt epoch and slam the weight
    # from 0 to full in a single step.
    warm_epochs = int(cfg.warmup_frac * cfg.epochs)
    ramp = _ramp(max(epoch - warm_epochs, 0), cfg.ramp_epochs)
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
    for s, emb, cache, labeled in zip(sources, src_emb, src_caches, src_labeled):
        # Only the labeled subset (the supervision budget) may seed the CE loss;
        # the other source nodes are hidden from supervision.
        n = min(cfg.nodes_per_step, labeled.numel())
        seeds = labeled[torch.randperm(labeled.numel(), device=device)[:n]]
        y = s.y[seeds]
        head_logits = model.classify(emb[seeds])
        per_src_losses.append(cls_loss(head_logits, y))
        # NOTE: a *training* metric — the same nodes this step backprops through.
        # It says nothing about generalisation and will climb towards 1.0 while
        # held-out performance falls; read `src_val_f1` in the log for that.
        per_src_metrics.append(compute_metrics(head_logits, y))
        if proto_w > 0:
            Fe, Ce, he = build_ego_batch_from_cache(
                cache, emb, seeds,
                anchor_weight=cfg.anchor_weight,
                anchor_mass_extra=cfg.anchor_mass_extra,
            )
            if epoch == 0 and not per_src_proto:
                # Scale sanity check: the two structure matrices must sit on a
                # comparable scale or the FGW geometry is broken. Means should
                # be within ~2x of each other.
                print(
                    f"[fgw-scale] C_ego mean={Ce.mean().item():.3f} "
                    f"std={Ce.std().item():.3f} | "
                    f"C_p mean={Cp.mean().item():.3f} std={Cp.std().item():.3f}"
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

        if ent_w > 0:
            L_ent = im_loss(p_t)
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
            # target ego toward the prototype of its predicted class and
            # away from the others. Contrastive (softmax over classes), so
            # uniformly shrinking every distance does not reduce the loss.
            L_align = align_loss(d_bc_t, p_t, cfg.tau)

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
        # Training-set macro-F1 (see the note where it is computed).
        "src_train_f1": sum(m["f1"] for m in per_src_metrics) / len(per_src_metrics),
    }


# --------------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(model: FGWPrototypeDA, data: Data, idx: torch.Tensor = None) -> dict:
    """Predictions come from the parametric head, so evaluation no longer
    needs ego-graphs or FGW solves (one cheap encode + MLP pass).

    ``idx`` restricts scoring to a node subset (used for the held-out source
    validation split); the encode still runs on the full graph, so the scored
    nodes keep their real neighbourhoods.
    """
    model.eval()
    emb = model.encode(data.x, data.edge_index)
    logits = model.classify(emb)
    y = data.y
    if idx is not None:
        logits, y = logits[idx], y[idx]
    metrics = compute_metrics(logits, y)
    metrics["loss"] = Fnn.cross_entropy(logits, y).item()
    return metrics


@torch.no_grad()
def _source_val_metrics(
    model: FGWPrototypeDA, sources: List[Data], src_val: List[torch.Tensor],
) -> dict:
    """Macro-F1 pooled over every source's held-out split.

    Pooling the logits (rather than averaging per-source F1) keeps the metric
    stable when one source contributes only a handful of validation nodes.
    """
    model.eval()
    logits, labels = [], []
    for s, val in zip(sources, src_val):
        if val.numel() == 0:
            continue
        emb = model.encode(s.x, s.edge_index)
        logits.append(model.classify(emb)[val])
        labels.append(s.y[val])
    if not logits:
        return None
    return compute_metrics(torch.cat(logits), torch.cat(labels))


# --------------------------------------------------- unsupervised criteria
@torch.no_grad()
def _target_criteria(
    model: FGWPrototypeDA, target: Data, sub: torch.Tensor, temp: float,
) -> dict:
    """Label-free selection scores on the target graph (higher = better).

    ``neg_ent``  – negative mean prediction entropy: confident predictions.
    ``snd``      – Soft Neighborhood Density (Saito et al., ICCV 2021): entropy
                   of the temperature-softmaxed cosine-similarity distribution
                   between L2-normalised target predictions. A representation
                   whose target points sit in dense, consistent neighbourhoods
                   scores high; one collapsed onto a few points scores low.

    Neither sees a target label, so selecting on them is legitimate — unlike
    selecting on target macro-F1, which is oracle selection.
    """
    model.eval()
    emb = model.encode(target.x, target.edge_index)
    p = Fnn.softmax(model.classify(emb), dim=1)

    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()

    # O(N^2) similarity matrix -> score a fixed subsample so the number is both
    # affordable and comparable across epochs.
    z = Fnn.normalize(p[sub], dim=1)
    sim = z @ z.t()
    sim.fill_diagonal_(-float("inf"))          # a point is not its own neighbour
    nb = Fnn.softmax(sim / temp, dim=1)
    snd = -(nb * torch.log(nb.clamp_min(1e-8))).sum(dim=1).mean()
    return {"neg_ent": -ent.item(), "snd": snd.item()}


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

    # Supervision budget: fix a labeled subset per source graph for the whole run
    # (the target is never labeled during training). Uses the already-seeded
    # global RNG, so the split is deterministic per seed and re-drawn each run.
    ratio = getattr(cfg, "source_label_ratio", 1.0)
    stratified = getattr(cfg, "source_label_stratified", False)
    val_frac = getattr(cfg, "source_val_frac", 0.0)
    # The budget is split train/val: only `src_labeled` reaches the CE loss, so
    # `src_val` stays a clean generalisation signal for checkpoint selection.
    splits = [_labeled_split(s, ratio, stratified, val_frac) for s in sources]
    src_labeled = [tr for tr, _ in splits]
    src_val = [va for _, va in splits]
    n_val_total = sum(va.numel() for va in src_val)
    if ratio < 1.0 or n_val_total:
        labeled_desc = ", ".join(
            f"{name} {tr.numel()}+{va.numel()}v/{s.num_nodes}"
            for name, s, tr, va in zip(cfg.source_domains, sources, src_labeled, src_val)
        )
        draw = "stratified" if stratified else "random"
        print(f"Supervision budget: {ratio:.0%} source labels ({draw} draw, "
              f"{val_frac:.0%} held out for selection) -> "
              f"train+val nodes [{labeled_desc}]")

    if cfg.no_da:
        # Pure source-supervised baseline: the head never touches the FGW
        # machinery, so there is no need to precompute any ego-graphs.
        src_caches = [None for _ in sources]
        tgt_cache = None
    else:
        # Sources: only the labeled subset is ever seeded (see train_step), so
        # restrict the precompute to those nodes. Target: every node may be
        # sampled for alignment/IM, so precompute all of them. When a source is
        # fully labeled (ratio >= 1) fall back to the all-nodes path to skip
        # building an identity row map.
        n_src_seeds = sum(lab.numel() for lab in src_labeled)
        print(
            f"Precomputing PPR ego-graphs (sources: {n_src_seeds} labeled seeds, "
            f"target: {target.num_nodes} nodes) ..."
        )
        src_caches = [
            _make_cache(
                s, cfg, device,
                seed_nodes=None if lab.numel() >= s.num_nodes else lab,
            )
            for s, lab in zip(sources, src_labeled)
        ]
        tgt_cache = _make_cache(target, cfg, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # Fixed target subsample for SND, so the score is comparable across epochs.
    snd_sub = torch.randperm(target.num_nodes, device=device)[
        : min(getattr(cfg, "snd_max_nodes", 2000), target.num_nodes)
    ]
    snd_temp = getattr(cfg, "snd_temp", 0.05)

    criterion = getattr(cfg, "model_selection", "last")
    if criterion == "src_val" and n_val_total == 0:
        print("No source validation nodes available -> model_selection falls "
              "back to 'last'.")
        criterion = "last"
    # Every candidate criterion is tracked at each eval epoch, not just the one
    # doing the selecting: the end-of-run table then shows what each *would*
    # have picked and what that costs against the oracle.
    history = []

    best_state = None
    best_score = -float("inf")
    best_epoch = None

    for epoch in range(1, cfg.epochs + 1):
        stats = train_step(
            model, sources, target, src_caches, src_labeled, tgt_cache,
            optimizer, cfg, epoch,
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            tgt_stats = evaluate(model, target)
            val_stats = _source_val_metrics(model, sources, src_val)
            unsup = _target_criteria(model, target, snd_sub, snd_temp)
            record = {
                "epoch": epoch,
                "src_val": val_stats["f1"] if val_stats else float("nan"),
                "snd": unsup["snd"],
                "neg_ent": unsup["neg_ent"],
                "tgt_f1": tgt_stats["f1"],
                "tgt_auc": tgt_stats["auc"],
            }
            history.append(record)

            if criterion in ("src_val", "snd", "neg_ent", "entropy"):
                key = "neg_ent" if criterion == "entropy" else criterion
                score = record[key]
                if score == score and score > best_score:   # NaN-safe
                    best_score, best_epoch = score, epoch
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }

            val_txt = (
                f"src_val_f1 {record['src_val']:.4f}" if val_stats
                else "src_val_f1   n/a "
            )
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
                f"src_train_f1 {stats['src_train_f1']:.4f}  {val_txt} | "
                f"snd {record['snd']:.4f}  "
                f"tgt ACC {tgt_stats['acc']:.4f}  "
                f"AUROC {tgt_stats['auc']:.4f}  "
                f"MacroF {tgt_stats['f1']:.4f}"
            )

    _report_selection(history, criterion, best_epoch)

    if best_state is not None:
        print(f"Restoring checkpoint from epoch {best_epoch} "
              f"(selected by '{criterion}').")
        model.load_state_dict(best_state)
    return model


def _report_selection(history: List[dict], criterion: str, best_epoch) -> None:
    """Print what each criterion picks and what it costs against the oracle.

    The oracle row selects on target macro-F1 and is *not* a usable criterion —
    it is the upper bound that says how much a label-free (or source-val)
    criterion gives up. Reporting that gap honestly is the point.
    """
    if not history:
        return
    criterion = "neg_ent" if criterion == "entropy" else criterion
    oracle = max(history, key=lambda r: r["tgt_f1"])
    print("\n" + "-" * 60)
    print("  Model selection (target labels never used to select)")
    print("-" * 60)
    print(f"  {'criterion':<12}{'epoch':>7}{'score':>10}{'tgt MacroF':>13}"
          f"{'vs oracle':>11}")
    for name, key in (("src_val_f1", "src_val"), ("SND", "snd"),
                      ("neg_entropy", "neg_ent")):
        usable = [r for r in history if r[key] == r[key]]
        if not usable:
            print(f"  {name:<12}{'n/a':>7}")
            continue
        pick = max(usable, key=lambda r: r[key])
        mark = " *" if criterion == key else "  "
        print(f"  {name:<12}{pick['epoch']:>7}{pick[key]:>10.4f}"
              f"{pick['tgt_f1']:>13.4f}{pick['tgt_f1'] - oracle['tgt_f1']:>+11.4f}"
              f"{mark}")
    last = history[-1]
    print(f"  {'last epoch':<12}{last['epoch']:>7}{'-':>10}{last['tgt_f1']:>13.4f}"
          f"{last['tgt_f1'] - oracle['tgt_f1']:>+11.4f}"
          f"{' *' if criterion == 'last' else '  '}")
    print(f"  {'oracle':<12}{oracle['epoch']:>7}{'-':>10}{oracle['tgt_f1']:>13.4f}"
          f"{0.0:>+11.4f}   (upper bound, not selectable)")
    if best_epoch is not None:
        print(f"  -> reporting epoch {best_epoch} chosen by '{criterion}'")
    print("-" * 60)
