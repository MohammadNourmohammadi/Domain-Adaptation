"""Training loop for the FGW prototype-graph DA method.

Prediction is parametric: a 2-layer MLP head on the encoder produces the
class logits used for the supervised source loss, evaluation, IM and
pseudo-labels. The FGW prototype machinery is used for *transfer* only
(target alignment + an auxiliary source term that keeps the prototypes
class-meaningful). Decoupling the prediction head from the FGW distances
is what prevents the alignment objective from collapsing the classifier
to the uniform ln-2 fixed point.

Three phases, driven by `PhaseScheduler` (absolute epoch counts plus a
validation-plateau trigger — *not* fractions of `cfg.epochs`):

  1. WARM-UP    – L_cls + L_proto + L_margin (+ L_sep + L_struct). The
                  head and the prototypes must become meaningful before
                  any target signal is introduced; aligning to random
                  prototypes just injects noise. Ends on an EMA-smoothed
                  held-out-source-F1 plateau — but only once L_proto has
                  actually dropped below chance-level, because that
                  precondition used to be stated and never checked. The
                  best warm-up checkpoint is restored before adapting.
  2. ADAPT      – ramp lambda_align and lambda_ent from 0 to their full
                  values with the same sigmoid schedule used for GRL.
                  Lasts `cfg.adapt_epochs`.
  3. REFINE     – additionally enable confidence-thresholded pseudo-label
                  cross-entropy on the target.

With `cfg.no_da` the prototypes and every target term are switched off,
leaving a pure encoder+head source-supervised baseline (the diagnostic
for reading the achievable in-domain ceiling).

Mini-batching is done over nodes, not graphs: every step encodes each
full graph once (cheap) and then samples `cfg.nodes_per_step` seeds
from each source / target to form FGW problems.

Two label-free estimators steer the transfer, both in `fgw_select`:

  * **Label shift.** The target prior is estimated by BBSE from the
    held-out source split at the moment warm-up ends, then frozen. It
    feeds the IM balance term (without which entropy minimisation is
    optimised by predicting one class everywhere) and the inference-time
    prior correction.
  * **Source selection.** Per-domain and per-node transferability read
    straight off the ego-graph FGW distances, replacing SelMAG's three
    pretext tasks plus meta-learned selector. Refreshed every
    `cfg.select_every` epochs once adaptation has begun.

Model selection uses a held-out slice of the source label budget
(`cfg.source_val_frac`) — never target labels. The end-of-run table also
reports what the label-free criteria (SND, prediction entropy) would have
picked and how far each lands from the target-oracle epoch.

`run_training` returns `(model, ctx)`; `ctx` carries the priors and the
ego caches so the caller can score the final model exactly as the loop did.
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch_geometric.data import Data

from .fgw_classifier import fgw_class_distances, fgw_logits
from .fgw_config import FGWConfig
from .fgw_distance import pairwise_fgw_distances
from .fgw_ego import EgoGraphCache, build_ego_batch_from_cache
from .fgw_encoder import svd_projection
from .fgw_losses import (
    align_loss,
    balance_scale,
    cls_loss,
    fgw_margin_loss,
    im_loss,
    pseudo_label_loss,
    separation_loss,
    struct_density_loss,
    vrex_loss,
)
from .fgw_model import FGWPrototypeDA
from .fgw_select import (
    class_prior,
    estimate_target_prior,
    mean_ego_structure,
    prior_corrected_logits,
    transferability,
)
from .graph_augment import augment_knn
from .utils import compute_metrics


# ---------------------------------------------------------------------- schedule
_RAMP_K = 5.0


def _ramp(epoch: int, ramp_epochs: int) -> float:
    """DANN-style sigmoid ramp, normalised to reach exactly 1.0 at the end.

    The textbook `2/(1+exp(-10p)) - 1` is already at 0.987 by p = 0.5, so a
    nominal 20-epoch ramp was really a 10-epoch one (the logs show w = 0.99 ten
    epochs after the switch). Using a gentler exponent and dividing by its
    endpoint keeps the sigmoid shape while making the configured length the
    length you actually get.
    """
    p = min(epoch / max(ramp_epochs, 1), 1.0)
    end = 2.0 / (1.0 + math.exp(-_RAMP_K)) - 1.0
    return (2.0 / (1.0 + math.exp(-_RAMP_K * p)) - 1.0) / end


class PhaseScheduler:
    """warmup -> adapt -> refine on absolute epoch counts + a val plateau.

    Two things this fixes over the old fractional schedule:

    * **Absolute lengths.** The encoder overfits the (small) labeled seed set
      on a fixed timescale of a few dozen epochs; that has nothing to do with
      how many epochs the run was given. Tying warm-up to a fraction of
      `cfg.epochs` meant a longer budget started adapting *later*, i.e. from a
      more thoroughly overfit model.
    * **Rewind on switch.** Warm-up stops at the first plateau of held-out
      source macro-F1 and the best warm-up weights are restored before the
      target terms come on. Adapting from the post-plateau model would start
      the transfer stage from an encoder that has already memorised its seeds.

    The plateau trigger needs a held-out source split; with none available the
    scheduler degrades to the fixed `warmup_epochs` cap (and no rewind, since
    there is no honest signal to pick a checkpoint by).
    """

    def __init__(self, cfg: FGWConfig, plateau: bool, proto_chance: float = 0.0):
        self.warmup_epochs = max(int(getattr(cfg, "warmup_epochs", 150)), 1)
        self.adapt_epochs = max(int(getattr(cfg, "adapt_epochs", 120)), 1)
        self.patience = max(int(getattr(cfg, "warmup_patience", 40)), 1)
        self.min_epochs = max(int(getattr(cfg, "warmup_min_epochs", 40)), 0)
        self.min_delta = float(getattr(cfg, "warmup_min_delta", 1e-4))
        self.ema_m = float(getattr(cfg, "warmup_val_ema", 0.7))
        # L_proto has to fall below this before the target terms are allowed
        # on; `proto_chance` is ln(num_classes), the value of a coin flip.
        gate = float(getattr(cfg, "warmup_proto_gate", 0.9))
        self.proto_ceiling = gate * proto_chance if gate > 0 else None
        self.plateau = plateau

        self.phase = "warmup"
        self.adapt_start = None      # first epoch on which align/ent are live
        self.best_val = -float("inf")
        self.best_epoch = None
        self.stale = 0
        self._best_state = None
        self._ema = None
        self.proto_ok = self.proto_ceiling is None
        self.last_proto = float("nan")

    def ramp_position(self, epoch: int) -> int:
        """Epochs elapsed since adaptation began (0 on the first adapt epoch).

        The ramp must be measured from the end of warm-up, not from epoch 0:
        alignment is off during warm-up, so absolute epochs would leave the
        ramp already saturated on the first adapt epoch and slam the weight
        from 0 to full in a single step.
        """
        return 0 if self.adapt_start is None else max(epoch - self.adapt_start, 0)

    def needs_val(self) -> bool:
        """Whether this epoch's plateau test requires a validation pass."""
        return self.plateau and self.phase == "warmup"

    def step(
        self, epoch: int, model, val_f1: float = None, proto_loss: float = None,
    ) -> str:
        """Advance the phase after `epoch` finished; returns a log note or None."""
        if self.phase == "refine":
            return None

        if self.phase == "warmup":
            if proto_loss is not None and proto_loss == proto_loss:
                self.last_proto = proto_loss
                if self.proto_ceiling is not None and proto_loss <= self.proto_ceiling:
                    self.proto_ok = True

            if self.plateau and val_f1 is not None and val_f1 == val_f1:
                # Track the best on the *raw* score (that is the checkpoint we
                # want to rewind to) but test the plateau on an EMA, so a
                # single noisy dip cannot burn the patience budget.
                if val_f1 > self.best_val + self.min_delta:
                    self.best_val, self.best_epoch = val_f1, epoch
                    self._best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                self._ema = (val_f1 if self._ema is None
                             else self.ema_m * self._ema + (1 - self.ema_m) * val_f1)
                if not hasattr(self, "_best_ema") or self._ema > self._best_ema + self.min_delta:
                    self._best_ema, self.stale = self._ema, 0
                else:
                    self.stale += 1

            armed = epoch >= self.min_epochs
            plateaued = self.plateau and armed and self.stale >= self.patience
            capped = epoch >= self.warmup_epochs
            if not (capped or (plateaued and self.proto_ok)):
                return None

            self.phase = "adapt"
            self.adapt_start = epoch + 1
            if plateaued and self.proto_ok:
                why = f"val plateau ({self.stale} epochs without an EMA gain)"
            else:
                why = f"warmup_epochs cap ({self.warmup_epochs})"
            note = f"[phase] warmup -> adapt at epoch {epoch + 1}: {why}"
            if self._best_state is not None:
                model.load_state_dict(self._best_state)
                note += (f"; rewound to the warm-up best (epoch "
                         f"{self.best_epoch}, src_val {self.best_val:.4f})")
                self._best_state = None      # the rewind is one-shot; free it
            if self.proto_ceiling is not None and not self.proto_ok:
                note += (f"\n  ! L_proto is {self.last_proto:.4f}, above the "
                         f"{self.proto_ceiling:.4f} gate: the FGW distances are "
                         f"still near chance, so alignment has no class-"
                         f"meaningful prototypes to pull towards. Raise "
                         f"--lambda_fgw_margin / --warmup_epochs, or lower "
                         f"--proto_size.")
            return note

        if epoch - self.adapt_start >= self.adapt_epochs - 1:
            self.phase = "refine"
            return (f"[phase] adapt -> refine at epoch {epoch + 1} "
                    f"(after {self.adapt_epochs} adapt epochs)")
        return None


# ---------------------------------------------------------------------- helpers
def _fmt_prior(p: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in p.tolist()) + "]"


def _make_optimizer(model: FGWPrototypeDA, cfg: FGWConfig) -> torch.optim.Optimizer:
    """Adam with the prototype bank in its own, decay-free parameter group.

    Weight decay on the bank is actively harmful, not merely useless:

    * `PrototypeBank.embeddings()` LayerNorms `Z`, so the forward pass does not
      depend on |Z| at all. Decay shrinks the parameter without changing any
      output, and since the gradient w.r.t. the normalised vector scales as
      1/|Z|, the effective angular step size grows without bound over a run.
    * On `E_logits` it destroys the thing the method reads. Decay pulls the
      logits to 0, so `sigmoid(0) = 0.5`, so `C_p` becomes a constant 0.5
      matrix — no structural information for the FGW term to use.
    """
    proto_params = list(model.prototypes.parameters())
    proto_ids = {id(p) for p in proto_params}
    rest = [p for p in model.parameters() if id(p) not in proto_ids]
    return torch.optim.Adam(
        [
            {"params": rest, "weight_decay": cfg.weight_decay},
            {"params": proto_params,
             "weight_decay": getattr(cfg, "proto_weight_decay", 0.0)},
        ],
        lr=cfg.lr,
    )


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
    phase: str,
    ramp_pos: int = 0,
    s_global: torch.Tensor = None,
    s_local: List[torch.Tensor] = None,
    target_prior: torch.Tensor = None,
    struct_density: float = 0.5,
    scale_probe: dict = None,
    class_weights: torch.Tensor = None,
) -> dict:
    """One optimisation step. `phase` and `ramp_pos` come from the
    `PhaseScheduler` (which owns the warm-up plateau test), so the schedule is
    not re-derived from `epoch` here.

    `s_global` / `s_local` are the FGW transferability weights (see
    `fgw_select.transferability`); both are None when selection is off, which
    reproduces the uniform-average behaviour exactly.
    """
    model.train()
    device = target.x.device
    da = not cfg.no_da
    ramp = _ramp(ramp_pos, cfg.ramp_epochs)
    align_w = cfg.lambda_align * ramp if (da and phase != "warmup") else 0.0
    ent_w = cfg.lambda_ent * ramp if (da and phase != "warmup") else 0.0
    pl_w = cfg.lambda_pl if (da and phase == "refine") else 0.0
    proto_w = cfg.lambda_proto if da else 0.0
    margin_w = cfg.lambda_fgw_margin if da else 0.0

    src_emb = [model.encode(s.x, s.edge_index) for s in sources]

    Fp = Cp = q = None
    if da:
        Fp, Cp, q = _proto_tensors(model, device)

    # ------------------------------------------------------ supervised sources
    # Prediction/CE go through the parametric head; the FGW distances feed an
    # auxiliary term (L_proto) that keeps the prototypes class-meaningful.
    per_src_losses = []          # head CE per source (drives L_cls and V-REx)
    per_src_proto = []           # FGW-prototype CE per source (anchoring)
    per_src_margin = []          # FGW hinge per source
    per_src_metrics = []
    for j, (s, emb, cache, labeled) in enumerate(
        zip(sources, src_emb, src_caches, src_labeled)
    ):
        # Only the labeled subset (the supervision budget) may seed the CE loss;
        # the other source nodes are hidden from supervision.
        n = min(cfg.nodes_per_step, labeled.numel())
        pick = torch.randperm(labeled.numel(), device=device)[:n]
        seeds = labeled[pick]
        y = s.y[seeds]
        # s_local is stored per *labeled node*, in the same order as
        # `src_labeled[j]`, so the sampling permutation indexes it directly.
        w_node = None if s_local is None else s_local[j][pick]
        # Class weights ride the same per-node channel, so they reach the head
        # CE, L_proto and the FGW hinge in one place. `cls_loss` renormalises to
        # mean 1, so this changes the *relative* weighting only.
        if class_weights is not None:
            cw = class_weights[y]
            w_node = cw if w_node is None else w_node * cw
        head_logits = model.classify(emb[seeds])
        per_src_losses.append(cls_loss(head_logits, y, w_node))
        # NOTE: a *training* metric — the same nodes this step backprops through.
        # It says nothing about generalisation and will climb towards 1.0 while
        # held-out performance falls; read `src_val_f1` in the log for that.
        per_src_metrics.append(compute_metrics(head_logits, y))
        if proto_w > 0 or margin_w > 0:
            Fe, Ce, he = build_ego_batch_from_cache(
                cache, emb, seeds,
                anchor_weight=cfg.anchor_weight,
                anchor_mass_extra=cfg.anchor_mass_extra,
            )
            if scale_probe is not None and not scale_probe:
                # Scale sanity check: the two structure matrices must sit on a
                # comparable scale or the FGW geometry is broken. Means should
                # be within ~2x of each other. (This used to be gated on
                # `epoch == 0` while the loop starts at 1, so it never once
                # printed — the single most useful diagnostic in the pipeline
                # was dead code.)
                scale_probe["done"] = True
                # Average over the *reached* pairs only. Unreached slots are
                # filled with 1 and carry no mass, so including them reports a
                # scale mismatch that does not exist: on Yelp the raw mean is
                # 0.81 against C_p's 0.45, while the pairs the solver actually
                # sees average 0.47. `he > 0` is exactly the support mask.
                v = (he > 0).to(Ce.dtype)
                pw = v.unsqueeze(-1) * v.unsqueeze(-2)
                pw = pw * (1.0 - torch.eye(pw.size(-1), device=pw.device))
                c_mean = (Ce * pw).sum() / pw.sum().clamp_min(1.0)
                c_var = ((Ce - c_mean) ** 2 * pw).sum() / pw.sum().clamp_min(1.0)
                print(
                    f"[fgw-scale] C_ego mean={c_mean.item():.3f} "
                    f"std={c_var.sqrt().item():.3f} (over reached pairs) | "
                    f"C_p mean={Cp.mean().item():.3f} std={Cp.std().item():.3f} "
                    f"(target density {struct_density:.3f})"
                )
            d_bcm = pairwise_fgw_distances(
                Fe, Ce, he, Fp, Cp, q,
                alpha=cfg.fgw_alpha, epsilon=cfg.fgw_epsilon,
                max_iter=cfg.fgw_max_iter,
            )
            per_src_proto.append(cls_loss(fgw_logits(d_bcm, cfg.tau), y, w_node))
            if margin_w > 0:
                per_src_margin.append(
                    fgw_margin_loss(
                        fgw_class_distances(d_bcm, cfg.tau), y,
                        cfg.fgw_margin, w_node,
                    )
                )

    # Per-domain transferability. `s_global` is normalised to mean 1, so with
    # selection off this is exactly the old uniform average.
    stacked_cls = torch.stack(per_src_losses)
    if s_global is None:
        L_cls = stacked_cls.mean()
    else:
        L_cls = (stacked_cls * s_global).mean()
    # Always computed, weighted by `cfg.lambda_vrex` (0 by default). The
    # variance of the per-source risks turned out to be a much better
    # *instability detector* than a loss: it is ~0 on healthy epochs and only
    # spikes when one source's CE blows up, so it is kept in the log and out of
    # the objective.
    L_vrex = vrex_loss(stacked_cls)

    def _pool(terms):
        if not terms:
            return L_cls.new_zeros(())
        t = torch.stack(terms)
        return t.mean() if s_global is None else (t * s_global).mean()

    L_proto = _pool(per_src_proto)
    L_margin = _pool(per_src_margin)

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
            # The prior term is what stops entropy minimisation from walking
            # to the "one class everywhere" optimum; `target_prior` is the
            # BBSE estimate (or the pooled source prior as a fallback), never
            # uniform when a better number is available.
            L_ent = im_loss(p_t, target_prior, cfg.im_balance_weight)
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
    if da:
        # Cosine repulsion on the prototype embeddings: bounded, satisfiable at
        # 0, and no FGW solve (see `separation_loss`).
        L_sep = separation_loss(model.prototypes.embeddings())
        # Density-targeted, not L1-to-zero: see `struct_density_loss`.
        L_struct = struct_density_loss(
            model.prototypes.soft_adjacency(), struct_density,
        )

    loss = (
        L_cls
        + proto_w * L_proto
        + margin_w * L_margin
        + align_w * L_align
        + ent_w * L_ent
        + cfg.lambda_sep * L_sep
        + cfg.lambda_vrex * L_vrex
        + cfg.lambda_struct * L_struct
        + pl_w * L_pl
    )

    optimizer.zero_grad()
    loss.backward()
    # The FGW path is quadratic in the embeddings and passes through a
    # logsumexp, so a single unlucky step (two ego-graphs landing on top of a
    # prototype, say) can produce a gradient orders of magnitude above the
    # norm — which is how the runs used to detonate every couple of hundred
    # epochs and then spend dozens recovering. Clipping bounds the damage.
    clip = float(getattr(cfg, "grad_clip", 1.0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return {
        "loss": loss.item(),
        "L_cls": L_cls.item(),
        "L_proto": L_proto.item(),
        "L_margin": L_margin.item(),
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
def predict_logits(
    model: FGWPrototypeDA,
    data: Data,
    idx: torch.Tensor = None,
    cfg: FGWConfig = None,
    cache: EgoGraphCache = None,
) -> torch.Tensor:
    """Class logits for `idx` (all nodes when None).

    `cfg.predict` selects the head, the FGW prototype distances, or the mean
    of their log-probabilities. The FGW paths need an ego-graph and an FGW
    solve per scored node, so they are far more expensive than the head and
    only pay off once L_proto is well below ln C — hence "head" by default.
    """
    model.eval()
    emb = model.encode(data.x, data.edge_index)
    head = model.classify(emb)
    if idx is not None:
        head = head[idx]

    mode = getattr(cfg, "predict", "head") if cfg is not None else "head"
    if mode == "head" or cache is None:
        return head

    nodes = (torch.arange(data.num_nodes, device=emb.device) if idx is None
             else idx)
    Fp, Cp, q = _proto_tensors(model, emb.device)
    chunks = []
    step = max(int(getattr(cfg, "eval_batch_nodes", 512)), 1)
    for start in range(0, nodes.numel(), step):
        seeds = nodes[start:start + step]
        Fe, Ce, he = build_ego_batch_from_cache(
            cache, emb, seeds,
            anchor_weight=cfg.anchor_weight,
            anchor_mass_extra=cfg.anchor_mass_extra,
        )
        d = pairwise_fgw_distances(
            Fe, Ce, he, Fp, Cp, q,
            alpha=cfg.fgw_alpha, epsilon=cfg.fgw_epsilon,
            max_iter=cfg.fgw_max_iter,
        )
        chunks.append(fgw_logits(d, cfg.tau))
    fgw = torch.cat(chunks, dim=0)

    if mode == "fgw":
        return fgw
    # "ensemble": average in log-probability space so the two heads, which
    # have quite different logit scales, contribute comparably.
    return 0.5 * (Fnn.log_softmax(head, dim=1) + Fnn.log_softmax(fgw, dim=1))


@torch.no_grad()
def evaluate(
    model: FGWPrototypeDA,
    data: Data,
    idx: torch.Tensor = None,
    cfg: FGWConfig = None,
    cache: EgoGraphCache = None,
    source_prior: torch.Tensor = None,
    target_prior: torch.Tensor = None,
) -> dict:
    """Score `data` (optionally restricted to `idx`).

    The encode always runs on the full graph, so scored nodes keep their real
    neighbourhoods. When both priors are supplied the logits are re-based onto
    the target prior before argmax — see `fgw_select.prior_corrected_logits`.
    Passing them only for the target keeps source metrics untouched.
    """
    logits = predict_logits(model, data, idx, cfg, cache)
    y = data.y if idx is None else data.y[idx]
    raw_loss = Fnn.cross_entropy(logits, y).item()
    if source_prior is not None and target_prior is not None:
        logits = prior_corrected_logits(logits, source_prior, target_prior)
    metrics = compute_metrics(logits, y)
    metrics["loss"] = raw_loss
    return metrics


def _val_key(cfg: FGWConfig, num_classes: int, n_val: int) -> str:
    """Which held-out-source statistic drives selection and the plateau test.

    Macro-F1 is the reported metric, so it is the aligned choice — but it is a
    mean of `num_classes` per-class F1 scores and needs enough held-out nodes
    per class to be a signal rather than variance. Cora_full is where that
    fails: 10% of a 3,299-node graph, 20% held out, is 66 nodes per source and
    330 pooled, spread over 70 classes, so half the classes contribute an F1
    estimated from zero or one node. That noise picks the reported checkpoint
    *and* the warm-up rewind point. Accuracy is lower-variance and slightly
    misaligned under imbalance; below the threshold it is the better trade.
    """
    mode = getattr(cfg, "val_metric", "auto")
    if mode in ("f1", "acc"):
        return mode
    per_class = float(getattr(cfg, "val_f1_min_per_class", 10.0))
    return "f1" if n_val >= per_class * max(num_classes, 1) else "acc"


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

    # Optional feature-kNN densification, before anything reads the topology
    # (SVD basis, ego caches, encoder). Sources and target alike; no labels.
    k_aug = int(getattr(cfg, "knn_augment", 0) or 0)
    if k_aug > 0:
        print(f"Densifying with {k_aug} feature-kNN edges per node below degree "
              f"{cfg.knn_min_degree} (transductive, label-free):")
        for name, g in zip(cfg.source_domains, sources):
            augment_knn(g, k_aug, cfg.knn_min_degree, name=name)
        augment_knn(target, k_aug, cfg.knn_min_degree,
                    name=f"{cfg.target_domain} (target)")

    # Frozen input projection: fit the basis on the stacked feature matrix of
    # every graph before anything is trained. No labels are involved, so this is
    # not model selection or leakage — it is a data-dependent change of basis
    # that removes the single biggest source of overfitting (the learned
    # feature_dim x proj_dim layer).
    if getattr(model.encoder, "frozen_proj", False):
        in_dim, proj_dim = model.encoder.in_dim, model.encoder.proj_dim
        n_nodes = sum(s.num_nodes for s in sources) + target.num_nodes
        model.encoder.set_projection(
            svd_projection([s.x for s in sources] + [target.x], proj_dim)
        )
        print(f"Frozen SVD input projection: {in_dim} -> {proj_dim} "
              f"(fitted on {n_nodes} unlabeled nodes, "
              f"{in_dim * proj_dim:,} trainable params removed)")

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
        # The stratified draw takes at least one node per class, so on a
        # long-tailed label set the realised budget sits slightly above `ratio`.
        # Print what was actually revealed rather than what was asked for.
        revealed = sum(tr.numel() + va.numel() for tr, va in splits)
        total_nodes = sum(s.num_nodes for s in sources)
        eff = revealed / max(total_nodes, 1)
        eff_note = "" if abs(eff - ratio) < 5e-3 else f" -> {eff:.1%} realised"
        print(f"Supervision budget: {ratio:.0%} source labels ({draw} draw, "
              f"{val_frac:.0%} held out for selection){eff_note} -> "
              f"train+val nodes [{labeled_desc}]")
        if stratified:
            covered = [
                int(torch.unique(s.y[tr]).numel())
                for s, tr in zip(sources, src_labeled)
            ]
            print(f"  classes covered by the training labels: "
                  f"{covered} of {cfg.num_classes}")

    # Everything the figure/report layer needs, filled in as the run proceeds
    # (see `src/plots.py`). Collecting it here rather than re-deriving it from
    # the log keeps the saved artefacts exact.
    support_stats: dict = {}
    events: List[dict] = []
    selection_trace: List[dict] = []

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
        # How much of each ego-graph is real. On an induced-subgraph split most
        # seeds sit in a component smaller than k, and the slots PPR never
        # reached now carry zero mass instead of a fixed set of strangers (see
        # `fgw_ego`). A low mean support is the signal that the structural half
        # of the method has little to work with — consider --knn_augment.
        for name, c in zip(list(cfg.source_domains) + [cfg.target_domain],
                           src_caches + [tgt_cache]):
            st = c.support_stats()
            if st:
                support_stats[name] = st
                print(f"  ego support {name}: mean {st['mean_support']:.1f}/"
                      f"{st['k']} nodes, {100 * st['frac_full']:.0f}% full, "
                      f"{100 * st['frac_singleton']:.0f}% isolated")

    optimizer = _make_optimizer(model, cfg)

    # Calibrate the prototype edge density so mean(C_p) lands on mean(C_ego).
    # Both matrices feed the same FGW structural term, so if they sit on
    # different scales that term is a class-independent offset and the
    # "graph" half of the method contributes nothing.
    struct_density = cfg.struct_density
    if struct_density is None:
        if cfg.no_da:
            struct_density = 0.5
        else:
            mean_c_ego = mean_ego_structure(src_caches + [tgt_cache])
            # C_p = 1 - A, so matching the means means density = 1 - mean(C_ego).
            struct_density = min(max(1.0 - mean_c_ego, 0.05), 0.95)
            print(f"Prototype density calibrated to {struct_density:.3f} "
                  f"(mean C_ego = {mean_c_ego:.3f})")

    # Class priors. The source prior is what the classifier is trained under;
    # the target one is estimated by BBSE once warm-up ends (see below) and is
    # never allowed to come from target labels.
    num_classes = cfg.num_classes
    src_prior = class_prior(
        torch.cat([s.y[tr] for s, tr in zip(sources, src_labeled)]), num_classes,
    ).to(device)
    if cfg.target_class_prior is not None:
        tgt_prior = torch.tensor(
            cfg.target_class_prior, dtype=torch.float, device=device,
        )
        tgt_prior = tgt_prior / tgt_prior.sum()
        prior_note = "given on the command line"
    else:
        tgt_prior = src_prior.clone()
        prior_note = ("source prior, BBSE estimate pending" if cfg.estimate_prior
                      else "source prior (--estimate_prior to run BBSE)")
    print(f"Class prior: source {_fmt_prior(src_prior)} | "
          f"target {_fmt_prior(tgt_prior)} [{prior_note}]; "
          f"IM balance x{balance_scale(tgt_prior):.2f}")

    # Inverse-frequency class weights on the supervised source terms. Computed
    # from the *training* half of the revealed labels only (the held-out split
    # stays untouched), normalised to mean 1 so the loss scale is unchanged.
    class_weights = None
    if getattr(cfg, "class_weighted_ce", False):
        power = float(getattr(cfg, "class_weight_power", 0.5))
        counts = torch.bincount(
            torch.cat([s.y[tr] for s, tr in zip(sources, src_labeled)]),
            minlength=num_classes,
        ).float().to(device)
        # A class with no labeled node is never indexed, but it must not be
        # allowed to set the normalising mean: `class_prior`'s 1e-6 floor would
        # give it a weight of ~1000 and shrink every real weight to nothing.
        present = counts > 0
        w = torch.ones_like(counts)
        w[present] = counts[present].pow(-power)
        class_weights = w / w[present].mean()
        print(f"Class-weighted CE on (inverse-frequency^{power}): weights "
              f"{class_weights[present].min():.2f}-"
              f"{class_weights[present].max():.2f} over "
              f"{int(present.sum())} labeled classes"
              f"{'' if bool(present.all()) else f' ({int((~present).sum())} of {num_classes} have no labeled node)'}")

    # FGW transferability weights; None until the first refresh.
    s_global = s_local = None
    select_on = (not cfg.no_da) and cfg.use_selection
    sel_sub = torch.randperm(target.num_nodes, device=device)[
        : min(cfg.select_target_nodes, target.num_nodes)
    ]

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
    val_key = _val_key(cfg, num_classes, n_val_total)
    val_label = f"src_val_{val_key}"
    if n_val_total:
        why = ("" if getattr(cfg, "val_metric", "auto") != "auto" else
               f" ({n_val_total} held-out nodes / {num_classes} classes = "
               f"{n_val_total / max(num_classes, 1):.1f} per class)")
        print(f"Held-out source split scored by "
              f"{'macro-F1' if val_key == 'f1' else 'accuracy'}{why}; this "
              f"drives both the warm-up plateau test and checkpoint selection.")
    # Every candidate criterion is tracked at each eval epoch, not just the one
    # doing the selecting: the end-of-run table then shows what each *would*
    # have picked and what that costs against the oracle.
    history = []

    best_state = None
    best_score = -float("inf")
    best_epoch = None

    # Phase schedule. The plateau trigger (and the rewind that goes with it)
    # needs the held-out source split; without it warm-up just runs to its cap.
    sched = PhaseScheduler(
        cfg, plateau=n_val_total > 0, proto_chance=math.log(num_classes),
    )
    scale_probe = {}          # one-shot flag for the [fgw-scale] diagnostic
    if not cfg.no_da:
        trigger = (f"held-out source F1 plateau (EMA, patience "
                   f"{sched.patience}, armed after {sched.min_epochs})"
                   if sched.plateau else "no plateau test (no source val split)")
        gate = ("" if sched.proto_ceiling is None
                else f", gated on L_proto <= {sched.proto_ceiling:.3f}")
        print(f"Schedule: warmup until {trigger}{gate}, capped at "
              f"{sched.warmup_epochs} epochs; then {sched.adapt_epochs} adapt "
              f"epochs (ramp {cfg.ramp_epochs}), then refine.")
        if sched.warmup_epochs >= cfg.epochs:
            print(f"  ! warmup_epochs ({sched.warmup_epochs}) >= epochs "
                  f"({cfg.epochs}): adaptation may never start.")
        if select_on:
            print(f"Source selection: FGW transferability (s_global, s_local) "
                  f"refreshed every {cfg.select_every} epochs against "
                  f"{sel_sub.numel()} target ego-graphs.")

    for epoch in range(1, cfg.epochs + 1):
        phase = sched.phase if not cfg.no_da else "srconly"

        # ---- refresh the FGW transferability weights
        # Only once adaptation has started: during warm-up the encoder is still
        # moving fast and the estimate would be measuring the initialisation.
        if (select_on and phase != "warmup"
                and (s_global is None or epoch % cfg.select_every == 0)):
            s_global, s_local, dom_d = transferability(
                model, sources, src_caches, src_labeled, target, tgt_cache,
                sel_sub, cfg,
            )
            selection_trace.append({
                "epoch": epoch,
                "weights": [float(v) for v in s_global.tolist()],
                "distances": [float(v) for v in dom_d.tolist()],
            })
            print("  [select] " + "  ".join(
                f"{n}: w {w:.2f} (d {d:.3f})"
                for n, w, d in zip(cfg.source_domains, s_global.tolist(),
                                   dom_d.tolist())
            ))

        stats = train_step(
            model, sources, target, src_caches, src_labeled, tgt_cache,
            optimizer, cfg, epoch,
            phase=phase, ramp_pos=sched.ramp_position(epoch),
            s_global=s_global, s_local=s_local,
            target_prior=tgt_prior, struct_density=struct_density,
            scale_probe=None if cfg.no_da else scale_probe,
            class_weights=class_weights,
        )

        is_eval = epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs
        # During warm-up the plateau test needs the validation score every
        # epoch, not just on the 5-epoch logging cadence — a switch that only
        # fires on multiples of 5 blurs the very boundary this is detecting.
        # It is a couple of cheap encodes; no ego-graphs or FGW solves.
        val_stats = (
            _source_val_metrics(model, sources, src_val)
            if is_eval or sched.needs_val() else None
        )

        if is_eval:
            tgt_stats = evaluate(
                model, target, cfg=cfg, cache=tgt_cache,
                source_prior=src_prior if cfg.prior_correct_eval else None,
                target_prior=tgt_prior if cfg.prior_correct_eval else None,
            )
            unsup = _target_criteria(model, target, snd_sub, snd_temp)
            record = {
                "epoch": epoch,
                "phase": stats["phase"],
                "src_val": val_stats[val_key] if val_stats else float("nan"),
                "src_val_f1": val_stats["f1"] if val_stats else float("nan"),
                "src_val_acc": val_stats["acc"] if val_stats else float("nan"),
                "src_train_f1": stats["src_train_f1"],
                "snd": unsup["snd"],
                "neg_ent": unsup["neg_ent"],
                "tgt_acc": tgt_stats["acc"],
                "tgt_f1": tgt_stats["f1"],
                "tgt_auc": tgt_stats["auc"],
                "loss": stats["loss"],
                "align_w": stats["align_w"],
                "ent_w": stats["ent_w"],
                **{k: stats[k] for k in (
                    "L_cls", "L_proto", "L_margin", "L_align", "L_ent",
                    "L_sep", "L_vrex", "L_struct", "L_pl")},
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
                f"{val_label} {record['src_val']:.4f}" if val_stats
                else f"{val_label}   n/a "
            )
            print(
                f"Epoch {epoch:3d} [{stats['phase']:>7}] "
                f"loss {stats['loss']:.4f}  "
                f"cls {stats['L_cls']:.4f}  "
                f"pr {stats['L_proto']:.4f}  "
                f"mg {stats['L_margin']:.4f}  "
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

        if not cfg.no_da:
            was = sched.phase
            note = sched.step(
                epoch, model, val_stats[val_key] if val_stats else None,
                proto_loss=stats["L_proto"],
            )
            if note:
                print(note)
            if sched.phase != was:
                events.append({"epoch": epoch + 1, "from": was,
                               "to": sched.phase,
                               "note": note.splitlines()[0] if note else ""})
            if phase == "warmup" and sched.phase == "adapt":
                # Estimate the target prior *here*, from the warm-up-best model:
                # it is the last point at which the classifier is purely
                # source-trained, so BBSE's label-shift assumption is clean. Do
                # it later and the estimate is contaminated by the very IM term
                # it is supposed to be steering — a feedback loop that would let
                # the collapse justify itself. The estimate is then frozen.
                if cfg.estimate_prior and cfg.target_class_prior is None:
                    est, why = estimate_target_prior(
                        model, sources, src_val, target, num_classes, src_prior,
                        max_cond=cfg.bbse_max_cond,
                    )
                    if est is not None:
                        tgt_prior = est
                        print(f"  [BBSE] target prior estimated as "
                              f"{_fmt_prior(tgt_prior)} (source "
                              f"{_fmt_prior(src_prior)})")
                    else:
                        print(f"  [BBSE] estimate rejected ({why}); keeping the "
                              f"source prior {_fmt_prior(src_prior)}")
                # Adam's moment estimates describe the gradients of the run we
                # just rewound out of, and the loss surface changes anyway when
                # the target terms come on. Start the adapt stage clean.
                optimizer = _make_optimizer(model, cfg)

    _report_selection(history, criterion, best_epoch, val_label)

    if best_state is not None:
        print(f"Restoring checkpoint from epoch {best_epoch} "
              f"(selected by '{criterion}').")
        model.load_state_dict(best_state)

    # Everything the caller needs to score the final model exactly the way the
    # training loop did (same prediction mode, same prior correction), plus the
    # trace the artefact/figure layer saves (`src/run_artifacts.py`).
    oracle = max(history, key=lambda r: r["tgt_f1"]) if history else None
    return model, {
        "src_prior": src_prior,
        "tgt_prior": tgt_prior if cfg.prior_correct_eval else None,
        "src_caches": src_caches,
        "tgt_cache": tgt_cache,
        "s_global": s_global,
        "history": history,
        "events": events,
        "selection": selection_trace,
        "support": support_stats,
        "best_epoch": best_epoch,
        "oracle_epoch": oracle["epoch"] if oracle else None,
        "criterion": criterion,
        "val_label": val_label,
        "proto_chance": math.log(num_classes),
        "struct_density": struct_density,
        "class_weights": class_weights,
    }


@torch.no_grad()
def summarize_run(
    model: FGWPrototypeDA,
    sources: List[Data],
    target: Data,
    cfg: FGWConfig,
    ctx: dict,
    seed: int,
    name_width: int = 5,
) -> dict:
    """Score every graph, print the per-seed table, and return the full record.

    Shared by all five runners so the printed table, the saved `metrics.json`
    and the figures can never disagree. The returned dict is what
    `src/plots.py` renders: final scores, the training trace from `ctx`, and
    the target-side detail (per-class F1, confusion, true label prior) that
    only exists once the run is over.
    """
    from sklearn.metrics import confusion_matrix, f1_score

    dev_sources = [s.to(cfg.device) for s in sources]
    dev_target = target.to(cfg.device)

    print("\n" + "-" * 60)
    print(f"  Seed {seed} results")
    print("-" * 60)
    source_scores = {}
    for name, g, cache in zip(cfg.source_domains, dev_sources, ctx["src_caches"]):
        s = evaluate(model, g, cfg=cfg, cache=cache)
        source_scores[name] = {k: float(s[k]) for k in ("acc", "auc", "f1")}
        print(f"  Source {name:>{name_width}}: ACC {s['acc']:.4f}  "
              f"AUROC {s['auc']:.4f}  MacroF {s['f1']:.4f}")
    tgt = evaluate(
        model, dev_target, cfg=cfg, cache=ctx["tgt_cache"],
        source_prior=ctx["src_prior"], target_prior=ctx["tgt_prior"],
    )
    print(f"  Target {cfg.target_domain:>{name_width}}: ACC {tgt['acc']:.4f}  "
          f"AUROC {tgt['auc']:.4f}  MacroF {tgt['f1']:.4f}")

    logits = predict_logits(model, dev_target, cfg=cfg, cache=ctx["tgt_cache"])
    if ctx["tgt_prior"] is not None:
        logits = prior_corrected_logits(logits, ctx["src_prior"], ctx["tgt_prior"])
    pred = logits.argmax(dim=1).cpu().numpy()
    true = dev_target.y.cpu().numpy()
    labels = list(range(cfg.num_classes))

    rec = {
        "seed": seed,
        **{k: float(tgt[k]) for k in ("acc", "auc", "f1")},
        "loss": float(tgt.get("loss", float("nan"))),
        "source_scores": source_scores,
        "per_class_f1": f1_score(true, pred, average=None, labels=labels,
                                 zero_division=0).tolist(),
        "per_class_support": np.bincount(
            true, minlength=cfg.num_classes).tolist(),
        "confusion": confusion_matrix(true, pred, labels=labels).tolist(),
        "src_prior": ctx["src_prior"].cpu().tolist(),
        "tgt_true_prior": (
            np.bincount(true, minlength=cfg.num_classes) / max(len(true), 1)
        ).tolist(),
    }
    for key in ("history", "events", "selection", "support", "best_epoch",
                "oracle_epoch", "criterion", "val_label", "proto_chance",
                "struct_density"):
        rec[key] = ctx.get(key)
    return rec


def _report_selection(
    history: List[dict], criterion: str, best_epoch, val_label: str = "src_val_f1",
) -> None:
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
    for name, key in ((val_label, "src_val"), ("SND", "snd"),
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
