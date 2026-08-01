"""Configuration for the FGW prototype-graph domain adaptation pipeline.

This is intentionally a separate dataclass from `src.config.Config` so the
new method does not interfere with the existing Causal-DANN setup. Defaults
follow the hyperparameter suggestions in the method note.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FGWConfig:
    # ------------------------------------------------------------------ data
    data_root: str = "data"
    source_domains: List[str] = field(default_factory=lambda: ["DE", "FR"])
    target_domain: str = "ENGB"
    feature_dim: int = 3170

    # Supervision budget: the fraction of *source* nodes whose labels are revealed
    # to the supervised CE loss. The remaining nodes stay in the graph — their
    # features and edges still flow through message passing — but their labels are
    # never shown to the loss; the target is always 0% labeled. 1.0 = every source
    # label. All three runners (Twitch/Citation/Yelp) override this to 0.1 by
    # default to match the SelMAG "10% labeled" protocol; pass --source_label_ratio
    # 1.0 for full supervision. Restricting to 10% also lets source ego-graphs be
    # precomputed for just the labeled subset (see fgw_train._make_cache). The
    # labeled subset is a random split drawn once per run (re-drawn when the seed
    # changes). This dataclass default stays 1.0 as the neutral library fallback.
    source_label_ratio: float = 1.0
    source_label_stratified: bool = False  # per-class draw vs. a pure random draw

    # Fraction of the *revealed* source labels held out of the CE loss and used
    # only for model selection. Without this there is no legitimate checkpoint
    # criterion: the training log's src_f1 is measured on the very nodes just
    # backpropped through, and picking an epoch by target metrics would be
    # oracle selection. 0.0 disables the split (and falls back to last-epoch
    # reporting unless an unsupervised criterion is chosen).
    source_val_frac: float = 0.2

    # Checkpoint criterion. "src_val": pooled held-out source macro-F1 (the
    # honest default). "snd" / "entropy": unsupervised target criteria, usable
    # when no source labels may be held out. "last": report the final epoch.
    # Target labels are never a selection input under any setting; the oracle
    # target number is printed for reference only.
    model_selection: str = "src_val"
    snd_temp: float = 0.05        # SND similarity softmax temperature
    snd_max_nodes: int = 2000     # subsample cap for the O(N^2) SND matrix

    # --------------------------------------------------------------- encoder
    proj_dim: int = 128
    hidden_dim: int = 32          # = d, the FGW embedding dimension
    use_layernorm: bool = True    # LayerNorm encoder output (scale-match FGW)

    # Replace the learned Linear(feature_dim, proj_dim) with a frozen basis of
    # the top `proj_dim` right singular vectors of the stacked (unlabeled)
    # feature matrix. The learned version is >90% of the model's parameters and
    # ~100 parameters per labeled node under a 10% budget, so it memorises the
    # seeds within a few dozen epochs; the SVD basis is fitted without labels
    # and leaves 0 trainable parameters in that layer.
    svd_proj: bool = True

    # ------------------------------------------------- parametric classifier
    head_hidden: int = 32         # hidden width of the MLP prediction head
    head_dropout: float = 0.5
    no_da: bool = False           # diagnostic: encoder+head on sources only

    # -------------------------------------------------------------- ego graph
    ego_size: int = 32            # k = center + (k-1) PPR neighbours
    ppr_alpha: float = 0.15       # restart probability
    ppr_iters: int = 20           # power-iteration steps
    anchor_weight: float = 1.0    # w on the anchor indicator coordinate
    anchor_mass_extra: float = 0.2  # extra probability mass on the center

    # --------------------------------------------------------- prototype bank
    num_classes: int = 2
    num_protos: int = 3           # M prototype graphs per class
    proto_size: int = 32          # n_p nodes per prototype
    adjacency_temp: float = 1.0   # temperature on the soft-adjacency sigmoid
    embed_init_scale: float = 1.0  # std of prototype embedding init

    # ------------------------------------------------------------ FGW solver
    fgw_alpha: float = 0.25       # trade-off between feature and structure
                                  # (low: structure ~uninformative on dense graphs)
    fgw_epsilon: Optional[float] = 0.05  # entropic Sinkhorn FGW regularisation
    fgw_max_iter: int = 50        # outer block-coordinate (FGW) iterations

    # -------------------------------------------------- classifier soft-min
    tau: float = 0.5              # temperature

    # ----------------------------------------------------------- loss weights
    lambda_proto: float = 0.3     # aux source CE through FGW (keeps protos
                                  # class-meaningful so alignment has anchors)
    lambda_align: float = 1.0
    lambda_ent: float = 0.5
    lambda_sep: float = 1.0       # cosine repulsion between class prototypes
    lambda_pl: float = 0.1
    # V-REx off by default. The variance of the per-source risks is ~0 whenever
    # training is healthy and only spikes on a blow-up, so as a loss it adds
    # nothing and as a *diagnostic* it is genuinely useful — L_vrex is still
    # computed and logged every epoch (see fgw_train.train_step), just not
    # optimised. Set > 0 to put it back in the objective.
    lambda_vrex: float = 0.0
    lambda_struct: float = 1e-3
    # Deprecated, ignored: both belonged to the FGW-distance hinge that
    # `separation_loss` replaced (the inter-class margin was unsatisfiable and
    # the intra-class one fought the M-way soft-min). Kept as fields so existing
    # configs and CLI wrappers keep loading.
    sep_margin: float = 1.0
    sep_intra_margin: float = 0.5
    pl_threshold: float = 0.8
    target_class_prior: Optional[Tuple[float, float]] = None

    # ---------------------------------------------------- training schedule
    lr: float = 1e-3
    weight_decay: float = 5e-3
    epochs: int = 100

    # Global gradient-norm clip applied on every step. The FGW feature cost is
    # quadratic in the embeddings and feeds a logsumexp, so occasional huge
    # gradients reach the shared encoder through L_proto; unclipped runs showed
    # periodic blow-ups (confidently-wrong CE ~2.3 followed by dozens of epochs
    # of recovery). 0 disables clipping.
    grad_clip: float = 1.0

    # Phase lengths are *absolute epoch counts*, not fractions of `epochs`.
    # Fractions coupled the schedule to the budget the wrong way round: with
    # warmup_frac=0.2, --epochs 350 started adapting at 70 and --epochs 1000 at
    # 200, so raising the budget only bought more epochs of overfitting before
    # any target signal arrived (and both runs then selected the same
    # checkpoint). The encoder saturates in a few dozen epochs regardless of
    # how long the run is, so the switch has to be pinned to that scale.
    warmup_epochs: int = 60       # hard cap on warm-up (L_cls + L_proto only)
    adapt_epochs: int = 120       # epochs of ramped align/IM before refine
    ramp_epochs: int = 20         # sigmoid ramp on align/ent weights

    # Warm-up normally ends *before* `warmup_epochs` on a held-out-source-F1
    # plateau: no improvement over the running best for `warmup_patience`
    # consecutive epochs. The best warm-up checkpoint is then restored before
    # adaptation starts — adapting from an already-overfit encoder leaves
    # nothing worth preserving. Needs `source_val_frac > 0`; without it the
    # switch falls back to the fixed `warmup_epochs` cap.
    warmup_patience: int = 15
    warmup_min_delta: float = 1e-4  # F1 gain that counts as an improvement

    # ----------------------------------------------- mini-batching over nodes
    nodes_per_step: int = 128
    eval_batch_nodes: int = 512

    # ------------------------------------------------------------- system
    device: str = "cpu"
    seed: int = 42
