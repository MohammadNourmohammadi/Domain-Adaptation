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
    # Twitch feature set. "musae" is the authors' 3170-d bag-of-words (the
    # only one still obtainable); "pyg" the 128-d release SelMAG used, whose
    # host is gone. Same six graphs either way. Ignored by the Citation / Yelp
    # runners, which set `feature_dim` directly.
    features: str = "musae"
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

    # Checkpoint criterion. "src_val": pooled held-out source score (the
    # honest default). "snd" / "entropy": unsupervised target criteria, usable
    # when no source labels may be held out. "last": report the final epoch.
    # Target labels are never a selection input under any setting; the oracle
    # target number is printed for reference only.
    model_selection: str = "src_val"

    # Which statistic `src_val` scores, and what the warm-up plateau test
    # watches. Macro-F1 is the metric being reported, so selecting on it is the
    # aligned choice — but it is an average over C per-class F1 scores, and with
    # few held-out nodes per class it is mostly variance. Cora_full is the case
    # that breaks: a 10% budget on 3,299-node graphs leaves 66 validation nodes
    # per source, 330 pooled, spread over 70 classes — half of them with zero or
    # one node — so the "best" epoch is close to a lottery, and that same signal
    # also drives the warm-up rewind. "auto" uses macro-F1 when the pooled split
    # holds at least `val_f1_min_per_class` nodes per class and plain accuracy
    # otherwise (lower variance, slightly misaligned under imbalance); "f1" and
    # "acc" force the choice.
    val_metric: str = "auto"
    val_f1_min_per_class: float = 10.0

    # Inverse-frequency class weights on the supervised source terms (head CE,
    # L_proto, L_margin), normalised to mean 1 over the batch.
    #
    # Off by default because it changes the objective on every dataset. It is
    # worth turning on wherever macro-F1 is the headline and the label
    # distribution is long-tailed: Arxiv has 17 of 40 classes below 1% of nodes
    # and Cora_full 70 classes spanning 15-928 papers, which is exactly why both
    # rows of the paper show macro-F1 far below accuracy (38.3/26.4 and
    # 55.4/31.2). Plain CE optimises accuracy, so the tail is free to be ignored.
    class_weighted_ce: bool = False
    # Exponent on the inverse frequency: 1.0 = full inverse-frequency, 0.5 = the
    # gentler square-root form that is usually the better bias/variance trade.
    class_weight_power: float = 0.5
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
    # k = 12, not 32. Twitch graphs average 16-34 undirected neighbours per
    # node, so a 32-node PPR neighbourhood has diameter 2 and its
    # diameter-normalised shortest-path matrix collapses into {0, 1/3, 2/3}
    # with almost no spread: the structural half of the FGW cost becomes a
    # near-constant and the solver is paying to compute a plain Wasserstein
    # distance. A smaller ego-graph keeps real topological variation.
    ego_size: int = 12            # k = center + (k-1) PPR neighbours
    ppr_alpha: float = 0.15       # restart probability
    ppr_iters: int = 20           # power-iteration steps
    anchor_weight: float = 1.0    # w on the anchor indicator coordinate
    anchor_mass_extra: float = 0.2  # extra probability mass on the center

    # ------------------------------------------------- graph densification
    # Feature-space kNN edges added to nodes whose degree is below
    # `knn_min_degree`. 0 = off (the default: this changes the input graph, so
    # it belongs in the method description, not in the defaults).
    #
    # The six-graph splits of Cora_full and Arxiv induce subgraphs, which
    # shatters them: average degree 0.95-2.06 with 32-50% isolated nodes
    # (Cora_full) and 1.5-4.4 with 9-48% isolated (Arxiv), against 6.6-64 and
    # ~0% isolated on Citation/Twitch/Yelp. Below `ego_size` neighbours there is
    # no ego-graph to speak of and a 2-layer GCN degenerates to an MLP. Reads
    # `x` only — never a label — and is applied to sources and target alike.
    # See `src/graph_augment.py`.
    knn_augment: int = 0
    knn_min_degree: int = 3

    # --------------------------------------------------------- prototype bank
    num_classes: int = 2
    num_protos: int = 3           # M prototype graphs per class
    # n_p = 8, not 32. With 32 free prototype nodes the marginal constraints
    # keep the entropic plan diffuse, so the FGW value tends to
    # `const - 2<F_e, mean(F_p)>`; the mean of 32 near-random LayerNorm'd
    # vectors is ~0 for every class, which is how d_0(v) and d_1(v) ended up
    # identical for every ego-graph (L_proto pinned at ln 2 for 350 epochs).
    # Fewer nodes force each prototype to commit to a specific local pattern.
    proto_size: int = 8           # n_p nodes per prototype
    adjacency_temp: float = 1.0   # temperature on the soft-adjacency sigmoid
    embed_init_scale: float = 1.0  # std of prototype embedding init

    # ------------------------------------------------------------ FGW solver
    fgw_alpha: float = 0.25       # trade-off between feature and structure
                                  # (low: structure ~uninformative on dense graphs)
    fgw_epsilon: Optional[float] = 0.05  # entropic Sinkhorn FGW regularisation
    # Outer block-coordinate iterations. The solver exits early once the plan
    # stops moving, so this is a cap rather than a fixed cost.
    fgw_max_iter: int = 20

    # -------------------------------------------------- classifier soft-min
    tau: float = 0.5              # temperature

    # ----------------------------------------------------------- loss weights
    lambda_proto: float = 0.3     # aux source CE through FGW (keeps protos
                                  # class-meaningful so alignment has anchors)
    lambda_align: float = 1.0
    lambda_ent: float = 0.5
    lambda_sep: float = 1.0       # cosine repulsion between class prototypes
    lambda_pl: float = 0.1

    # Hinge on the source-conditional FGW distances. Cross-entropy alone has a
    # vanishing gradient exactly at d_0 = d_1, which is where the runs got
    # stuck; the hinge pushes with constant force until the correct class is
    # `fgw_margin` closer than the nearest wrong one. This is the term that
    # makes the FGW branch class-discriminative at all.
    lambda_fgw_margin: float = 0.5
    fgw_margin: float = 0.5

    # Weight on the class-balance half of the IM objective (the KL between the
    # mean target prediction and the assumed prior). Never set this to 0
    # without knowing why: pure conditional-entropy minimisation is optimised
    # by predicting one class everywhere.
    im_balance_weight: float = 1.0
    # V-REx off by default. The variance of the per-source risks is ~0 whenever
    # training is healthy and only spikes on a blow-up, so as a loss it adds
    # nothing and as a *diagnostic* it is genuinely useful — L_vrex is still
    # computed and logged every epoch (see fgw_train.train_step), just not
    # optimised. Set > 0 to put it back in the objective.
    lambda_vrex: float = 0.0
    # Penalty pulling each prototype's edge density towards `struct_density`.
    # The old form was plain L1 on the soft adjacency, which drove A -> 0 and
    # therefore C_p = 1 - A -> all-ones: a constant matrix with no structural
    # information. `struct_density=None` calibrates the target from the data
    # so that mean(C_p) matches mean(C_ego).
    lambda_struct: float = 1e-2
    struct_density: Optional[float] = None
    # Deprecated, ignored: both belonged to the FGW-distance hinge that
    # `separation_loss` replaced (the inter-class margin was unsatisfiable and
    # the intra-class one fought the M-way soft-min). Kept as fields so existing
    # configs and CLI wrappers keep loading.
    sep_margin: float = 1.0
    sep_intra_margin: float = 0.5
    pl_threshold: float = 0.8

    # ------------------------------------------------------------ label shift
    # Known target class prior, used by the IM balance term (and, if enabled,
    # the inference-time correction). When None the pooled *source* prior is
    # used, unless `estimate_prior` turns on BBSE.
    target_class_prior: Optional[Tuple[float, ...]] = None
    # BBSE is **off by default**. It assumes p(x | y) is domain-invariant, i.e.
    # that the domains differ by label shift alone. Twitch does not satisfy
    # that — the graphs differ in structure and feature distribution too — and
    # no source-side diagnostic can detect the violation, because the confusion
    # matrix BBSE inverts is measured entirely on source data. On Twitch it
    # confidently returned [0.844, 0.156] against a truth of [0.755, 0.245],
    # over-stating the majority class, and feeding that to the balance term
    # pushed the model toward the collapse the term exists to prevent.
    #
    # The source prior is the safer default: the balance term's job is to make
    # collapse unprofitable, which it does for any non-degenerate prior, and
    # erring toward the *minority* is the direction that helps macro-F1. Turn
    # BBSE on where label shift really is the dominant shift.
    estimate_prior: bool = False
    # Largest cond(P(pred|y)) at which the BBSE estimate is trusted; above it
    # the pooled source prior is used instead. See `estimate_target_prior`.
    bbse_max_cond: float = 4.0
    # Re-base the logits onto the target prior before argmax at evaluation.
    #
    # **Off by default, deliberately.** The correction is the Bayes-optimal rule
    # for 0-1 loss under label shift, but 0-1 loss on a graph that is 24.5%
    # positive is maximised by predicting the majority class — so the rule
    # optimises straight towards the degenerate solution. Measured on Twitch at
    # a fixed checkpoint: raw argmax ACC 0.638 / MacroF 0.544, corrected ACC
    # 0.751 / MacroF 0.434. It spent 11 points of the headline metric to buy 11
    # points of accuracy, and still landed below the 0.755 majority baseline it
    # was chasing. Macro-F1 is maximised at a quite different threshold.
    #
    # Left switchable because it is a legitimate ablation and the right choice
    # when accuracy is the target metric and the shift estimate is trustworthy.
    prior_correct_eval: bool = False

    # ------------------------------------------------- source selection (ours)
    # FGW-native replacement for SelMAG's meta-learned selector: per-domain
    # (s_global) and per-node (s_local) transferability read straight off the
    # ego-graph FGW geometry. See `fgw_select.transferability`.
    use_selection: bool = True
    select_every: int = 20        # refresh period, in epochs
    select_target_nodes: int = 32  # target ego-graphs sampled for the estimate
    select_temp_global: float = 0.1
    select_temp_local: float = 0.25
    select_clip: float = 5.0      # max up/down-weighting factor
    select_chunk: int = 256       # source egos per FGW batch

    # ----------------------------------------------------------- prediction
    # "head": the MLP head alone (default). "fgw": the prototype distances
    # alone. "ensemble": mean of the two log-probability vectors. The FGW paths
    # need an ego-graph + FGW solve for every scored node, so they cost far
    # more at evaluation; only worth enabling once L_proto is well below ln C.
    predict: str = "head"

    # ---------------------------------------------------- training schedule
    lr: float = 1e-3
    weight_decay: float = 5e-3
    # Weight decay for the prototype bank, which gets its own parameter group.
    # It must be 0. `PrototypeBank.embeddings()` LayerNorms Z, so the forward
    # pass is invariant to |Z| and decay shrinks the parameter without changing
    # any output — which silently inflates the effective angular step size over
    # training. On `E_logits` it is worse than useless: decay drives the logits
    # to 0, hence sigmoid -> 0.5, hence C_p -> a constant 0.5 matrix, erasing
    # the prototype structure the FGW distance is supposed to read.
    proto_weight_decay: float = 0.0
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
    warmup_epochs: int = 150      # hard cap on warm-up (L_cls + L_proto only)
    adapt_epochs: int = 120       # epochs of ramped align/IM before refine
    ramp_epochs: int = 20         # sigmoid ramp on align/ent weights

    # Warm-up normally ends *before* `warmup_epochs` on a held-out-source-F1
    # plateau. Three guards, all of which the previous version lacked and all
    # of which it tripped over on Twitch (it switched at epoch 29 having
    # rewound to a checkpoint scoring src_val 0.5413, while the very same model
    # went on to reach 0.6347 later in the run):
    #
    #  * `warmup_val_ema` smooths the plateau signal. Held-out macro-F1 over a
    #    few hundred pooled validation nodes is very noisy epoch to epoch, and
    #    the raw curve produced 15 consecutive "no gain" epochs during an
    #    ordinary dip.
    #  * `warmup_min_epochs` arms the patience counter only after the model has
    #    had a real chance to fit.
    #  * `warmup_proto_gate` additionally requires the FGW branch to have
    #    become class-discriminative — L_proto below `gate * ln(C)` — before
    #    the target terms come on. Alignment pulls the target toward the
    #    prototypes, so switching while L_proto is still at chance means
    #    aligning to noise. Set to 0 to disable the gate (the hard
    #    `warmup_epochs` cap always wins in the end either way).
    warmup_patience: int = 40
    warmup_min_epochs: int = 40
    warmup_min_delta: float = 1e-4  # F1 gain that counts as an improvement
    warmup_val_ema: float = 0.7     # EMA momentum on the plateau signal
    warmup_proto_gate: float = 0.9

    # ----------------------------------------------- mini-batching over nodes
    nodes_per_step: int = 128
    eval_batch_nodes: int = 512

    # ------------------------------------------------------- run artefacts
    # Each run gets its own numbered folder under `out_dir` holding the console
    # log, this config, per-epoch history, final metrics and the paper figures.
    # See `src/run_artifacts.py`.
    out_dir: str = "runs"
    no_figures: bool = False      # keep the log/CSV/JSON, skip the plots
    no_save: bool = False         # write nothing (the old behaviour)
    note: str = ""                # free text stored in config.json / index.csv

    # ------------------------------------------------------------- system
    device: str = "cpu"
    seed: int = 42
