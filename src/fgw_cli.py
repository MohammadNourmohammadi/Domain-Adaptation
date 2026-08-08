"""Shared command-line surface for the three FGW runners.

`main_fgw.py`, `main_citation_fgw.py` and `main_yelp_fgw.py` differ only in how
they load data; every method-level flag is identical. Keeping the flags here
means a new loss term or schedule knob is wired into all three at once, rather
than being added to one runner and silently missing from the others (which is
how the Twitch runner ended up the only one able to reach half the config).

    add_method_args(parser, ego_size=16)      # per-dataset default overrides
    ...
    cfg = FGWConfig(**method_kwargs(args), <dataset-specific fields>)
"""

import argparse


def add_method_args(parser: argparse.ArgumentParser, **defaults) -> None:
    """Register every method flag. `defaults` overrides any listed default."""

    def d(name, fallback):
        return defaults.get(name, fallback)

    g = parser.add_argument_group("supervision budget")
    g.add_argument("--source_label_ratio", type=float,
                   default=d("source_label_ratio", 0.1),
                   help="fraction of source nodes whose labels are revealed to "
                        "the supervised loss; the rest stay in the graph but "
                        "their labels are hidden. 1.0 = full supervision")
    g.add_argument("--source_label_stratified",
                   action=argparse.BooleanOptionalAction,
                   default=d("source_label_stratified", False),
                   help="draw the labeled subset per-class instead of at "
                        "random; on a long-tailed label set a pure random 10%% "
                        "draw leaves whole classes with no supervision at all")
    g.add_argument("--source_val_frac", type=float,
                   default=d("source_val_frac", 0.2),
                   help="fraction of the revealed source labels held out of the "
                        "loss and used for model selection")
    g.add_argument("--model_selection", type=str,
                   default=d("model_selection", "src_val"),
                   choices=["src_val", "snd", "entropy", "last"],
                   help="checkpoint criterion (target labels are never used)")
    g.add_argument("--val_metric", type=str, default=d("val_metric", "auto"),
                   choices=["auto", "f1", "acc"],
                   help="what the held-out source split is scored by, for both "
                        "checkpoint selection and the warm-up plateau test. "
                        "'auto' falls back from macro-F1 to accuracy when the "
                        "split is too small to estimate C per-class F1 scores")

    g = parser.add_argument_group("run artefacts")
    g.add_argument("--out_dir", type=str, default=d("out_dir", "runs"),
                   help="where each run's own numbered folder is created "
                        "(log, config, history.csv, metrics.json, figures/)")
    g.add_argument("--no_figures", action="store_true",
                   help="save the log and the CSV/JSON but skip the figures")
    g.add_argument("--no_save", action="store_true",
                   help="do not create a run folder at all")
    g.add_argument("--note", type=str, default="",
                   help="free-text note stored in config.json and index.csv, "
                        "e.g. --note 'ablation: no kNN augmentation'")

    g = parser.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=d("epochs", 350))
    g.add_argument("--lr", type=float, default=d("lr", 1e-3))
    g.add_argument("--weight_decay", type=float, default=d("weight_decay", 5e-3))
    g.add_argument("--proto_weight_decay", type=float,
                   default=d("proto_weight_decay", 0.0),
                   help="weight decay for the prototype bank; keep at 0. Z is "
                        "LayerNormed so decay cannot change any output, and on "
                        "E_logits it drives C_p to a constant matrix")
    g.add_argument("--grad_clip", type=float, default=d("grad_clip", 1.0),
                   help="global gradient-norm clip (0 disables)")
    g.add_argument("--nodes_per_step", type=int, default=d("nodes_per_step", 128))
    g.add_argument("--seed", type=int, default=d("seed", 42))
    g.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run once per seed and report mean +- std, as the paper "
                        "does (it averages 5 runs). Overrides --seed")

    g = parser.add_argument_group("encoder / head")
    g.add_argument("--proj_dim", type=int, default=d("proj_dim", 128))
    g.add_argument("--hidden_dim", type=int, default=d("hidden_dim", 32))
    g.add_argument("--no_svd_proj", action="store_true",
                   help="learn the input projection instead of freezing it to "
                        "the unsupervised SVD basis")
    g.add_argument("--head_hidden", type=int, default=None,
                   help="MLP head width (defaults to hidden_dim)")
    g.add_argument("--head_dropout", type=float, default=d("head_dropout", 0.5))
    g.add_argument("--no_layernorm", action="store_true",
                   help="disable LayerNorm on the encoder output")
    g.add_argument("--no_da", action="store_true",
                   help="diagnostic: encoder+head on sources only (no DA)")

    g = parser.add_argument_group("FGW geometry")
    g.add_argument("--ego_size", type=int, default=d("ego_size", 12),
                   help="k; too large and the ego-graph's shortest-path matrix "
                        "flattens and the structural half of FGW says nothing")
    g.add_argument("--proto_size", type=int, default=d("proto_size", 8),
                   help="n_p; too large and the entropic plan stays diffuse, "
                        "making every class equidistant from every ego-graph")
    g.add_argument("--knn_augment", type=int, default=d("knn_augment", 0),
                   metavar="K",
                   help="add K feature-space kNN edges to every node with "
                        "degree < --knn_min_degree, on the sources and the "
                        "target alike (labels are never read). 0 = off. Meant "
                        "for the induced-subgraph splits (Cora_full, Arxiv) "
                        "where 32-50%% of nodes are isolated and neither the "
                        "GCN nor the ego-graph has anything to work with")
    g.add_argument("--knn_min_degree", type=int, default=d("knn_min_degree", 3),
                   help="degree below which a node counts as low-degree for "
                        "--knn_augment")
    g.add_argument("--num_protos", type=int, default=d("num_protos", 3))
    g.add_argument("--fgw_alpha", type=float, default=d("fgw_alpha", 0.25))
    g.add_argument("--fgw_epsilon", type=float, default=d("fgw_epsilon", 0.05))
    g.add_argument("--tau", type=float, default=d("tau", 0.5))
    g.add_argument("--predict", type=str, default=d("predict", "head"),
                   choices=["head", "fgw", "ensemble"],
                   help="what produces the reported logits; the FGW paths cost "
                        "an ego-graph + solve per scored node and only pay off "
                        "once L_proto is well below ln C")

    g = parser.add_argument_group("loss weights")
    g.add_argument("--lambda_proto", type=float, default=d("lambda_proto", 0.3))
    g.add_argument("--lambda_align", type=float, default=d("lambda_align", 1.0))
    g.add_argument("--lambda_ent", type=float, default=d("lambda_ent", 0.5))
    g.add_argument("--lambda_sep", type=float, default=d("lambda_sep", 1.0))
    g.add_argument("--lambda_pl", type=float, default=d("lambda_pl", 0.1))
    g.add_argument("--lambda_fgw_margin", type=float,
                   default=d("lambda_fgw_margin", 0.5),
                   help="hinge separating the correct-class FGW distance from "
                        "the nearest wrong one; this is what makes the "
                        "prototypes class-discriminative at all")
    g.add_argument("--fgw_margin", type=float, default=d("fgw_margin", 0.5))
    g.add_argument("--class_weighted_ce", action=argparse.BooleanOptionalAction,
                   default=d("class_weighted_ce", False),
                   help="inverse-frequency class weights on the supervised "
                        "source terms. Plain CE optimises accuracy; on a "
                        "long-tailed label set that leaves macro-F1 — the "
                        "headline metric — paying for the tail")
    g.add_argument("--class_weight_power", type=float,
                   default=d("class_weight_power", 0.5),
                   help="exponent on the inverse frequency (1.0 = full, "
                        "0.5 = square-root, the usual better trade)")
    g.add_argument("--im_balance_weight", type=float,
                   default=d("im_balance_weight", 1.0),
                   help="weight on the class-balance half of IM; 0 turns IM into "
                        "pure entropy minimisation, whose optimum is predicting "
                        "one class everywhere")
    g.add_argument("--lambda_vrex", type=float, default=d("lambda_vrex", 0.0),
                   help="V-REx weight; 0 keeps it as a log-only instability "
                        "diagnostic (the default)")
    g.add_argument("--lambda_struct", type=float, default=d("lambda_struct", 1e-2))
    g.add_argument("--struct_density", type=float, default=None,
                   help="target prototype edge density; default calibrates it "
                        "from the data so mean(C_p) = mean(C_ego)")
    g.add_argument("--pl_threshold", type=float, default=d("pl_threshold", 0.8))
    g.add_argument("--sep_intra_margin", type=float, default=0.5,
                   help="(deprecated, ignored)")

    g = parser.add_argument_group("label shift")
    g.add_argument("--target_prior", type=float, nargs="+", default=None,
                   metavar="P",
                   help="known target class prior, e.g. --target_prior 0.755 "
                        "0.245; default estimates it by BBSE")
    g.add_argument("--estimate_prior", action="store_true",
                   help="estimate the target prior by BBSE from the held-out "
                        "source split. Off by default: BBSE assumes the domains "
                        "differ by label shift alone, and when that fails it "
                        "over-states the majority class — which is the one "
                        "direction that feeds collapse. The pooled source prior "
                        "is used otherwise")
    g.add_argument("--prior_correct", action="store_true",
                   help="re-base the logits onto the target prior before argmax "
                        "at evaluation. Off by default: it is Bayes-optimal for "
                        "accuracy, but accuracy on an imbalanced target is "
                        "maximised by the majority rule, so it costs macro-F1")
    g.add_argument("--bbse_max_cond", type=float, default=4.0,
                   help="reject the BBSE prior estimate above this cond(P(pred|y)) "
                        "and fall back to the source prior; BBSE inverts the "
                        "confusion matrix and over-states the majority class "
                        "once the classifier nears chance")

    g = parser.add_argument_group("source selection")
    g.add_argument("--no_selection", action="store_true",
                   help="disable the FGW transferability weights and average "
                        "the sources uniformly (the ablation)")
    g.add_argument("--select_every", type=int, default=d("select_every", 20))
    g.add_argument("--select_target_nodes", type=int,
                   default=d("select_target_nodes", 32))

    g = parser.add_argument_group("phase schedule")
    g.add_argument("--warmup_epochs", type=int, default=d("warmup_epochs", 150),
                   help="hard cap on warm-up, in absolute epochs (it normally "
                        "ends earlier, on a held-out source F1 plateau)")
    g.add_argument("--warmup_min_epochs", type=int,
                   default=d("warmup_min_epochs", 40),
                   help="epochs before the plateau counter is armed")
    g.add_argument("--warmup_patience", type=int,
                   default=d("warmup_patience", 40),
                   help="epochs without an EMA-smoothed held-out source F1 gain "
                        "that end warm-up (the best warm-up checkpoint is "
                        "restored before adaptation starts)")
    g.add_argument("--warmup_proto_gate", type=float,
                   default=d("warmup_proto_gate", 0.9),
                   help="warm-up also waits for L_proto <= gate*ln(C); 0 "
                        "disables, so adaptation may start against prototypes "
                        "carrying no class information")
    g.add_argument("--adapt_epochs", type=int, default=d("adapt_epochs", 120))
    g.add_argument("--ramp_epochs", type=int, default=d("ramp_epochs", 20))


def method_kwargs(args) -> dict:
    """Translate parsed args into `FGWConfig` keyword arguments."""
    head_hidden = args.head_hidden if args.head_hidden is not None else args.hidden_dim
    return dict(
        out_dir=args.out_dir,
        no_figures=args.no_figures,
        no_save=args.no_save,
        note=args.note,
        source_label_ratio=args.source_label_ratio,
        source_label_stratified=args.source_label_stratified,
        source_val_frac=args.source_val_frac,
        model_selection=args.model_selection,
        val_metric=args.val_metric,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proto_weight_decay=args.proto_weight_decay,
        grad_clip=args.grad_clip,
        nodes_per_step=args.nodes_per_step,
        seed=args.seed,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        svd_proj=not args.no_svd_proj,
        use_layernorm=not args.no_layernorm,
        head_hidden=head_hidden,
        head_dropout=args.head_dropout,
        no_da=args.no_da,
        ego_size=args.ego_size,
        proto_size=args.proto_size,
        num_protos=args.num_protos,
        knn_augment=args.knn_augment,
        knn_min_degree=args.knn_min_degree,
        fgw_alpha=args.fgw_alpha,
        fgw_epsilon=args.fgw_epsilon,
        tau=args.tau,
        predict=args.predict,
        lambda_proto=args.lambda_proto,
        lambda_align=args.lambda_align,
        lambda_ent=args.lambda_ent,
        lambda_sep=args.lambda_sep,
        lambda_pl=args.lambda_pl,
        lambda_fgw_margin=args.lambda_fgw_margin,
        fgw_margin=args.fgw_margin,
        im_balance_weight=args.im_balance_weight,
        class_weighted_ce=args.class_weighted_ce,
        class_weight_power=args.class_weight_power,
        lambda_vrex=args.lambda_vrex,
        lambda_struct=args.lambda_struct,
        struct_density=args.struct_density,
        sep_intra_margin=args.sep_intra_margin,
        pl_threshold=args.pl_threshold,
        target_class_prior=(tuple(args.target_prior)
                            if args.target_prior is not None else None),
        estimate_prior=args.estimate_prior,
        prior_correct_eval=args.prior_correct,
        bbse_max_cond=args.bbse_max_cond,
        use_selection=not args.no_selection,
        select_every=args.select_every,
        select_target_nodes=args.select_target_nodes,
        warmup_epochs=args.warmup_epochs,
        warmup_min_epochs=args.warmup_min_epochs,
        warmup_patience=args.warmup_patience,
        warmup_proto_gate=args.warmup_proto_gate,
        adapt_epochs=args.adapt_epochs,
        ramp_epochs=args.ramp_epochs,
    )


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_list(args) -> list:
    return args.seeds if args.seeds else [args.seed]
