"""FGW prototype-graph domain adaptation on the Yelp POI networks.

Runs the same method as `main_fgw.py` / `main_citation_fgw.py` (shared encoder
+ parametric head + FGW prototype transfer) but on the six Yelp city graphs —
Madison, Glendale, Gilbert, Las Vegas, Toronto, Phoenix. Multi-source: train on
every city except one, adapt to the held-out target (Phoenix by default).

On first use everything is auto-downloaded: the raw Yelp dump (old round with the
paper's cities) from a public Google Drive via gdown, and the GLOVE vectors from
the Hugging Face mirror. The dump is streamed once and every city graph is cached
under `--data_root`; later runs just load the cache. Use `--no_download` to
require manually placed files, `--gdrive_id` / `--glove_url` for different
sources — see `data/yelp/README.md`.

Usage:
    python main_yelp_fgw.py                                   # -> Phoenix target
    python main_yelp_fgw.py --target Toronto                  # sources auto-fill
    python main_yelp_fgw.py --sources Madison Glendale --target Phoenix
    python main_yelp_fgw.py --glove_path data/yelp/glove/glove.6B.100d.txt
"""

import argparse

import torch

from src.yelp_loader import (
    YELP_CITIES,
    YELP_FEATURE_DIM,
    YELP_NUM_CLASSES,
    YELP_TARGET_CITY,
    load_sources_target,
)
from src.fgw_config import FGWConfig
from src.fgw_model import FGWPrototypeDA
from src.fgw_train import evaluate, run_training
from src.utils import set_seed


def parse_args() -> FGWConfig:
    parser = argparse.ArgumentParser(description="FGW-prototype DA on Yelp")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="source cities; default = all Yelp cities except "
                             "--target")
    parser.add_argument("--target", type=str, default=YELP_TARGET_CITY,
                        choices=YELP_CITIES)
    parser.add_argument("--data_root", type=str, default="data/yelp")
    parser.add_argument("--glove_path", type=str, default=None,
                        help="GLOVE vectors file (default: "
                             "data/yelp/glove/glove.6B.300d.txt)")
    parser.add_argument("--glove_url", type=str, default=None,
                        help="URL of the glove.6B.zip to auto-download "
                             "(default: the Hugging Face mirror)")
    parser.add_argument("--no_download", action="store_true",
                        help="do NOT auto-download; require the raw dump to be "
                             "present under --data_root")
    parser.add_argument("--gdrive_id", type=str, default=None,
                        help="Google-Drive file id of the Yelp .tgz to download "
                             "(default: the built-in old-round SelMAG dump)")
    parser.add_argument("--source_label_ratio", type=float, default=0.1,
                        help="fraction of source nodes whose labels are revealed "
                             "to the supervised loss (supervision budget); the "
                             "rest stay in the graph but their labels are hidden. "
                             "Default 0.1 matches the SelMAG 10%%-labeled setting; "
                             "use 1.0 for all source labels")
    parser.add_argument("--source_label_stratified", action="store_true",
                        help="draw the labeled subset per-class (stratified) "
                             "instead of a pure random draw over all nodes")
    parser.add_argument("--min_common_reviewers", type=int, default=1,
                        help="min shared reviewers for a co-review edge")
    parser.add_argument("--max_user_degree", type=int, default=100,
                        help="skip users reviewing more than this many POIs in a "
                             "city (bounds the co-review pair blow-up)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--ego_size", type=int, default=32)
    parser.add_argument("--proto_size", type=int, default=32)
    parser.add_argument("--num_protos", type=int, default=3)
    parser.add_argument("--head_hidden", type=int, default=None,
                        help="MLP head width (defaults to hidden_dim)")
    parser.add_argument("--head_dropout", type=float, default=0.5)
    parser.add_argument("--no_layernorm", action="store_true",
                        help="disable LayerNorm on the encoder output")
    parser.add_argument("--no_da", action="store_true",
                        help="diagnostic: encoder+head on sources only (no DA)")
    parser.add_argument("--fgw_alpha", type=float, default=0.5)
    parser.add_argument("--fgw_epsilon", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--lambda_proto", type=float, default=0.3)
    parser.add_argument("--lambda_sep", type=float, default=1.0)
    parser.add_argument("--lambda_pl", type=float, default=0.1)
    parser.add_argument("--lambda_vrex", type=float, default=1.0)
    parser.add_argument("--lambda_struct", type=float, default=1e-3)
    parser.add_argument("--sep_intra_margin", type=float, default=0.5)
    parser.add_argument("--pl_threshold", type=float, default=0.8)
    parser.add_argument("--nodes_per_step", type=int, default=128)
    parser.add_argument("--warmup_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--directed", action="store_true",
                        help="keep co-review edges directed (default: symmetrise)")
    args = parser.parse_args()

    # "Every source except one": default sources = all cities but the target.
    if args.sources is None:
        args.sources = [c for c in YELP_CITIES if c != args.target]

    unknown = [c for c in args.sources + [args.target] if c not in YELP_CITIES]
    if unknown:
        raise ValueError(f"unknown Yelp city(ies) {unknown}; "
                         f"choose from {YELP_CITIES}")
    if args.target in args.sources:
        raise ValueError(
            f"target '{args.target}' must not also be a source: {args.sources}"
        )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    head_hidden = args.head_hidden if args.head_hidden is not None else args.hidden_dim

    cfg = FGWConfig(
        data_root=args.data_root,
        source_domains=args.sources,
        target_domain=args.target,
        feature_dim=YELP_FEATURE_DIM,
        num_classes=YELP_NUM_CLASSES,
        source_label_ratio=args.source_label_ratio,
        source_label_stratified=args.source_label_stratified,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        use_layernorm=not args.no_layernorm,
        head_hidden=head_hidden,
        head_dropout=args.head_dropout,
        no_da=args.no_da,
        ego_size=args.ego_size,
        proto_size=args.proto_size,
        num_protos=args.num_protos,
        fgw_alpha=args.fgw_alpha,
        fgw_epsilon=args.fgw_epsilon,
        tau=args.tau,
        lambda_proto=args.lambda_proto,
        lambda_sep=args.lambda_sep,
        lambda_pl=args.lambda_pl,
        lambda_vrex=args.lambda_vrex,
        lambda_struct=args.lambda_struct,
        sep_intra_margin=args.sep_intra_margin,
        pl_threshold=args.pl_threshold,
        target_class_prior=None,
        nodes_per_step=args.nodes_per_step,
        warmup_frac=args.warmup_frac,
        seed=args.seed,
        device=device,
    )
    cfg.symmetrize = not args.directed
    cfg.glove_path = args.glove_path
    cfg.min_common_reviewers = args.min_common_reviewers
    cfg.max_user_degree = args.max_user_degree
    cfg.auto_download = not args.no_download
    cfg.gdrive_id = args.gdrive_id
    cfg.glove_url = args.glove_url
    return cfg


def _label_hist(y: torch.Tensor, num_classes: int) -> str:
    counts = torch.bincount(y, minlength=num_classes).tolist()
    return " ".join(f"{c}:{n}" for c, n in enumerate(counts))


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    print("=" * 60)
    print("  FGW prototype-graph Domain Adaptation  (Yelp)")
    print("=" * 60)
    print(f"  Sources       : {cfg.source_domains}")
    print(f"  Target        : {cfg.target_domain}")
    _draw = "stratified" if cfg.source_label_stratified else "random"
    print(f"  Source labels : {cfg.source_label_ratio:.0%} ({_draw} draw; "
          f"target 0%)")
    print(f"  Device        : {cfg.device}")
    print(f"  feature_dim   : {cfg.feature_dim}")
    print(f"  num_classes   : {cfg.num_classes}")
    print(f"  proj/hidden   : {cfg.proj_dim} / {cfg.hidden_dim}")
    print(f"  ego_size k    : {cfg.ego_size}")
    print(f"  proto_size n_p: {cfg.proto_size}  (M={cfg.num_protos} per class)")
    print(f"  fgw alpha,eps : {cfg.fgw_alpha}, {cfg.fgw_epsilon}")
    print(f"  head/LN       : {cfg.head_hidden} / {cfg.use_layernorm}")
    print(f"  mode          : {'source-only (no DA)' if cfg.no_da else 'FGW domain adaptation'}")
    print("=" * 60)

    print("\nLoading data ...")
    sources, target = load_sources_target(
        cfg.data_root, cfg.source_domains, cfg.target_domain,
        glove_path=cfg.glove_path,
        min_common_reviewers=cfg.min_common_reviewers,
        max_user_degree=cfg.max_user_degree,
        symmetrize=getattr(cfg, "symmetrize", True),
        auto_download=getattr(cfg, "auto_download", True),
        gdrive_id=getattr(cfg, "gdrive_id", None),
        glove_url=getattr(cfg, "glove_url", None),
    )

    # Derive the feature/class dims from the data itself and keep the config
    # honest (guards against a GLOVE dim / class count mismatch).
    feat_dims = {g.x.size(1) for g in sources + [target]}
    if len(feat_dims) != 1:
        raise ValueError(f"inconsistent feature dims across cities: {feat_dims}")
    cfg.feature_dim = feat_dims.pop()
    cfg.num_classes = int(max(int(g.y.max()) for g in sources + [target]) + 1)

    for name, g in zip(cfg.source_domains, sources):
        print(f"  {name}: {g.num_nodes} nodes, {g.edge_index.size(1)} edges, "
              f"labels [{_label_hist(g.y, cfg.num_classes)}]")
    print(f"  {cfg.target_domain} (target): {target.num_nodes} nodes, "
          f"{target.edge_index.size(1)} edges, "
          f"labels [{_label_hist(target.y, cfg.num_classes)}]")

    model = FGWPrototypeDA(
        in_dim=cfg.feature_dim,
        proj_dim=cfg.proj_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        num_protos=cfg.num_protos,
        proto_size=cfg.proto_size,
        anchor_weight=cfg.anchor_weight,
        adjacency_temp=cfg.adjacency_temp,
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
        use_layernorm=cfg.use_layernorm,
        embed_init_scale=cfg.embed_init_scale,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    print("\nTraining ...\n")
    model = run_training(model, sources, target, cfg)

    sources = [s.to(cfg.device) for s in sources]
    target = target.to(cfg.device)

    print("\n" + "=" * 60)
    print("  Final results")
    print("=" * 60)
    for name, g in zip(cfg.source_domains, sources):
        s = evaluate(model, g)
        print(f"  Source {name:>10}: ACC {s['acc']:.4f}  "
              f"AUROC {s['auc']:.4f}  MacroF {s['f1']:.4f}")
    tgt_stats = evaluate(model, target)
    print(f"  Target {cfg.target_domain:>10}: ACC {tgt_stats['acc']:.4f}  "
          f"AUROC {tgt_stats['auc']:.4f}  MacroF {tgt_stats['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
