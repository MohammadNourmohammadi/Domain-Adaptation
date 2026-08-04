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
import statistics

import torch

from src.yelp_loader import (
    YELP_CITIES,
    YELP_FEATURE_DIM,
    YELP_NUM_CLASSES,
    YELP_TARGET_CITY,
    load_sources_target,
)
from src.fgw_cli import add_method_args, method_kwargs, pick_device, seed_list
from src.fgw_config import FGWConfig
from src.fgw_model import FGWPrototypeDA
from src.fgw_train import evaluate, run_training
from src.utils import majority_baseline, set_seed


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
    parser.add_argument("--min_common_reviewers", type=int, default=1,
                        help="POIs are linked when at least this many reviewers "
                             "are shared")
    parser.add_argument("--max_user_degree", type=int, default=100,
                        help="drop reviewers above this POI count before "
                             "building co-review edges")
    parser.add_argument("--directed", action="store_true",
                        help="keep co-review edges directed (default: symmetrise)")
    # Yelp is 6-class with dense 300-d GLOVE features; everything below is the
    # shared method surface (see src/fgw_cli.py).
    add_method_args(parser, hidden_dim=64, fgw_alpha=0.5, ego_size=16)
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

    cfg = FGWConfig(
        data_root=args.data_root,
        source_domains=args.sources,
        target_domain=args.target,
        feature_dim=YELP_FEATURE_DIM,
        num_classes=YELP_NUM_CLASSES,
        device=pick_device(),
        **method_kwargs(args),
    )
    cfg.symmetrize = not args.directed
    cfg.glove_path = args.glove_path
    cfg.min_common_reviewers = args.min_common_reviewers
    cfg.max_user_degree = args.max_user_degree
    cfg.auto_download = not args.no_download
    cfg.gdrive_id = args.gdrive_id
    cfg.glove_url = args.glove_url
    return cfg, seed_list(args)


def _label_hist(y: torch.Tensor, num_classes: int) -> str:
    counts = torch.bincount(y, minlength=num_classes).tolist()
    return " ".join(f"{c}:{n}" for c, n in enumerate(counts))


def run_once(cfg: FGWConfig, sources, target, seed: int) -> dict:
    set_seed(seed)
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
        frozen_proj=cfg.svd_proj,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nTraining ...\n")
    model, ctx = run_training(model, sources, target, cfg)

    dev_sources = [s.to(cfg.device) for s in sources]
    dev_target = target.to(cfg.device)

    print("\n" + "-" * 60)
    print(f"  Seed {seed} results")
    print("-" * 60)
    for name, g, cache in zip(cfg.source_domains, dev_sources, ctx["src_caches"]):
        st = evaluate(model, g, cfg=cfg, cache=cache)
        print(f"  Source {name:>10}: ACC {st['acc']:.4f}  "
              f"AUROC {st['auc']:.4f}  MacroF {st['f1']:.4f}")
    tgt = evaluate(
        model, dev_target, cfg=cfg, cache=ctx["tgt_cache"],
        source_prior=ctx["src_prior"], target_prior=ctx["tgt_prior"],
    )
    print(f"  Target {cfg.target_domain:>10}: ACC {tgt['acc']:.4f}  "
          f"AUROC {tgt['auc']:.4f}  MacroF {tgt['f1']:.4f}")
    return tgt


def main():
    cfg, seeds = parse_args()
    set_seed(cfg.seed)

    print("=" * 60)
    print("  FGW prototype-graph Domain Adaptation  (Yelp)")
    print("=" * 60)
    print(f"  Sources       : {cfg.source_domains}")
    print(f"  Target        : {cfg.target_domain}")
    _draw = "stratified" if cfg.source_label_stratified else "random"
    print(f"  Source labels : {cfg.source_label_ratio:.0%} ({_draw} draw; "
          f"target 0%)")
    print(f"  Selection     : {cfg.model_selection} "
          f"({cfg.source_val_frac:.0%} of source labels held out)")
    print(f"  Device        : {cfg.device}")
    print(f"  feature_dim   : {cfg.feature_dim}")
    print(f"  num_classes   : {cfg.num_classes}")
    print(f"  proj/hidden   : {cfg.proj_dim} / {cfg.hidden_dim}"
          f"{'  (frozen SVD basis)' if cfg.svd_proj else '  (learned proj)'}")
    print(f"  schedule      : warmup <= {cfg.warmup_epochs} (patience "
          f"{cfg.warmup_patience}) -> adapt {cfg.adapt_epochs} -> refine, "
          f"of {cfg.epochs} epochs")
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

    runs = [run_once(cfg, sources, target, sd) for sd in seeds]

    print("\n" + "=" * 60)
    print(f"  Final results — {cfg.target_domain}, {len(runs)} run(s)")
    print("=" * 60)
    base = majority_baseline(target.y, cfg.num_classes)
    print(f"  majority class : ACC {base['acc']:.4f}  AUROC {base['auc']:.4f}  "
          f"MacroF {base['f1']:.4f}   (no-skill reference)")
    for key, label in (("acc", "ACC"), ("auc", "AUROC"), ("f1", "MacroF")):
        vals = [r[key] for r in runs]
        if len(vals) > 1:
            print(f"  {label:<14}: {statistics.mean(vals):.4f} "
                  f"+- {statistics.stdev(vals):.4f}   "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")
        else:
            print(f"  {label:<14}: {vals[0]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
