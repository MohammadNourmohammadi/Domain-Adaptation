"""FGW prototype-graph domain adaptation on Cora_full.

Runs the same method as `main_fgw.py` / `main_citation_fgw.py` /
`main_yelp_fgw.py` (shared encoder + parametric head + FGW prototype transfer)
on the six graphs SelMAG (arXiv:2406.10425) derives from the full Cora citation
network by grouping papers on word diversity. Multi-source: train on every
group except one, adapt to the held-out target.

`cora.npz` (12.3 MB) is downloaded on first use into `--data_root` and the six
graphs are derived from it; later runs reuse the file. See
`src/cora_full_loader.py` for how the split reproduces GOOD's `word` domain.

Usage:
    python main_cora_full_fgw.py                       # W0..W4 -> W5
    python main_cora_full_fgw.py --target W0           # sources auto-fill
    python main_cora_full_fgw.py --show_split          # verify vs Table 3
    python main_cora_full_fgw.py --seeds 1 2 3 4 5     # paper-style 5 runs
    python main_cora_full_fgw.py --no_da               # source-only baseline
"""

import argparse
import os

import torch

from src.cora_full_loader import (
    CORA_FULL_DOMAINS,
    SPLIT_MODES,
    CORA_FULL_FEATURE_DIM,
    CORA_FULL_NUM_CLASSES,
    CORA_FULL_TARGET,
    load_sources_target,
    split_summary,
)
from src.fgw_cli import add_method_args, method_kwargs, pick_device, seed_list
from src.fgw_config import FGWConfig
from src.fgw_model import FGWPrototypeDA
from src.fgw_train import run_training, summarize_run
from src.run_artifacts import Tee, allocate_run_dir, report_and_save
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="FGW-prototype DA on Cora_full")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="source groups; default = all groups but --target")
    parser.add_argument("--target", type=str, default=CORA_FULL_TARGET,
                        choices=CORA_FULL_DOMAINS,
                        help="held-out group (default: the lowest word-diversity "
                             "one, which is GOOD's out-of-distribution end)")
    parser.add_argument("--data_root", type=str, default="data/cora_full")
    parser.add_argument("--directed", action="store_true",
                        help="keep citation edges directed (default: symmetrise)")
    parser.add_argument("--no_download", action="store_true",
                        help="require cora.npz to be present already")
    parser.add_argument("--split_by", type=str, default="word", choices=SPLIT_MODES,
                        help="how the six groups are cut. 'word' is GOOD's word "
                             "domain (x.sum(1)), which is what the paper cites; "
                             "'nnz' the literal count of words used; 'degree' "
                             "GOOD's other domain; 'random' a structure-blind "
                             "control. None reproduces Table 3's edge count — "
                             "see --show_split")
    parser.add_argument("--show_split", action="store_true",
                        help="print the six graphs' sizes against SelMAG's "
                             "Table 3 averages and exit")
    # Cora_full is 70-class with an 8710-dim bag-of-words: a wider encoder than
    # Twitch, same as the Citation setting. Everything else is the shared
    # method surface (see src/fgw_cli.py).
    #
    # Three defaults differ from the dense datasets, all of them consequences of
    # 70 classes on 3,299-node induced subgraphs:
    #   * `source_label_stratified` — a random 10% draw over 70 long-tailed
    #     classes leaves 4-13 of them with no labeled node at all, and macro-F1
    #     counts every one of them.
    #   * `class_weighted_ce` — plain CE optimises accuracy; the reported metric
    #     is macro-F1 and the class sizes span 15-928 papers.
    #   * `val_metric=auto` (shared default) falls back to accuracy here, since
    #     330 pooled held-out nodes cannot estimate 70 per-class F1 scores.
    # `--no-source_label_stratified` / `--no-class_weighted_ce` restore the
    # dense-dataset behaviour for ablations.
    add_method_args(parser, hidden_dim=64, fgw_alpha=0.5, ego_size=16,
                    source_label_stratified=True, class_weighted_ce=True)
    args = parser.parse_args()

    if args.sources is None:
        args.sources = [d for d in CORA_FULL_DOMAINS if d != args.target]

    unknown = [d for d in args.sources + [args.target] if d not in CORA_FULL_DOMAINS]
    if unknown:
        raise ValueError(f"unknown Cora_full group(s) {unknown}; "
                         f"choose from {CORA_FULL_DOMAINS}")
    if args.target in args.sources:
        raise ValueError(
            f"target '{args.target}' must not also be a source: {args.sources}"
        )

    cfg = FGWConfig(
        data_root=args.data_root,
        source_domains=args.sources,
        target_domain=args.target,
        feature_dim=CORA_FULL_FEATURE_DIM,
        num_classes=CORA_FULL_NUM_CLASSES,
        device=pick_device(),
        **method_kwargs(args),
    )
    cfg.symmetrize = not args.directed
    cfg.auto_download = not args.no_download
    cfg.show_split = args.show_split
    cfg.split_by = args.split_by
    return cfg, seed_list(args)


def _label_hist(y: torch.Tensor, num_classes: int) -> str:
    """Compact label summary. 70 classes is too many to print one by one."""
    counts = torch.bincount(y, minlength=num_classes)
    present = int((counts > 0).sum())
    top = torch.topk(counts, k=min(5, num_classes))
    top_txt = " ".join(f"{int(c)}:{int(n)}" for c, n in zip(top.indices, top.values))
    return f"{present}/{num_classes} classes present, top5 {top_txt}"


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
    return summarize_run(model, sources, target, cfg, ctx, seed,
                         name_width=5)


def _main(cfg, seeds, run_dir):
    set_seed(cfg.seed)

    if cfg.show_split:
        print("Cora_full six-graph split\n")
        print(split_summary(cfg.data_root, symmetrize=cfg.symmetrize))
        return

    print("=" * 60)
    print("  FGW prototype-graph Domain Adaptation  (Cora_full)")
    print("=" * 60)
    print(f"  Sources       : {cfg.source_domains}")
    print(f"  Target        : {cfg.target_domain}")
    _draw = "stratified" if cfg.source_label_stratified else "random"
    print(f"  Source labels : {cfg.source_label_ratio:.0%} ({_draw} draw; "
          f"target 0%)")
    print(f"  Selection     : {cfg.model_selection} "
          f"({cfg.source_val_frac:.0%} of source labels held out)")
    print(f"  Seeds         : {seeds}")
    print(f"  Device        : {cfg.device}")
    print(f"  feature_dim   : {cfg.feature_dim}")
    print(f"  num_classes   : {cfg.num_classes}")
    print(f"  proj/hidden   : {cfg.proj_dim} / {cfg.hidden_dim}"
          f"{'  (frozen SVD basis)' if cfg.svd_proj else '  (learned proj)'}")
    print(f"  schedule      : warmup <= {cfg.warmup_epochs} (armed at "
          f"{cfg.warmup_min_epochs}, patience {cfg.warmup_patience}, proto gate "
          f"{cfg.warmup_proto_gate}) -> adapt {cfg.adapt_epochs} -> refine, "
          f"of {cfg.epochs} epochs")
    print(f"  ego_size k    : {cfg.ego_size}"
          f"{'' if not cfg.knn_augment else f'  (+{cfg.knn_augment} kNN edges below degree {cfg.knn_min_degree})'}")
    print(f"  class weights : "
          f"{f'inverse-freq^{cfg.class_weight_power}' if cfg.class_weighted_ce else 'off'}"
          f"   val metric: {cfg.val_metric}")
    print(f"  proto_size n_p: {cfg.proto_size}  (M={cfg.num_protos} per class)")
    print(f"  fgw alpha,eps : {cfg.fgw_alpha}, {cfg.fgw_epsilon}")
    print(f"  head/LN       : {cfg.head_hidden} / {cfg.use_layernorm}")
    print(f"  mode          : "
          f"{'source-only (no DA)' if cfg.no_da else 'FGW domain adaptation'}")
    print(f"  selection     : "
          f"{'FGW s_global/s_local' if cfg.use_selection else 'off (uniform sources)'}")
    print(f"  predict       : {cfg.predict}")
    print(f"  split_by      : {cfg.split_by}")
    print("=" * 60)

    print("\nLoading data ...")
    sources, target = load_sources_target(
        cfg.data_root, cfg.source_domains, cfg.target_domain,
        symmetrize=getattr(cfg, "symmetrize", True),
        auto_download=getattr(cfg, "auto_download", True),
        split_by=getattr(cfg, "split_by", "word"),
    )

    # Derive the feature/class dims from the data itself and keep the config
    # honest. Cora_full's rarest classes have only a handful of papers, so an
    # individual group can easily miss some — always size the head from the
    # global class count, not from what happens to appear here.
    feat_dims = {g.x.size(1) for g in sources + [target]}
    if len(feat_dims) != 1:
        raise ValueError(f"inconsistent feature dims across groups: {feat_dims}")
    cfg.feature_dim = feat_dims.pop()
    cfg.num_classes = max(
        CORA_FULL_NUM_CLASSES,
        int(max(int(g.y.max()) for g in sources + [target]) + 1),
    )

    for name, g in zip(cfg.source_domains, sources):
        lo, hi = g.score_range
        print(f"  {name}: {g.num_nodes} nodes, {g.edge_index.size(1)} edges, "
              f"{cfg.split_by} {lo:.3g}-{hi:.3g}, "
              f"{_label_hist(g.y, cfg.num_classes)}")
    lo, hi = target.score_range
    print(f"  {cfg.target_domain} (target): {target.num_nodes} nodes, "
          f"{target.edge_index.size(1)} edges, {cfg.split_by} {lo:.3g}-{hi:.3g}, "
          f"{_label_hist(target.y, cfg.num_classes)}")

    runs = [run_once(cfg, sources, target, s) for s in seeds]

    report_and_save(
        cfg, seeds, runs, target, dataset="cora_full",
        root=cfg.out_dir, run_dir=run_dir,
        make_figures=not cfg.no_figures,
    )


def main():
    """Allocate this run's folder, then mirror everything printed into it."""
    cfg, seeds = parse_args()
    if getattr(cfg, "show_split", False) or cfg.no_save:
        return _main(cfg, seeds, None)
    run_dir = allocate_run_dir(cfg.out_dir, "cora_full", cfg.target_domain)
    with Tee(os.path.join(run_dir, "log.txt")):
        return _main(cfg, seeds, run_dir)


if __name__ == "__main__":
    main()
