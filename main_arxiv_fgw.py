"""FGW prototype-graph domain adaptation on Arxiv (ogbn-arxiv).

Runs the same method as `main_fgw.py` / `main_citation_fgw.py` /
`main_yelp_fgw.py` / `main_cora_full_fgw.py` (shared encoder + parametric head +
FGW prototype transfer) on the six graphs SelMAG (arXiv:2406.10425) derives from
the OGB arXiv citation network by publication year. Multi-source: train on every
band except one, adapt to the held-out target.

`arxiv.zip` (83 MB) is downloaded on first use into `--data_root`, unpacked, and
its raw CSVs parsed once into a `.npy` cache; later runs reuse the cache. See
`src/arxiv_loader.py` for where the year cuts fall and why the default is not
the literal equal-size split.

Usage:
    python main_arxiv_fgw.py                        # Y0..Y4 -> Y5 (2019-2020)
    python main_arxiv_fgw.py --target Y0            # sources auto-fill
    python main_arxiv_fgw.py --show_split           # verify vs Table 3
    python main_arxiv_fgw.py --split_by year        # literal equal-size cut
    python main_arxiv_fgw.py --year_cuts 2011 2014 2016 2017 2018
    python main_arxiv_fgw.py --seeds 1 2 3 4 5      # paper-style 5 runs
    python main_arxiv_fgw.py --no_da                # source-only baseline
"""

import argparse
import statistics

import torch

from src.arxiv_loader import (
    ARXIV_DOMAINS,
    ARXIV_FEATURE_DIM,
    ARXIV_NUM_CLASSES,
    ARXIV_TARGET,
    SPLIT_MODES,
    load_sources_target,
    split_summary,
)
from src.fgw_cli import add_method_args, method_kwargs, pick_device, seed_list
from src.fgw_config import FGWConfig
from src.fgw_model import FGWPrototypeDA
from src.fgw_train import evaluate, run_training
from src.utils import majority_baseline, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="FGW-prototype DA on Arxiv")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="source bands; default = all bands but --target")
    parser.add_argument("--target", type=str, default=ARXIV_TARGET,
                        choices=ARXIV_DOMAINS,
                        help="held-out band (default: the most recent one, "
                             "which is what the paper adapts to)")
    parser.add_argument("--data_root", type=str, default="data/arxiv")
    parser.add_argument("--directed", action="store_true",
                        help="keep citation edges directed (default: symmetrise)")
    parser.add_argument("--no_download", action="store_true",
                        help="require the ogbn-arxiv dump to be present already")
    parser.add_argument("--split_by", type=str, default="year_tail",
                        choices=SPLIT_MODES,
                        help="how the six bands are cut. 'year_tail' (default) "
                             "keeps 2019-2020 together as the target; 'year' is "
                             "the literal equal-size cut, whose 2020-only target "
                             "is 81%% isolated nodes; 'degree' is GOOD's other "
                             "arXiv domain and 'random' a structure-blind "
                             "control. None reproduces Table 3's edge count — "
                             "see --show_split")
    parser.add_argument("--year_cuts", type=int, nargs=5, default=None,
                        metavar="Y",
                        help="explicit inclusive upper year of each of the first "
                             "five bands, e.g. --year_cuts 2011 2014 2016 2017 "
                             "2018; overrides --split_by")
    parser.add_argument("--show_split", action="store_true",
                        help="print the six bands' sizes against SelMAG's "
                             "Table 3 averages and exit")
    # 128-dim dense word2vec features and 40 classes: a wider encoder than
    # Twitch, same as the Cora_full setting. The bands are sparse (average
    # degree 1.5-4.4), so ego-graphs stay small however large k is. Everything
    # else is the shared method surface (see src/fgw_cli.py).
    add_method_args(parser, hidden_dim=64, fgw_alpha=0.5, ego_size=12)
    args = parser.parse_args()

    if args.sources is None:
        args.sources = [d for d in ARXIV_DOMAINS if d != args.target]

    unknown = [d for d in args.sources + [args.target] if d not in ARXIV_DOMAINS]
    if unknown:
        raise ValueError(f"unknown arXiv band(s) {unknown}; "
                         f"choose from {ARXIV_DOMAINS}")
    if args.target in args.sources:
        raise ValueError(
            f"target '{args.target}' must not also be a source: {args.sources}"
        )

    cfg = FGWConfig(
        data_root=args.data_root,
        source_domains=args.sources,
        target_domain=args.target,
        feature_dim=ARXIV_FEATURE_DIM,
        num_classes=ARXIV_NUM_CLASSES,
        device=pick_device(),
        **method_kwargs(args),
    )
    cfg.symmetrize = not args.directed
    cfg.auto_download = not args.no_download
    cfg.show_split = args.show_split
    cfg.split_by = args.split_by
    cfg.year_cuts = args.year_cuts
    return cfg, seed_list(args)


def _label_hist(y: torch.Tensor, num_classes: int) -> str:
    """Compact label summary. 40 classes is too many to print one by one."""
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

    dev_sources = [s.to(cfg.device) for s in sources]
    dev_target = target.to(cfg.device)

    print("\n" + "-" * 60)
    print(f"  Seed {seed} results")
    print("-" * 60)
    for name, g, cache in zip(cfg.source_domains, dev_sources, ctx["src_caches"]):
        s = evaluate(model, g, cfg=cfg, cache=cache)
        print(f"  Source {name:>5}: ACC {s['acc']:.4f}  "
              f"AUROC {s['auc']:.4f}  MacroF {s['f1']:.4f}")
    tgt = evaluate(
        model, dev_target, cfg=cfg, cache=ctx["tgt_cache"],
        source_prior=ctx["src_prior"], target_prior=ctx["tgt_prior"],
    )
    print(f"  Target {cfg.target_domain:>5}: ACC {tgt['acc']:.4f}  "
          f"AUROC {tgt['auc']:.4f}  MacroF {tgt['f1']:.4f}")
    return tgt


def main():
    cfg, seeds = parse_args()
    set_seed(cfg.seed)

    if cfg.show_split:
        print("Arxiv six-graph split\n")
        print(split_summary(cfg.data_root, symmetrize=cfg.symmetrize,
                            year_cuts=cfg.year_cuts))
        return

    print("=" * 60)
    print("  FGW prototype-graph Domain Adaptation  (Arxiv)")
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
    print(f"  ego_size k    : {cfg.ego_size}")
    print(f"  proto_size n_p: {cfg.proto_size}  (M={cfg.num_protos} per class)")
    print(f"  fgw alpha,eps : {cfg.fgw_alpha}, {cfg.fgw_epsilon}")
    print(f"  head/LN       : {cfg.head_hidden} / {cfg.use_layernorm}")
    print(f"  mode          : "
          f"{'source-only (no DA)' if cfg.no_da else 'FGW domain adaptation'}")
    print(f"  selection     : "
          f"{'FGW s_global/s_local' if cfg.use_selection else 'off (uniform sources)'}")
    print(f"  predict       : {cfg.predict}")
    print(f"  split_by      : {cfg.split_by}"
          f"{'' if cfg.year_cuts is None else f'  cuts={cfg.year_cuts}'}")
    print("=" * 60)

    print("\nLoading data ...")
    sources, target = load_sources_target(
        cfg.data_root, cfg.source_domains, cfg.target_domain,
        symmetrize=getattr(cfg, "symmetrize", True),
        auto_download=getattr(cfg, "auto_download", True),
        split_by=getattr(cfg, "split_by", "year_tail"),
        year_cuts=getattr(cfg, "year_cuts", None),
    )

    # Derive the feature/class dims from the data itself and keep the config
    # honest. A single band can miss one of the 40 subject areas, so always
    # size the head from the global class count, not from what appears here.
    feat_dims = {g.x.size(1) for g in sources + [target]}
    if len(feat_dims) != 1:
        raise ValueError(f"inconsistent feature dims across bands: {feat_dims}")
    cfg.feature_dim = feat_dims.pop()
    cfg.num_classes = max(
        ARXIV_NUM_CLASSES,
        int(max(int(g.y.max()) for g in sources + [target]) + 1),
    )

    def _describe(name, g, tag=""):
        lo, hi = g.year_range
        deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes)
        iso = 100.0 * float((deg == 0).sum()) / g.num_nodes
        print(f"  {name}{tag}: {g.num_nodes:,} nodes, {g.edge_index.size(1):,} "
              f"edges, years {lo}-{hi}, {iso:.1f}% isolated, "
              f"{_label_hist(g.y, cfg.num_classes)}")

    for name, g in zip(cfg.source_domains, sources):
        _describe(name, g)
    _describe(cfg.target_domain, target, " (target)")

    runs = [run_once(cfg, sources, target, s) for s in seeds]

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
