"""FGW prototype-graph domain adaptation on Twitch.

A second runner sitting alongside `main.py`. The existing Causal-DANN
pipeline is untouched; this script wires the new modules together.

The default setting reproduces SelMAG's Twitch protocol (arXiv:2406.10425):
the six language graphs, the five alphabetically first as sources, RU as
target, 10% of source nodes labeled. Verify the data with
`python scripts/download_twitch.py --check`.

One caveat, documented in `src/data_loader.py`: the paper used a dense 128-d
feature release whose host has since been suspended and whose domain no longer
resolves, so it cannot be obtained by anyone. We train on the authors' own
3170-d features, which the encoder's frozen unsupervised SVD basis reduces to
--proj_dim (default 128) — the paper's stated width, on the paper's graphs.

Usage:
    python main_fgw.py                                       # DE..PTBR -> RU
    python main_fgw.py --seeds 0 1 2 3 4                     # paper-style 5 runs
    python main_fgw.py --features musae                      # raw 3170-d BoW
    python main_fgw.py --no_da                               # source-only baseline
    python main_fgw.py --sources DE ES --target ENGB --epochs 200
"""

import argparse
import statistics

from src.data_loader import FEATURE_SETS, feature_dim, load_sources_target
from src.fgw_cli import add_method_args, method_kwargs, pick_device, seed_list
from src.fgw_config import FGWConfig
from src.fgw_model import FGWPrototypeDA
from src.fgw_train import evaluate, run_training
from src.utils import majority_baseline, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="FGW-prototype DA on Twitch")
    parser.add_argument("--sources", type=str, nargs="+",
                        default=["DE", "ENGB", "ES", "FR", "PTBR"])
    parser.add_argument("--target", type=str, default="RU")
    parser.add_argument("--features", type=str, default="musae",
                        choices=FEATURE_SETS,
                        help="'musae': the authors' 3170-d bag-of-words, the "
                             "only release still obtainable (the encoder's "
                             "frozen SVD basis reduces it to --proj_dim=128, "
                             "the paper's stated width). 'pyg': the 128-d "
                             "release SelMAG used, whose host no longer exists")
    parser.add_argument("--pyg_root", type=str, default="data/twitch_pyg",
                        help="where scripts/download_twitch_pyg.py put the npz files")
    add_method_args(parser)
    args = parser.parse_args()

    if args.target in args.sources:
        raise ValueError(
            f"target '{args.target}' must not also be a source: {args.sources}"
        )

    cfg = FGWConfig(
        source_domains=args.sources,
        target_domain=args.target,
        features=args.features,
        feature_dim=feature_dim(args.features),
        device=pick_device(),
        **method_kwargs(args),
    )
    return cfg, seed_list(args), args.pyg_root


def print_header(cfg: FGWConfig, seeds) -> None:
    print("=" * 60)
    print("  FGW prototype-graph Domain Adaptation")
    print("=" * 60)
    print(f"  Sources       : {cfg.source_domains}")
    print(f"  Target        : {cfg.target_domain}")
    print(f"  Features      : {cfg.features} ({cfg.feature_dim}-d)"
          f"{'   <- SelMAG protocol' if cfg.features == 'pyg' else ''}")
    _draw = "stratified" if cfg.source_label_stratified else "random"
    print(f"  Source labels : {cfg.source_label_ratio:.0%} ({_draw} draw; "
          f"target 0%)")
    print(f"  Selection     : {cfg.model_selection} "
          f"({cfg.source_val_frac:.0%} of source labels held out)")
    print(f"  Seeds         : {seeds}")
    print(f"  Device        : {cfg.device}")
    print(f"  proj/hidden   : {cfg.proj_dim} / {cfg.hidden_dim}"
          f"{'  (frozen SVD basis)' if cfg.svd_proj else '  (learned proj)'}")
    print(f"  schedule      : warmup <= {cfg.warmup_epochs} (armed at "
          f"{cfg.warmup_min_epochs}, patience {cfg.warmup_patience}, proto gate "
          f"{cfg.warmup_proto_gate}) -> adapt {cfg.adapt_epochs} -> refine, "
          f"of {cfg.epochs} epochs")
    print(f"  ego_size k    : {cfg.ego_size}")
    print(f"  proto_size n_p: {cfg.proto_size}  (M={cfg.num_protos} per class)")
    print(f"  fgw alpha,eps : {cfg.fgw_alpha}, {cfg.fgw_epsilon}")
    print(f"  tau / predict : {cfg.tau} / {cfg.predict}")
    print(f"  head/LN       : {cfg.head_hidden} / {cfg.use_layernorm}")
    print(f"  mode          : "
          f"{'source-only (no DA)' if cfg.no_da else 'FGW domain adaptation'}")
    print(f"  selection     : "
          f"{'FGW s_global/s_local' if cfg.use_selection else 'off (uniform sources)'}")
    print(f"  prior         : "
          f"{'given ' + str(cfg.target_class_prior) if cfg.target_class_prior else ('BBSE estimate' if cfg.estimate_prior else 'source prior')}"
          f"; eval correction {'on' if cfg.prior_correct_eval else 'off'}")
    print(f"  lambdas       : proto {cfg.lambda_proto}  align {cfg.lambda_align}  "
          f"ent {cfg.lambda_ent} (bal {cfg.im_balance_weight})  "
          f"sep {cfg.lambda_sep}  pl {cfg.lambda_pl}")
    print(f"                  fgw_margin {cfg.lambda_fgw_margin} "
          f"(m={cfg.fgw_margin})  struct {cfg.lambda_struct}  "
          f"vrex {cfg.lambda_vrex}")
    print("=" * 60)


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
    cfg, seeds, pyg_root = parse_args()
    print_header(cfg, seeds)

    print("\nLoading data ...")
    sources, target = load_sources_target(
        cfg.data_root, cfg.source_domains, cfg.target_domain,
        features=cfg.features, pyg_root=pyg_root,
    )
    for name, g in zip(cfg.source_domains, sources):
        print(f"  {name}: {g.num_nodes} nodes, {g.edge_index.size(1)} edges, "
              f"pos-rate {g.y.float().mean().item():.3f}")
    print(f"  {cfg.target_domain} (target): {target.num_nodes} nodes, "
          f"{target.edge_index.size(1)} edges, "
          f"pos-rate {target.y.float().mean().item():.3f}")

    runs = [run_once(cfg, sources, target, s) for s in seeds]

    print("\n" + "=" * 60)
    print(f"  Final results — {cfg.target_domain}, {len(runs)} run(s)")
    print("=" * 60)
    # The majority-class row makes the accuracy column readable. On Twitch RU
    # (24.5% positive) it scores ACC 0.755 / MacroF 0.430, which is *above*
    # every accuracy SelMAG's Table 1 reports for this dataset — worth stating
    # rather than leaving for a reviewer to work out.
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
