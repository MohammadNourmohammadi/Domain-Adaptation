"""FGW prototype-graph domain adaptation on Twitch.

A second runner sitting alongside `main.py`. The existing Causal-DANN
pipeline is untouched; this script wires the new modules together.

Usage:
    python main_fgw.py                                       # DE,FR -> ENGB
    python main_fgw.py --sources DE ES --target ENGB
    python main_fgw.py --sources ENGB ES RU --target FR --epochs 200
"""

import argparse

import torch

from src.data_loader import GLOBAL_FEATURE_DIM, load_sources_target
from src.fgw_config import FGWConfig
from src.fgw_model import FGWPrototypeDA
from src.fgw_train import evaluate, run_training, _make_cache
from src.utils import set_seed


def parse_args() -> FGWConfig:
    parser = argparse.ArgumentParser(description="FGW-prototype DA on Twitch")
    parser.add_argument("--sources", type=str, nargs="+", default=["DE", "FR"])
    parser.add_argument("--target", type=str, default="ENGB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--ego_size", type=int, default=32)
    parser.add_argument("--proto_size", type=int, default=32)
    parser.add_argument("--num_protos", type=int, default=3)
    parser.add_argument("--fgw_alpha", type=float, default=0.5)
    parser.add_argument("--fgw_epsilon", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--lambda_ent", type=float, default=0.5)
    parser.add_argument("--lambda_sep", type=float, default=0.1)
    parser.add_argument("--lambda_pl", type=float, default=0.1)
    parser.add_argument("--lambda_vrex", type=float, default=1.0)
    parser.add_argument("--lambda_struct", type=float, default=1e-3)
    parser.add_argument("--nodes_per_step", type=int, default=128)
    parser.add_argument("--warmup_frac", type=float, default=0.2)
    parser.add_argument("--refine_frac", type=float, default=0.6)
    parser.add_argument("--ramp_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    return FGWConfig(
        source_domains=args.sources,
        target_domain=args.target,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        ego_size=args.ego_size,
        proto_size=args.proto_size,
        num_protos=args.num_protos,
        fgw_alpha=args.fgw_alpha,
        fgw_epsilon=args.fgw_epsilon,
        tau=args.tau,
        lambda_align=args.lambda_align,
        lambda_ent=args.lambda_ent,
        lambda_sep=args.lambda_sep,
        lambda_pl=args.lambda_pl,
        lambda_vrex=args.lambda_vrex,
        lambda_struct=args.lambda_struct,
        nodes_per_step=args.nodes_per_step,
        warmup_frac=args.warmup_frac,
        refine_frac=args.refine_frac,
        ramp_epochs=args.ramp_epochs,
        seed=args.seed,
        device=device,
    )


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    print("=" * 60)
    print("  FGW prototype-graph Domain Adaptation")
    print("=" * 60)
    print(f"  Sources       : {cfg.source_domains}")
    print(f"  Target        : {cfg.target_domain}")
    print(f"  Device        : {cfg.device}")
    print(f"  proj/hidden   : {cfg.proj_dim} / {cfg.hidden_dim}")
    print(f"  ego_size k    : {cfg.ego_size}")
    print(f"  proto_size n_p: {cfg.proto_size}  (M={cfg.num_protos} per class)")
    print(f"  fgw alpha,eps : {cfg.fgw_alpha}, {cfg.fgw_epsilon}")
    print(f"  tau           : {cfg.tau}")
    print(f"  lambda_align  : {cfg.lambda_align}")
    print(f"  lambda_ent    : {cfg.lambda_ent}")
    print(f"  lambda_sep    : {cfg.lambda_sep}")
    print(f"  lambda_pl     : {cfg.lambda_pl}")
    print(f"  lambda_vrex   : {cfg.lambda_vrex}")
    print(f"  lambda_struct : {cfg.lambda_struct}")
    print("=" * 60)

    print("\nLoading data ...")
    sources, target = load_sources_target(
        cfg.data_root, cfg.source_domains, cfg.target_domain,
    )
    for name, g in zip(cfg.source_domains, sources):
        print(f"  {name}: {g.num_nodes} nodes, {g.edge_index.size(1)} edges, "
              f"pos-rate {g.y.float().mean().item():.3f}")
    print(f"  {cfg.target_domain} (target): {target.num_nodes} nodes, "
          f"{target.edge_index.size(1)} edges, "
          f"pos-rate {target.y.float().mean().item():.3f}")

    model = FGWPrototypeDA(
        in_dim=GLOBAL_FEATURE_DIM,
        proj_dim=cfg.proj_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        num_protos=cfg.num_protos,
        proto_size=cfg.proto_size,
        anchor_weight=cfg.anchor_weight,
        adjacency_temp=cfg.adjacency_temp,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    print("\nTraining ...\n")
    model = run_training(model, sources, target, cfg)

    sources = [s.to(cfg.device) for s in sources]
    target = target.to(cfg.device)

    # Rebuild caches on the trained model's device so we can evaluate
    # every domain with the same machinery used during training.
    src_caches = [_make_cache(s, cfg, cfg.device) for s in sources]
    tgt_cache = _make_cache(target, cfg, cfg.device)

    print("\n" + "=" * 60)
    print("  Final results")
    print("=" * 60)
    for name, g, cache in zip(cfg.source_domains, sources, src_caches):
        s = evaluate(model, g, cache, cfg)
        print(f"  Source {name:>5}: acc {s['acc']:.4f}  "
              f"f1 {s['f1']:.4f}  auc {s['auc']:.4f}")
    tgt_stats = evaluate(model, target, tgt_cache, cfg)
    print(f"  Target {cfg.target_domain:>5}: acc {tgt_stats['acc']:.4f}  "
          f"f1 {tgt_stats['f1']:.4f}  auc {tgt_stats['auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
