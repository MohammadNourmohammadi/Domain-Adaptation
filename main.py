"""
Causal-Masked Adversarial Domain Adaptation on Twitch (multi-source).

N source domains (labeled) + 1 target domain (unlabeled): the encoder is
trained to predict labels on every source while a domain classifier with
gradient reversal makes the embeddings indistinguishable across all N+1
domains. A learned per-edge causal mask filters spurious edges.

Usage:
    python main.py                                           # DE,FR -> ENGB
    python main.py --sources DE ES --target ENGB
    python main.py --sources ENGB ES RU --target FR --epochs 200
"""

import argparse

import torch

from src.config import Config
from src.data_loader import GLOBAL_FEATURE_DIM, load_sources_target
from src.models import CausalGNN_DANN
from src.train import evaluate, run_training
from src.utils import set_seed


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Causal-DANN on Twitch")
    parser.add_argument(
        "--sources", type=str, nargs="+", default=["DE", "FR"],
        help="One or more source domains (space-separated)",
    )
    parser.add_argument("--target", type=str, default="ENGB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lambda_domain", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--lambda_counter", type=float, default=0.5)
    parser.add_argument("--lambda_vrex", type=float, default=1.0)
    parser.add_argument("--counter_margin", type=float, default=1.0)
    parser.add_argument("--drop_edge_p", type=float, default=0.15)
    parser.add_argument("--grl_warmup_epochs", type=int, default=20)
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

    return Config(
        source_domains=args.sources,
        target_domain=args.target,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        lambda_domain=args.lambda_domain,
        lambda_sparse=args.lambda_sparse,
        lambda_counter=args.lambda_counter,
        lambda_vrex=args.lambda_vrex,
        counter_margin=args.counter_margin,
        drop_edge_p=args.drop_edge_p,
        grl_warmup_epochs=args.grl_warmup_epochs,
        seed=args.seed,
        device=device,
    )


def main():
    config = parse_args()
    set_seed(config.seed)

    print("=" * 60)
    print("  Causal-Masked Adversarial Domain Adaptation")
    print("=" * 60)
    print(f"  Sources      : {config.source_domains}")
    print(f"  Target       : {config.target_domain}")
    print(f"  Device       : {config.device}")
    print(f"  proj/hidden  : {config.proj_dim} / {config.hidden_dim}")
    print(f"  weight_decay : {config.weight_decay}")
    print(f"  lambda_dom   : {config.lambda_domain}")
    print(f"  lambda_sparse: {config.lambda_sparse}  (binary-entropy mask)")
    print(f"  lambda_count : {config.lambda_counter}  (counterfactual, margin {config.counter_margin})")
    print(f"  lambda_vrex  : {config.lambda_vrex}")
    print(f"  drop_edge_p  : {config.drop_edge_p}")
    print("=" * 60)

    print("\nLoading data ...")
    sources, target = load_sources_target(
        config.data_root, config.source_domains, config.target_domain,
    )
    for name, g in zip(config.source_domains, sources):
        print(f"  {name}: {g.num_nodes} nodes, {g.edge_index.size(1)} edges, "
              f"pos-rate {g.y.float().mean().item():.3f}")
    print(f"  {config.target_domain} (target): {target.num_nodes} nodes, "
          f"{target.edge_index.size(1)} edges, "
          f"pos-rate {target.y.float().mean().item():.3f}")

    num_domains = len(sources) + 1  # one class per source + 1 for target
    model = CausalGNN_DANN(
        in_dim=GLOBAL_FEATURE_DIM,
        proj_dim=config.proj_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_domains=num_domains,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}  "
          f"(domain classifier out: {num_domains})")

    print("\nTraining ...\n")
    model = run_training(model, sources, target, config)

    sources = [s.to(config.device) for s in sources]
    target = target.to(config.device)

    print("\n" + "=" * 60)
    print("  Final results")
    print("=" * 60)
    for name, g in zip(config.source_domains, sources):
        s = evaluate(model, g)
        print(f"  Source {name:>5}: acc {s['acc']:.4f}  "
              f"f1 {s['f1']:.4f}  auc {s['auc']:.4f}")
    tgt_stats = evaluate(model, target)
    print(f"  Target {config.target_domain:>5}: acc {tgt_stats['acc']:.4f}  "
          f"f1 {tgt_stats['f1']:.4f}  auc {tgt_stats['auc']:.4f}")
    print(f"  Mean causal edge weight (target): "
          f"{tgt_stats['avg_edge_weight']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
