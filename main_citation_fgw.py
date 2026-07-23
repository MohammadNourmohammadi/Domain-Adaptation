"""FGW prototype-graph domain adaptation on the Citation networks.

Runs the same method as `main_fgw.py` (shared encoder + parametric head +
FGW prototype transfer) but on the three ArnetMiner citation graphs from
pygda — ACMv9, Citationv1, DBLPv7 — instead of Twitch. Multi-source:
train on every domain except one, adapt to the held-out target graph.

The raw files are downloaded from the pygda Google Drive folder on first
use and cached under `--data_root`; later runs just load the cache.

Usage:
    python main_citation_fgw.py                                  # -> DBLPv7 target
    python main_citation_fgw.py --target ACMv9                   # sources auto-fill
    python main_citation_fgw.py --sources ACMv9 Citationv1 --target DBLPv7
    python main_citation_fgw.py --target ACMv9 --epochs 200
"""

import argparse

import torch

from src.citation_loader import (
    CITATION_DOMAINS,
    CITATION_FEATURE_DIM,
    CITATION_NUM_CLASSES,
    load_sources_target,
)
from src.fgw_config import FGWConfig
from src.fgw_model import FGWPrototypeDA
from src.fgw_train import evaluate, run_training
from src.utils import set_seed


def parse_args() -> FGWConfig:
    parser = argparse.ArgumentParser(description="FGW-prototype DA on Citation")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="source domains; default = all citation domains "
                             "except --target")
    parser.add_argument("--target", type=str, default="DBLPv7",
                        choices=CITATION_DOMAINS)
    parser.add_argument("--data_root", type=str, default="data/citation")
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
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--lambda_ent", type=float, default=0.5)
    parser.add_argument("--lambda_sep", type=float, default=1.0)
    parser.add_argument("--lambda_pl", type=float, default=0.1)
    parser.add_argument("--lambda_vrex", type=float, default=1.0)
    parser.add_argument("--lambda_struct", type=float, default=1e-3)
    parser.add_argument("--sep_intra_margin", type=float, default=0.5)
    parser.add_argument("--pl_threshold", type=float, default=0.8)
    parser.add_argument("--nodes_per_step", type=int, default=128)
    parser.add_argument("--warmup_frac", type=float, default=0.2)
    parser.add_argument("--refine_frac", type=float, default=0.6)
    parser.add_argument("--ramp_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--directed", action="store_true",
                        help="keep citation edges directed (default: symmetrise)")
    args = parser.parse_args()

    # "Every source except one": default sources = all domains but the target.
    if args.sources is None:
        args.sources = [d for d in CITATION_DOMAINS if d != args.target]

    unknown = [d for d in args.sources + [args.target] if d not in CITATION_DOMAINS]
    if unknown:
        raise ValueError(f"unknown citation domain(s) {unknown}; "
                         f"choose from {CITATION_DOMAINS}")
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
        feature_dim=CITATION_FEATURE_DIM,
        num_classes=CITATION_NUM_CLASSES,
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
        lambda_align=args.lambda_align,
        lambda_ent=args.lambda_ent,
        lambda_sep=args.lambda_sep,
        lambda_pl=args.lambda_pl,
        lambda_vrex=args.lambda_vrex,
        lambda_struct=args.lambda_struct,
        sep_intra_margin=args.sep_intra_margin,
        pl_threshold=args.pl_threshold,
        target_class_prior=None,
        nodes_per_step=args.nodes_per_step,
        warmup_frac=args.warmup_frac,
        refine_frac=args.refine_frac,
        ramp_epochs=args.ramp_epochs,
        seed=args.seed,
        device=device,
    )
    cfg.symmetrize = not args.directed
    return cfg


def _label_hist(y: torch.Tensor, num_classes: int) -> str:
    counts = torch.bincount(y, minlength=num_classes).tolist()
    return " ".join(f"{c}:{n}" for c, n in enumerate(counts))


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    print("=" * 60)
    print("  FGW prototype-graph Domain Adaptation  (Citation)")
    print("=" * 60)
    print(f"  Sources       : {cfg.source_domains}")
    print(f"  Target        : {cfg.target_domain}")
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
        symmetrize=getattr(cfg, "symmetrize", True),
    )

    # Derive the feature/class dims from the data itself and keep the config
    # honest (guards against a domain with an unexpected shape).
    feat_dims = {g.x.size(1) for g in sources + [target]}
    if len(feat_dims) != 1:
        raise ValueError(f"inconsistent feature dims across domains: {feat_dims}")
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
