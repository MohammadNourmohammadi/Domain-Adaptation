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
    feature_dim: int = 3170

    # --------------------------------------------------------------- encoder
    proj_dim: int = 64
    hidden_dim: int = 32          # = d, the FGW embedding dimension

    # -------------------------------------------------------------- ego graph
    ego_size: int = 32            # k = center + (k-1) PPR neighbours
    ppr_alpha: float = 0.15       # restart probability
    ppr_iters: int = 20           # power-iteration steps
    anchor_weight: float = 1.0    # w on the anchor indicator coordinate
    anchor_mass_extra: float = 0.2  # extra probability mass on the center

    # --------------------------------------------------------- prototype bank
    num_classes: int = 2
    num_protos: int = 3           # M prototype graphs per class
    proto_size: int = 32          # n_p nodes per prototype
    adjacency_temp: float = 1.0   # temperature on the soft-adjacency sigmoid

    # ------------------------------------------------------------ FGW solver
    fgw_alpha: float = 0.5        # trade-off between feature and structure
    fgw_epsilon: Optional[float] = 0.05  # entropic Sinkhorn FGW regularisation
    fgw_max_iter: int = 50        # outer block-coordinate (FGW) iterations

    # -------------------------------------------------- classifier soft-min
    tau: float = 0.5              # temperature

    # ----------------------------------------------------------- loss weights
    lambda_align: float = 1.0
    lambda_ent: float = 0.5
    lambda_sep: float = 0.1
    lambda_pl: float = 0.1
    lambda_vrex: float = 1.0
    lambda_struct: float = 1e-3
    sep_margin: float = 1.0
    pl_threshold: float = 0.9
    target_class_prior: Optional[Tuple[float, float]] = None

    # ---------------------------------------------------- training schedule
    lr: float = 1e-3
    weight_decay: float = 5e-3
    epochs: int = 100
    warmup_frac: float = 0.2      # fraction of epochs with L_cls (+ L_sep) only
    refine_frac: float = 0.6      # after this fraction, enable L_pl
    ramp_epochs: int = 20         # sigmoid ramp on align/ent weights

    # ----------------------------------------------- mini-batching over nodes
    nodes_per_step: int = 128
    eval_batch_nodes: int = 512

    # ------------------------------------------------------------- system
    device: str = "cpu"
    seed: int = 42
