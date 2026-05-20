from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Data
    data_root: str = "data"
    source_domains: List[str] = field(default_factory=lambda: ["DE", "FR"])
    target_domain: str = "ENGB"
    feature_dim: int = 3170  # binary BoW dim shared across all Twitch domains

    # Model
    proj_dim: int = 64        # project raw features down before GNN
    hidden_dim: int = 32
    num_classes: int = 2

    # Loss weights
    lambda_domain: float = 1.0    # DANN domain loss weight
    lambda_sparse: float = 0.01   # binary-entropy penalty on edge mask
    lambda_counter: float = 0.5   # counterfactual (anti-mask) loss weight
    lambda_vrex: float = 1.0      # V-REx (variance of per-source risks)
    counter_margin: float = 1.0   # hinge margin (nats) for counterfactual loss

    # Augmentation
    drop_edge_p: float = 0.15     # DropEdge probability during training

    # Training
    lr: float = 1e-3
    weight_decay: float = 5e-3
    epochs: int = 100
    grl_warmup_epochs: int = 20  # ramp GRL lambda over this many epochs
    device: str = "cpu"
    seed: int = 42
