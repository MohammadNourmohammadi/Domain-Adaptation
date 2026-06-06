"""From FGW distances to per-class predictions.

Given distances d_{c,m}(v) we aggregate over the M prototypes per class
with a smooth (soft) minimum so gradients flow to every prototype, then
turn the per-class scores into logits / probabilities.
"""

import torch
import torch.nn.functional as F


def fgw_class_distances(d_bcm: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft-min over M prototypes per class.

    d_c(v) = -tau * log sum_m exp(-d_{c,m}(v) / tau)
    """
    return -tau * torch.logsumexp(-d_bcm / tau, dim=-1)


def fgw_logits(d_bcm: torch.Tensor, tau: float) -> torch.Tensor:
    """Per-class logits suitable for cross-entropy: -d_c(v) / tau."""
    return -fgw_class_distances(d_bcm, tau) / tau


def fgw_probs(d_bcm: torch.Tensor, tau: float) -> torch.Tensor:
    return F.softmax(fgw_logits(d_bcm, tau), dim=-1)
