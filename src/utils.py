import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def grl_lambda_schedule(epoch: int, warmup_epochs: int) -> float:
    """DANN-style sigmoid ramp from 0 to 1 over warmup_epochs."""
    p = min(epoch / max(warmup_epochs, 1), 1.0)
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    metrics = {
        "acc": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, average="macro", zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, probs)
    else:
        metrics["auc"] = float("nan")
    return metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
