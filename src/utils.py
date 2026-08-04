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
    prob = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    num_classes = logits.size(1)

    metrics = {
        "acc": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, average="macro", zero_division=0),
    }
    # AUROC needs >=2 classes present in y_true.
    present = np.unique(y_true)
    if present.size < 2:
        metrics["auc"] = float("nan")
    elif num_classes == 2:
        metrics["auc"] = roc_auc_score(y_true, prob[:, 1])
    else:
        # One-vs-rest macro AUROC over only the classes present in y_true.
        # sklearn's multi_class="ovr" with labels=range(num_classes) scores
        # every class, so a class absent from the batch gives an all-negative
        # OVR column -> "Only one class present" UndefinedMetricWarning (mini-
        # batches during training routinely miss classes). Restricting to
        # present classes is warning-free and equals sklearn's macro-OVR when
        # all classes are present (e.g. full-graph evaluation), so reported
        # numbers are unchanged; each present column has both labels because
        # >=2 classes are present.
        aucs = [
            roc_auc_score((y_true == c).astype(int), prob[:, c])
            for c in present
        ]
        metrics["auc"] = float(np.mean(aucs))
    return metrics


@torch.no_grad()
def majority_baseline(labels: torch.Tensor, num_classes: int) -> dict:
    """Scores of always predicting the most frequent class.

    Worth printing next to any result on an imbalanced target. Twitch RU is
    24.5% positive, so this no-skill rule scores ACC 0.755 — higher than every
    accuracy SelMAG's Table 1 reports on Twitch — while macro-F1 is only 0.430.
    Without the row, an accuracy column on this dataset is uninterpretable.
    """
    y = labels.cpu()
    counts = torch.bincount(y, minlength=num_classes)
    pred = torch.full_like(y, int(counts.argmax()))
    logits = torch.zeros(y.numel(), num_classes)
    logits[torch.arange(y.numel()), pred] = 1.0
    m = compute_metrics(logits, y)
    m["auc"] = 0.5          # a constant score has no ranking information
    return m


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
