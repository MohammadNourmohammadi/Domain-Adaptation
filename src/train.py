from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

from .config import Config
from .models import CausalGNN_DANN
from .utils import compute_metrics, grl_lambda_schedule


_EPS = 1e-6


def _drop_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return edge_index
    new_ei, _ = dropout_edge(
        edge_index, p=p, force_undirected=True, training=True,
    )
    return new_ei


def _binary_entropy(w: torch.Tensor) -> torch.Tensor:
    """Per-edge binary entropy: 0 at w=0 or w=1, max at w=0.5.

    Penalising this term pushes each weight toward a confident 0 (drop)
    or 1 (keep) rather than collapsing the whole mask to zero.
    """
    return -(w * (w + _EPS).log() + (1.0 - w) * (1.0 - w + _EPS).log())


def train_step(
    model: CausalGNN_DANN,
    sources: List[Data],
    target: Data,
    optimizer: torch.optim.Optimizer,
    config: Config,
    epoch: int,
) -> dict:
    """One full-batch pass over every source and the target graph.

    Domain labels: source_i -> i,  target -> len(sources).
    """
    model.train()
    alpha = grl_lambda_schedule(epoch, config.grl_warmup_epochs)
    device = target.x.device
    num_sources = len(sources)

    label_losses = []
    domain_losses = []
    sparsity_terms = []
    counter_losses = []
    per_source_metrics = []

    for i, src in enumerate(sources):
        ei = _drop_edges(src.edge_index, config.drop_edge_p)

        logits_s, dom_s, w_s = model(src.x, ei, alpha)
        ce_pos = F.cross_entropy(logits_s, src.y)
        label_losses.append(ce_pos)

        dom_label = torch.full(
            (src.num_nodes,), i, dtype=torch.long, device=device,
        )
        domain_losses.append(F.cross_entropy(dom_s, dom_label))

        sparsity_terms.append(_binary_entropy(w_s).mean())

        # Counterfactual: the kept edges (mask w) should be more useful for
        # predicting y than the anti-mask (1 - w). Hinge: anti-mask CE must
        # be at least `counter_margin` nats worse than the regular CE.
        logits_anti = model.predict_with_weights(src.x, ei, 1.0 - w_s)
        ce_anti = F.cross_entropy(logits_anti, src.y)
        counter_losses.append(
            F.relu(ce_pos.detach() - ce_anti + config.counter_margin)
        )

        per_source_metrics.append(compute_metrics(logits_s, src.y))

    # target: only domain loss + sparsity (no labels used)
    ei_t = _drop_edges(target.edge_index, config.drop_edge_p)
    _, dom_t, w_t = model(target.x, ei_t, alpha)
    dom_label_t = torch.full(
        (target.num_nodes,), num_sources, dtype=torch.long, device=device,
    )
    domain_losses.append(F.cross_entropy(dom_t, dom_label_t))
    sparsity_terms.append(_binary_entropy(w_t).mean())

    loss_label = torch.stack(label_losses).mean()
    loss_domain = torch.stack(domain_losses).mean()
    loss_sparse = torch.stack(sparsity_terms).mean()
    loss_counter = torch.stack(counter_losses).mean()
    # V-REx: penalise variance of per-source risks so the predictor has to
    # be (near-)optimal on every source simultaneously — a proxy for using
    # only invariant (causal) features. Biased var so single-source gives 0.
    loss_vrex = torch.stack(label_losses).var(unbiased=False)

    loss = (
        loss_label
        + config.lambda_domain * loss_domain
        + config.lambda_sparse * loss_sparse
        + config.lambda_counter * loss_counter
        + config.lambda_vrex * loss_vrex
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_label": loss_label.item(),
        "loss_domain": loss_domain.item(),
        "loss_sparse": loss_sparse.item(),
        "loss_counter": loss_counter.item(),
        "loss_vrex": loss_vrex.item(),
        "alpha": alpha,
        "src_f1_mean": sum(m["f1"] for m in per_source_metrics) / num_sources,
        "per_source_metrics": per_source_metrics,
    }


@torch.no_grad()
def evaluate(model: CausalGNN_DANN, data: Data) -> dict:
    model.eval()
    logits, _, edge_weights = model(data.x, data.edge_index, alpha=0.0)
    loss = F.cross_entropy(logits, data.y).item()
    metrics = compute_metrics(logits, data.y)
    metrics["loss"] = loss
    metrics["avg_edge_weight"] = float(edge_weights.mean().item())
    return metrics


def run_training(
    model: CausalGNN_DANN,
    sources: List[Data],
    target: Data,
    config: Config,
) -> CausalGNN_DANN:
    device = config.device
    sources = [s.to(device) for s in sources]
    target = target.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    best_tgt_f1 = -1.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        stats = train_step(model, sources, target, optimizer, config, epoch)
        tgt = evaluate(model, target)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch:3d} | "
                f"loss {stats['loss']:.4f} "
                f"(lbl {stats['loss_label']:.4f}  "
                f"dom {stats['loss_domain']:.4f}  "
                f"sp {stats['loss_sparse']:.4f}  "
                f"cf {stats['loss_counter']:.4f}  "
                f"vx {stats['loss_vrex']:.4f}) | "
                f"alpha {stats['alpha']:.3f} | "
                f"mean src_f1 {stats['src_f1_mean']:.4f} | "
                f"tgt_acc {tgt['acc']:.4f}  tgt_f1 {tgt['f1']:.4f}  "
                f"tgt_auc {tgt['auc']:.4f}"
            )

        if tgt["f1"] > best_tgt_f1:
            best_tgt_f1 = tgt["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model
