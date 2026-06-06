"""Batched Fused Gromov-Wasserstein distance.

The previous version called POT's conditional-gradient solver
(`ot.gromov.fused_gromov_wasserstein2`) once per (ego, class, prototype)
triplet, in a Python triple loop. With the exact (`epsilon=None`) solver
every inner iteration falls back to `ot.emd`, a single-threaded CPU
network-simplex routine, so on CUDA inputs each tiny 32x32 problem was
shuttled GPU->CPU->GPU dozens of times. That made the run sync-bound:
neither the CPU nor the GPU was busy, yet everything crawled.

This rewrite solves *all* B*C*M problems at once as batched tensor ops on
whatever device the inputs live on, using a log-stabilised entropic
(Sinkhorn) FGW with the square-loss decomposition of Peyre et al. (2016).
The transport plans are computed under `no_grad` and detached; the FGW
value is then evaluated as an explicit (differentiable) function of the
inputs at the fixed plan (envelope theorem), so gradients still flow to:
  - F_e (encoder weights / features / inputs);
  - F_p (prototype embeddings);
  - C_p (prototype structure matrices, via the soft adjacency).
"""

from typing import Optional

import torch


# Default entropic regularisation used when the caller passes epsilon=None.
# Small enough to stay close to the exact OT plan, large enough that the
# log-domain Sinkhorn converges quickly and stably.
_DEFAULT_EPSILON = 0.05

# Inner Sinkhorn sweeps per outer FGW (block-coordinate) iteration.
_SINKHORN_ITERS = 50


def _log_sinkhorn(
    cost: torch.Tensor,   # (P, k, n)
    log_p: torch.Tensor,  # (P, k)
    log_q: torch.Tensor,  # (P, n)
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
    """Entropic OT plan in the log domain (numerically stable).

    Returns the transport plan T of shape (P, k, n) with row marginals
    exp(log_p) and column marginals exp(log_q).
    """
    Mr = -cost / epsilon                      # (P, k, n)
    f = torch.zeros_like(log_p)               # (P, k)
    g = torch.zeros_like(log_q)               # (P, n)
    for _ in range(n_iters):
        f = epsilon * (log_p - torch.logsumexp(Mr + (g / epsilon).unsqueeze(1), dim=2))
        g = epsilon * (log_q - torch.logsumexp(Mr + (f / epsilon).unsqueeze(2), dim=1))
    log_T = Mr + (f / epsilon).unsqueeze(2) + (g / epsilon).unsqueeze(1)
    return log_T.exp()


def pairwise_fgw_distances(
    F_e: torch.Tensor,        # (B, k_e, d+1)
    C_e: torch.Tensor,        # (B, k_e, k_e)
    h_e: torch.Tensor,        # (B, k_e)
    F_p: torch.Tensor,        # (C, M, n_p, d+1)
    C_p: torch.Tensor,        # (C, M, n_p, n_p)
    q: torch.Tensor,          # (n_p,)
    alpha: float = 0.5,
    epsilon: Optional[float] = None,
    max_iter: int = 50,
) -> torch.Tensor:
    """Returns FGW distances of shape (B, C, M).

    All work runs on the device of the inputs (e.g. CUDA) as batched
    tensor ops; nothing is moved to host.
    """
    eps = epsilon if (epsilon is not None and epsilon > 0) else _DEFAULT_EPSILON

    B, k, D = F_e.shape
    C_cls, M, n_p, _ = F_p.shape
    P = B * C_cls * M

    # ----- broadcast every (ego, class, prototype) pair into a flat batch P
    F_p_flat = F_p.reshape(C_cls * M, n_p, D)          # (CM, n_p, D)
    C_p_flat = C_p.reshape(C_cls * M, n_p, n_p)        # (CM, n_p, n_p)

    Fe_b = (
        F_e.unsqueeze(1).expand(B, C_cls * M, k, D).reshape(P, k, D)
    )
    Fp_b = (
        F_p_flat.unsqueeze(0).expand(B, C_cls * M, n_p, D).reshape(P, n_p, D)
    )
    C1 = C_e.unsqueeze(1).expand(B, C_cls * M, k, k).reshape(P, k, k)
    C2 = C_p_flat.unsqueeze(0).expand(B, C_cls * M, n_p, n_p).reshape(P, n_p, n_p)
    p_marg = h_e.unsqueeze(1).expand(B, C_cls * M, k).reshape(P, k)
    q_marg = q.view(1, n_p).expand(P, n_p)

    # feature cost M_ij = ||F_e[i] - F_p[j]||^2
    Mf = torch.cdist(Fe_b, Fp_b) ** 2                  # (P, k, n_p)

    # square-loss GW constants (Peyre 2016): h1(a)=a, h2(b)=2b, f(x)=x^2
    a = torch.bmm(C1 ** 2, p_marg.unsqueeze(-1)).squeeze(-1)   # (P, k)
    b = torch.bmm(C2 ** 2, q_marg.unsqueeze(-1)).squeeze(-1)   # (P, n_p)
    constC = a.unsqueeze(-1) + b.unsqueeze(1)                  # (P, k, n_p)

    log_p = (p_marg + 1e-30).log()
    log_q = (q_marg + 1e-30).log()

    # ----- block-coordinate descent: linearise GW at T, Sinkhorn, repeat
    with torch.no_grad():
        T = p_marg.unsqueeze(-1) * q_marg.unsqueeze(1)        # (P, k, n_p)
        for _ in range(max(1, max_iter)):
            tens = constC - 2.0 * torch.bmm(torch.bmm(C1, T), C2)
            cost = (1.0 - alpha) * Mf + 2.0 * alpha * tens
            T = _log_sinkhorn(cost, log_p, log_q, eps, _SINKHORN_ITERS)
        T = T.detach()

    # ----- differentiable FGW value at the fixed plan (envelope theorem)
    tens = constC - 2.0 * torch.bmm(torch.bmm(C1, T), C2)
    gw = (tens * T).sum(dim=(1, 2))
    feat = (Mf * T).sum(dim=(1, 2))
    fgw = (1.0 - alpha) * feat + alpha * gw                    # (P,)

    return fgw.view(B, C_cls, M)
