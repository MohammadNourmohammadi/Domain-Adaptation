"""Personalized PageRank ego-graph extraction.

For each seed node v we want a stable, fixed-size local neighbourhood
G_v = (F_v, C_v, h_v) where the k most informative neighbours of v are
selected by PPR. Because PPR scores depend only on graph structure
(not on the encoder), we precompute them once per graph and reuse the
neighbour lists / shortest-path matrices across all epochs.

Embeddings are looked up at every step (they change every iteration),
so the cache only holds the structural pieces.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_scipy_sparse_matrix


class EgoGraphCache:
    """Per-graph cache of PPR top-k neighbours and structure matrices."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        ppr_alpha: float,
        ppr_iters: int,
        ego_size: int,
    ):
        self.num_nodes = num_nodes
        self.ppr_alpha = ppr_alpha
        self.ppr_iters = ppr_iters
        self.k = ego_size

        A_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
        deg = np.asarray(A_scipy.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        D_inv = csr_matrix(
            (1.0 / deg, (np.arange(num_nodes), np.arange(num_nodes))),
            shape=(num_nodes, num_nodes),
        )
        P = (D_inv @ A_scipy).tocoo()  # row-normalised transition matrix
        self._P_indices = torch.tensor(np.vstack([P.row, P.col]), dtype=torch.long)
        self._P_values = torch.tensor(P.data, dtype=torch.float)
        self._P_shape = (num_nodes, num_nodes)
        self._A_scipy = A_scipy

        self._nbrs_dev: torch.Tensor = None
        self._struct_dev: torch.Tensor = None

    # ------------------------------------------------------------ internals
    def _sparse_P(self, device) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._P_indices.to(device),
            self._P_values.to(device),
            self._P_shape,
        ).coalesce()

    def _ppr_batch(self, seeds: torch.Tensor) -> torch.Tensor:
        device = seeds.device
        P = self._sparse_P(device)
        B = seeds.numel()
        N = self.num_nodes
        e = torch.zeros(N, B, device=device)
        e[seeds, torch.arange(B, device=device)] = 1.0
        p = e.clone()
        for _ in range(self.ppr_iters):
            p = (1.0 - self.ppr_alpha) * torch.sparse.mm(P, p) + self.ppr_alpha * e
        return p.t()  # (B, N)

    def _topk_neighbours(self, seeds: torch.Tensor) -> torch.Tensor:
        scores = self._ppr_batch(seeds)
        seed_rows = torch.arange(seeds.numel(), device=seeds.device)
        scores[seed_rows, seeds] = -float("inf")  # exclude the seed itself
        _, top_idx = torch.topk(scores, k=self.k - 1, dim=1)
        return torch.cat([seeds.view(-1, 1), top_idx], dim=1)  # (B, k)

    def _structure_matrix(self, nbrs_1d: np.ndarray) -> np.ndarray:
        sub = self._A_scipy[nbrs_1d, :][:, nbrs_1d]
        d = shortest_path(sub, directed=False, unweighted=True)
        d = np.where(np.isinf(d), float(self.k), d)
        return d.astype(np.float32)

    # ------------------------------------------------------------- precompute
    def precompute_all(self, batch_size: int = 512, device: str = "cpu") -> None:
        """Fill the neighbour list and structure tensor for every node.

        PPR runs on CPU regardless of `device`: sparse COO tensors are
        unsupported on MPS, and the result is purely structural so it
        doesn't matter where it's computed. The final cached tensors are
        moved to `device` so the per-step batch builder can index them
        without a host transfer.
        """
        N = self.num_nodes
        k = self.k
        all_nbrs = torch.empty(N, k, dtype=torch.long)
        all_C = torch.empty(N, k, k, dtype=torch.float32)

        seeds_all = torch.arange(N)  # CPU
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            seeds = seeds_all[start:end]
            nbrs = self._topk_neighbours(seeds).detach()
            all_nbrs[start:end] = nbrs
            nbrs_np = nbrs.numpy()
            for i in range(end - start):
                d = self._structure_matrix(nbrs_np[i])
                all_C[start + i] = torch.from_numpy(d) / float(self.k)  # to [0,1]

        self._nbrs_dev = all_nbrs.to(device)
        self._struct_dev = all_C.to(device)


def build_ego_batch_from_cache(
    cache: EgoGraphCache,
    embeddings: torch.Tensor,        # (N, d)
    seeds: torch.Tensor,             # (B,)
    anchor_weight: float,
    anchor_mass_extra: float,
):
    """Materialise the (F, C, h) triplets for a batch of seed nodes.

    Returns:
        F : (B, k, d + 1)   – encoder embeddings + anchor indicator coord
        C : (B, k, k)       – pairwise shortest-path distances on [0, 1]
        h : (B, k)          – probability simplex mass (centre is boosted)
    """
    device = embeddings.device
    k = cache.k
    nbrs = cache._nbrs_dev[seeds]            # (B, k)
    C_ = cache._struct_dev[seeds]            # (B, k, k)
    B = seeds.numel()

    F_emb = embeddings[nbrs]                 # (B, k, d)
    anchor = embeddings.new_zeros(B, k, 1)
    anchor[:, 0, 0] = anchor_weight
    F_ = torch.cat([F_emb, anchor], dim=-1)  # (B, k, d+1)

    h_ = embeddings.new_full((B, k), 1.0 / k)
    h_[:, 0] = h_[:, 0] + anchor_mass_extra
    h_ = h_ / h_.sum(dim=1, keepdim=True)

    return F_, C_, h_
