"""Feature-space kNN densification for sparse graphs.

Why this exists
---------------
The six-graph splits of Cora_full and Arxiv induce *subgraphs*: only edges with
both endpoints inside the group survive, so a citation network with a healthy
average degree turns into six fragments averaging 1-2.5 neighbours per node,
with 32-50% (Cora_full) and 9-48% (Arxiv) of nodes fully isolated. Two things
break at once on data like that:

* a 2-layer GCN is a plain MLP for every isolated node — message passing has
  nothing to pass;
* an ego-graph method has no ego-graph. Even after the support masking in
  `fgw_ego`, half the seeds reduce to a single point and the FGW distance
  collapses to a feature distance, so the structural half of the method
  contributes nothing on exactly the datasets where it is being judged.

Adding k feature-space nearest neighbours to the *low-degree* nodes restores a
neighbourhood for them while leaving well-connected nodes' topology alone. This
is a transductive graph transform, not supervision: it reads `x` only, never
`y`, and is applied identically to every source and to the target. It is off by
default (`--knn_augment 0`) because it changes the input graph and therefore has
to be reported as part of the method rather than smuggled in.

Cost is O(n_low * N * d) similarities, computed in row chunks so the full
similarity matrix is never materialised.
"""

from typing import Optional

import torch
from torch_geometric.data import Data


@torch.no_grad()
def knn_edges(
    x: torch.Tensor,
    rows: torch.Tensor,
    k: int,
    chunk: int = 1024,
) -> torch.Tensor:
    """Top-`k` cosine neighbours (excluding self) for each node in `rows`.

    Returns an edge_index of shape (2, rows.numel() * k), directed rows -> nbr.
    """
    z = torch.nn.functional.normalize(x, dim=1)
    src_parts, dst_parts = [], []
    for start in range(0, rows.numel(), chunk):
        idx = rows[start:start + chunk]
        sim = z[idx] @ z.t()                                  # (chunk, N)
        sim[torch.arange(idx.numel(), device=x.device), idx] = -float("inf")
        top = sim.topk(k=min(k, sim.size(1) - 1), dim=1).indices
        src_parts.append(idx.view(-1, 1).expand_as(top).reshape(-1))
        dst_parts.append(top.reshape(-1))
    if not src_parts:
        return x.new_empty((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])


def _unique_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """De-duplicate edge columns via a flat key.

    `torch.unique(..., dim=1)` is the obvious call but it is not implemented on
    MPS (`aten::unique_dim`), and on CUDA it lexicographically sorts whole rows,
    which is far more work than this needs. Packing each edge into one int64
    `src * N + dst` reduces it to a 1-D unique, which every backend supports.
    N^2 for the largest graph here (169,343 nodes) is 2.9e10, comfortably inside
    int64.
    """
    key = edge_index[0] * num_nodes + edge_index[1]
    key = torch.unique(key)
    return torch.stack([key // num_nodes, key % num_nodes])


@torch.no_grad()
def augment_knn(
    data: Data,
    k: int,
    min_degree: int = 3,
    chunk: int = 1024,
    name: str = "",
    verbose: bool = True,
) -> Optional[dict]:
    """Add `k` feature-kNN edges to every node with degree < `min_degree`.

    Mutates `data.edge_index` in place (symmetrised and de-duplicated) and
    returns a before/after summary, or None when `k <= 0` or the graph has
    already been augmented.

    The idempotence guard matters: the runners hand the *same* `Data` objects to
    every seed of a multi-seed run, so without it seed 2 would densify an
    already-densified graph and each seed would train on a different topology.
    """
    if k <= 0:
        return None
    if getattr(data, "_knn_augmented", False):
        return None
    n = data.num_nodes
    device = data.edge_index.device
    deg = torch.bincount(data.edge_index[0], minlength=n)
    low = (deg < min_degree).nonzero(as_tuple=False).view(-1)
    before = {
        "name": name,
        "edges": int(data.edge_index.size(1)),
        "mean_degree": float(deg.float().mean()),
        "isolated": float((deg == 0).float().mean()),
        "low": int(low.numel()),
    }
    if low.numel() == 0:
        data._knn_augmented = True
        return {**before, "added": 0,
                "mean_degree_after": before["mean_degree"],
                "isolated_after": before["isolated"]}

    new = knn_edges(data.x, low.to(device), k, chunk=chunk)
    both = torch.cat([data.edge_index, new, new.flip(0)], dim=1)
    data.edge_index = _unique_edges(both, n)
    data._knn_augmented = True

    deg2 = torch.bincount(data.edge_index[0], minlength=n)
    out = {
        **before,
        "added": int(data.edge_index.size(1)) - before["edges"],
        "mean_degree_after": float(deg2.float().mean()),
        "isolated_after": float((deg2 == 0).float().mean()),
    }
    if verbose:
        print(f"  kNN-augment {name}: {low.numel()} low-degree nodes "
              f"(< {min_degree}) got {k} feature neighbours each; "
              f"mean degree {before['mean_degree']:.2f} -> "
              f"{out['mean_degree_after']:.2f}, isolated "
              f"{100 * before['isolated']:.1f}% -> "
              f"{100 * out['isolated_after']:.1f}%")
    return out
