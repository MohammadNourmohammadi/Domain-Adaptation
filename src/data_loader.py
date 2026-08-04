"""Twitch loaders.

Two feature representations of the *same six graphs* are supported:

``musae`` (default)
    The authors' own release, verified byte-identical to SNAP's `twitch.zip`:
    **3170-dim** binary bag-of-words. Fetch/verify with
    `scripts/download_twitch.py`.

``pyg``
    The dense **128-dim** ``torch_geometric.datasets.Twitch`` release, which is
    what SelMAG (arXiv:2406.10425) used — its Table 3 lists "#Attributes 128".
    **These files no longer exist.** graphmining.ai was suspended and the domain
    now returns NXDOMAIN (pytorch_geometric#10346, #10510, #10672), so nobody
    can fetch them. The path is kept in case they resurface; it raises a clear
    error otherwise.

The graphs and labels are the paper's either way. Counting each undirected edge
twice plus one self-loop per node — how PyG's STATS table, and hence Table 3,
counts — the six graphs average 148,724 edges and 5,686.67 nodes against Table
3's 148,724 and 5,687. (The node figure includes FR's two duplicate target rows,
ids 3754 and 1018; they carry identical labels, so the graph is unaffected. PyG
documents FR as 6,551 nodes for the same reason; it really has 6,549.)

Since the 128-d file is unrecoverable, the faithful substitute is the encoder's
frozen unsupervised SVD basis, which reduces these 3170-d features to
``--proj_dim`` (default 128). That runs at the paper's stated width, on the
paper's graphs, via a reduction that is written down rather than lost.
"""

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# DE stores features under a different filename
_FEATURE_FILENAME = {
    "DE": "musae_DE.json",
}

AVAILABLE_DOMAINS = ["DE", "ENGB", "ES", "FR", "PTBR", "RU"]

# Raw MUSAE bag-of-words width (max feature index across all domains + 1).
MUSAE_FEATURE_DIM = 3170
# Backwards-compatible alias; `feature_dim()` is the preferred accessor.
GLOBAL_FEATURE_DIM = MUSAE_FEATURE_DIM

PYG_FEATURE_DIM = 128
# Repo domain name -> the name PyG files the same graph under.
_PYG_NAME = {"DE": "DE", "ENGB": "EN", "ES": "ES", "FR": "FR",
             "PTBR": "PT", "PT": "PT", "RU": "RU"}
DEFAULT_PYG_ROOT = "data/twitch_pyg"

FEATURE_SETS = ("pyg", "musae")


def feature_dim(features: str = "musae") -> int:
    if features == "pyg":
        return PYG_FEATURE_DIM
    if features == "musae":
        return MUSAE_FEATURE_DIM
    raise ValueError(f"unknown feature set {features!r}; expected one of {FEATURE_SETS}")


# ------------------------------------------------------------------ musae (3170-d)
def _load_musae(data_root: str, domain: str) -> Data:
    domain_dir = os.path.join(data_root, domain)

    feat_file = _FEATURE_FILENAME.get(domain, f"musae_{domain}_features.json")
    with open(os.path.join(domain_dir, feat_file)) as f:
        raw_feats: dict = json.load(f)

    num_nodes = len(raw_feats)
    x = torch.zeros(num_nodes, MUSAE_FEATURE_DIM)
    for node_str, indices in raw_feats.items():
        nid = int(node_str)
        for fi in indices:
            if fi < MUSAE_FEATURE_DIM:
                x[nid, fi] = 1.0

    edges_df = pd.read_csv(os.path.join(domain_dir, f"musae_{domain}_edges.csv"))
    src = torch.tensor(edges_df["from"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["to"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    target_df = pd.read_csv(os.path.join(domain_dir, f"musae_{domain}_target.csv"))
    y = torch.zeros(num_nodes, dtype=torch.long)
    for _, row in target_df.iterrows():
        nid = int(row["new_id"])
        if nid < num_nodes:
            y[nid] = int(row["mature"] in (True, "True"))

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


# -------------------------------------------------------------------- pyg (128-d)
def _load_pyg(pyg_root: str, domain: str) -> Data:
    name = _PYG_NAME[domain]
    path = os.path.join(pyg_root, "raw", f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — and it almost certainly cannot be obtained.\n"
            f"The 128-d Twitch release SelMAG used was hosted on graphmining.ai, "
            f"whose account was suspended; the domain now returns NXDOMAIN, so "
            f"no mirror of the original remains (pytorch_geometric#10346, "
            f"#10510, #10672).\n\n"
            f"Use the authors' own features instead:\n"
            f"    python main_fgw.py --features musae\n"
            f"The encoder's frozen unsupervised SVD basis reduces them to "
            f"--proj_dim (default 128), i.e. the paper's stated width on the "
            f"paper's graphs.\n\n"
            f"If you do locate the npz files, drop them in {pyg_root}/raw/."
        )
    z = np.load(path, allow_pickle=True)
    x = torch.from_numpy(np.asarray(z["features"])).to(torch.float)
    y = torch.from_numpy(np.asarray(z["target"])).to(torch.long)
    edges = torch.from_numpy(np.asarray(z["edges"])).to(torch.long)
    if edges.size(0) == 2 and edges.size(1) != 2:      # already (2, E)
        edge_index = edges
    else:                                              # stored as (E, 2)
        edge_index = edges.t().contiguous()
    num_nodes = x.size(0)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


# ------------------------------------------------------------------------- api
def load_twitch_domain(
    data_root: str,
    domain: str,
    features: str = "musae",
    pyg_root: str = DEFAULT_PYG_ROOT,
) -> Data:
    """Load a single Twitch domain as a PyG ``Data``.

    Returns a ``Data`` with:
        x          – (N, 128) dense or (N, 3170) binary float32 features
        edge_index – (2, E)   long, undirected
        y          – (N,)     long binary labels (mature = 1)
    """
    if features == "pyg":
        return _load_pyg(pyg_root, domain)
    if features == "musae":
        return _load_musae(data_root, domain)
    raise ValueError(f"unknown feature set {features!r}; expected one of {FEATURE_SETS}")


def load_source_target(
    data_root: str, source: str, target: str, features: str = "musae",
    pyg_root: str = DEFAULT_PYG_ROOT,
) -> Tuple[Data, Data]:
    return (
        load_twitch_domain(data_root, source, features, pyg_root),
        load_twitch_domain(data_root, target, features, pyg_root),
    )


def load_sources_target(
    data_root: str,
    sources: List[str],
    target: str,
    features: str = "musae",
    pyg_root: str = DEFAULT_PYG_ROOT,
) -> Tuple[List[Data], Data]:
    """Load multiple source domains and a single target domain."""
    src_graphs = [
        load_twitch_domain(data_root, s, features, pyg_root) for s in sources
    ]
    tgt_graph = load_twitch_domain(data_root, target, features, pyg_root)
    return src_graphs, tgt_graph
