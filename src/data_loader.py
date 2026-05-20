import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data


# DE stores features under a different filename
_FEATURE_FILENAME = {
    "DE": "musae_DE.json",
}

AVAILABLE_DOMAINS = ["DE", "ENGB", "ES", "FR", "PTBR", "RU"]
GLOBAL_FEATURE_DIM = 3170  # max feature index across all Twitch domains + 1


def load_twitch_domain(data_root: str, domain: str) -> Data:
    """Load a single Twitch domain as a PyG Data object.

    Returns a ``Data`` with:
        x          – (N, GLOBAL_FEATURE_DIM)  float32 binary features
        edge_index – (2, E)                   long, undirected
        y          – (N,)                     long binary labels (mature=1)
    """
    domain_dir = os.path.join(data_root, domain)

    feat_file = _FEATURE_FILENAME.get(domain, f"musae_{domain}_features.json")
    with open(os.path.join(domain_dir, feat_file)) as f:
        raw_feats: dict = json.load(f)

    num_nodes = len(raw_feats)
    x = torch.zeros(num_nodes, GLOBAL_FEATURE_DIM)
    for node_str, indices in raw_feats.items():
        nid = int(node_str)
        for fi in indices:
            if fi < GLOBAL_FEATURE_DIM:
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


def load_source_target(
    data_root: str, source: str, target: str
) -> Tuple[Data, Data]:
    return (
        load_twitch_domain(data_root, source),
        load_twitch_domain(data_root, target),
    )


def load_sources_target(
    data_root: str, sources: List[str], target: str,
) -> Tuple[List[Data], Data]:
    """Load multiple source domains and a single target domain."""
    src_graphs = [load_twitch_domain(data_root, s) for s in sources]
    tgt_graph = load_twitch_domain(data_root, target)
    return src_graphs, tgt_graph
