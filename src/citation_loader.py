"""Citation-network loader for the FGW prototype-graph DA pipeline.

Three ArnetMiner citation graphs — ACMv9, Citationv1, DBLPv7 — in the exact
raw format used by pygda's ``CitationDataset`` (``<name>_docs.txt`` dense
features, ``<name>_edgelist.txt`` citations, ``<name>_labels.txt`` integer
labels). All three share a 6775-dim bag-of-words feature space and 5 classes,
so a single shared encoder transfers across them.

Download policy (what the user asked for):

  * On first use the raw ``.txt`` files for a domain are pulled from the pygda
    Google Drive folder into ``<data_root>/<domain>/raw/`` and parsed once into
    a cached ``<data_root>/<domain>/processed_fgw.pt`` tensor bundle.
  * Every later call just ``torch.load``s that cache — no re-download, no
    re-parse of the (190 MB) text files.

Source of the raw files: the pygda project's public Drive folder
    https://drive.google.com/drive/folders/1ntNt3qHE4p9Us8Re9tZDaB-tdtqwV8AX
Repo: https://github.com/pygda-team/pygda
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


CITATION_DOMAINS = ["ACMv9", "Citationv1", "DBLPv7"]
CITATION_FEATURE_DIM = 6775   # shared BoW vocabulary across the three graphs
CITATION_NUM_CLASSES = 5      # research-area labels

# Direct Google-Drive file ids for each raw file (resolved from the pygda
# Citation folder), so only the exact files needed are fetched — not the whole
# multi-dataset folder.
_GDRIVE_IDS = {
    "ACMv9": {
        "docs": "1GafTgL1Qx4Hwbf3Grymy5opFYq1oVqCB",
        "edgelist": "1EUD7xtBNDAX0y4_zA4vBdH2wOEyD3JOS",
        "labels": "1ehrLt4uFukbq-dEqrfVnf0ygYeN_01yE",
    },
    "Citationv1": {
        "docs": "1Xud7scWzHBKmTdfOhYkjVL_aCZMkkzB9",
        "edgelist": "1Wvcasp_SZ5HJO8FFuKUy4MCGg3acEdkQ",
        "labels": "1Fry4jBR7QliDmQryMoSG6dfxaMJjfztY",
    },
    "DBLPv7": {
        "docs": "1C-xRuxZlgl4NGJERGqBwt402dN04DySG",
        "edgelist": "1pmoiv2rmt5iZqrYp4_eJItS0brzOD9sj",
        "labels": "1Ennw4sVcInSB0y_-hKLX8Y89xpjk9Eci",
    },
}


def _raw_filename(domain: str, kind: str) -> str:
    return f"{domain}_{kind}.txt"


def _download_raw(domain: str, raw_dir: str) -> None:
    """Fetch any missing raw ``.txt`` files for ``domain`` into ``raw_dir``."""
    try:
        import gdown
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "Downloading the citation dataset needs `gdown`. "
            "Install it with:  pip install gdown\n"
            "(also listed in requirements.txt)."
        ) from e

    os.makedirs(raw_dir, exist_ok=True)
    for kind, file_id in _GDRIVE_IDS[domain].items():
        out = os.path.join(raw_dir, _raw_filename(domain, kind))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        print(f"    downloading {domain}_{kind}.txt ...", flush=True)
        gdown.download(id=file_id, output=out, quiet=True)
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            raise RuntimeError(
                f"Failed to download {out} from Google Drive. Check your network "
                f"or download the pygda Citation folder manually into {raw_dir}."
            )


def _parse_raw(domain: str, raw_dir: str, symmetrize: bool = True) -> Data:
    """Parse the three raw ``.txt`` files into a PyG ``Data`` object.

    Mirrors pygda's ``CitationDataset.process`` (comma-separated dense
    features, comma-separated edge pairs, one integer label per line), then
    symmetrises the citation edges so the shared GCN sees an undirected graph
    — consistent with the Twitch loader and standard citation-GDA practice.
    """
    # --- edges: comma-separated "src,dst" per line -> (2, E) long
    edge_path = os.path.join(raw_dir, _raw_filename(domain, "edgelist"))
    edges = np.loadtxt(edge_path, delimiter=",", dtype=np.int64)
    edge_index = torch.from_numpy(edges).t().contiguous()
    if symmetrize:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # --- features: each line is a dense comma-separated vector -> (N, F) float
    # (pandas is far faster than np.loadtxt on the ~190 MB docs file)
    docs_path = os.path.join(raw_dir, _raw_filename(domain, "docs"))
    feats = pd.read_csv(docs_path, header=None, dtype=np.float32).to_numpy()
    x = torch.from_numpy(feats).to(torch.float)

    # --- labels: one integer per line -> (N,) long
    label_path = os.path.join(raw_dir, _raw_filename(domain, "labels"))
    y = torch.from_numpy(
        np.loadtxt(label_path, dtype=np.int64)
    ).to(torch.long)

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=x.size(0))


def load_citation_domain(
    data_root: str, domain: str, symmetrize: bool = True,
) -> Data:
    """Load one citation domain, downloading + caching on first use.

    ``<data_root>/<domain>/processed_fgw.pt`` is the cache. If it exists it is
    loaded directly; otherwise the raw files are downloaded (if missing),
    parsed, and the result is cached before returning.
    """
    if domain not in _GDRIVE_IDS:
        raise ValueError(
            f"Unknown citation domain '{domain}'. "
            f"Choose from {CITATION_DOMAINS}."
        )

    domain_dir = os.path.join(data_root, domain)
    cache_path = os.path.join(domain_dir, "processed_fgw.pt")

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    raw_dir = os.path.join(domain_dir, "raw")
    have_all = all(
        os.path.exists(os.path.join(raw_dir, _raw_filename(domain, k)))
        for k in ("docs", "edgelist", "labels")
    )
    if not have_all:
        print(f"  [{domain}] raw files missing -> downloading from Google Drive")
        _download_raw(domain, raw_dir)

    data = _parse_raw(domain, raw_dir, symmetrize=symmetrize)
    os.makedirs(domain_dir, exist_ok=True)
    torch.save(data, cache_path)
    print(f"  [{domain}] cached parsed graph -> {cache_path}")
    return data


def load_sources_target(
    data_root: str, sources: List[str], target: str,
    symmetrize: bool = True,
) -> Tuple[List[Data], Data]:
    """Load several source citation graphs and one target graph.

    Same signature as ``src.data_loader.load_sources_target`` so the FGW
    runner is symmetric across the Twitch and citation settings.
    """
    src_graphs = [
        load_citation_domain(data_root, s, symmetrize=symmetrize) for s in sources
    ]
    tgt_graph = load_citation_domain(data_root, target, symmetrize=symmetrize)
    return src_graphs, tgt_graph
