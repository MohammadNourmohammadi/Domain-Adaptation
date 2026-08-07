"""Cora_full loader for the FGW prototype-graph DA pipeline.

The full Cora citation network (Bojchevski & Günnemann, ICLR 2018): 19,793
papers, an 8,710-dim binary bag-of-words, and 70 fine-grained subject classes.
SelMAG (arXiv:2406.10425) Table 3 lists 6 graphs averaging 3,298 nodes / 3,697
edges with 8,710 attributes and 70 classes — and 19,793 / 6 = 3,298.83, so the
six graphs partition *every* node.

How the six graphs are made
---------------------------
The paper says it clusters "papers into 6 groups based on different frequencies
in selected-words usage following GOOD [29]". GOOD's `word` domain is

    num_word = graph.x.sum(1)

(GOOD/data/good_datasets/good_cora.py, `DomainGetter.get_word`). Note what that
actually is here: **cora.npz stores TF-IDF weights, not a binary bag-of-words**
— 1,061,977 distinct values — so `x.sum(1)` is a sum of weights, not a count of
words, and it is very nearly unique per paper (19,476 distinct values over
19,793 papers). GOOD's "one indivisible domain per distinct value" therefore
degenerates to one domain per paper, and cutting into six near-equal blocks is
the only sensible reading. That is what `split_by="word"` does.

Each group induces a subgraph (only edges with both endpoints inside are kept),
which is what makes the six graphs disjoint. Ordering follows GOOD's word
domain, which reverses the ascending sort so the out-of-distribution end is the
low tail; ``W0`` is the highest-scoring group and ``W5`` the lowest, and ``W5``
is the default target — the same "hold out the far end of the shift" convention
as Twitch (RU last) and arxiv (most recent year last).

Honest caveat on reproducing Table 3
------------------------------------
Node counts confirm nothing: 19,793 / 6 = 3,298.83 matches Table 3's 3,298, but
*every* 6-way partition does, so that column carries no information about the
recipe. The edge column is the only discriminating number, and none of the
plausible recipes reproduces its 3,697 exactly:

    split_by=word    GOOD's word domain, x.sum(1)      2,150   (-42%)
    split_by=nnz     literal count of words used       2,160   (-42%)
    split_by=degree  GOOD's other documented domain    3,572   ( -3%)
    split_by=random  structure-blind control           1,746   (-53%)
                     k-means(6) on the features        6,347   (+72%)

Note the tension: the paper's *prose* says word frequency, but the *number*
only fits `degree`, GOOD's other Cora domain, to within 3%. One of the two is
probably a slip in the paper. We default to `word` because that is what the
text claims and prose is the harder thing to mistype, but `--split_by degree`
is a defensible alternative and worth reporting alongside — if your numbers
differ noticeably between the two, say so rather than quietly picking the
flattering one.

The doubled-plus-self-loops convention that Table 3 provably uses for Twitch
gives 7,598 here, so the table is not self-consistent across rows either.

Download policy
---------------
On first use ``cora.npz`` is fetched from the authors' repository into
``<data_root>/cora.npz`` (default ``data/cora_full/``) and validated against the
upstream byte count; every later run reuses that file and never touches the
network. A manually placed copy in ``<data_root>/raw/`` is picked up too, the
way the Citation and Yelp loaders accept a hand-placed dump.

Downloads are resumable and retried across two mirrors, because this particular
file truncates often (see ``CORA_FULL_BYTES``); a short or truncated file is
re-fetched automatically rather than raising and asking you to delete it.

Unlike the Citation and Yelp loaders there is deliberately no
``processed_fgw.pt``: the six graphs take a couple of seconds to derive from the
npz, and caching them would mean writing ~700 MB of densified feature matrices
(19,793 x 8,710 float32) to disk for a two-second saving.
"""

import os
import urllib.error
import urllib.request
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data


# Mirrors, tried in order. raw.githubusercontent.com is the redirect target of
# the github.com/.../raw/... form, so hitting it directly is one hop shorter and
# noticeably less likely to drop the connection mid-file.
CORA_FULL_URLS = (
    "https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/cora.npz",
    "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz",
)
CORA_FULL_URL = CORA_FULL_URLS[0]      # what the error messages point at
# Exact size of the upstream file. Both of our own fetch attempts silently
# truncated (5.7 MB and 1.7 MB of the 12.3 MB), and a truncated .npz still
# passes a `file(1)` check because the zip header is intact — only the central
# directory at the *end* is missing. Validating the length is the cheap way to
# catch that before it turns into a confusing parse error.
CORA_FULL_BYTES = 12_857_843

CORA_FULL_NUM_GRAPHS = 6
CORA_FULL_DOMAINS = [f"W{i}" for i in range(CORA_FULL_NUM_GRAPHS)]
CORA_FULL_TARGET = "W5"          # lowest word diversity = GOOD's OOD end
CORA_FULL_FEATURE_DIM = 8710
CORA_FULL_NUM_CLASSES = 70

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# One build per process: `load_sources_target` asks for six graphs and we would
# otherwise re-read and re-split the npz once per domain.
_CACHE: dict = {}


# ------------------------------------------------------------------ download
def _resume_get(url: str, tmp: str, timeout: int) -> int:
    """Fetch `url` into `tmp`, continuing a partial file. Returns bytes on disk.

    A dropped connection leaves a valid prefix, so the retry asks for the rest
    with a Range header instead of starting over. Servers that ignore Range
    answer 200 and we restart the file rather than appending to it.
    """
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": _UA}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resumed = r.status == 206
        with open(tmp, "ab" if resumed else "wb") as f:
            if not resumed:
                f.truncate(0)
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    return os.path.getsize(tmp)


def _download(path: str, timeout: int = 300, attempts: int = 3) -> None:
    """Fetch cora.npz, retrying across mirrors until the byte count matches."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    last = ""
    for attempt in range(attempts):
        for url in CORA_FULL_URLS:
            print(f"  [Cora_full] downloading {url}"
                  f"{f' (attempt {attempt + 1}/{attempts})' if attempt else ''}",
                  flush=True)
            try:
                got = _resume_get(url, tmp, timeout)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                last = str(exc)
                print(f"  [Cora_full] {exc}")
                continue
            if got == CORA_FULL_BYTES:
                os.replace(tmp, path)
                print(f"  [Cora_full] saved {got / 1e6:.1f} MB -> {path}")
                return
            # Short read: keep the prefix so the next pass resumes from it.
            last = f"got {got:,} of {CORA_FULL_BYTES:,} bytes"
            print(f"  [Cora_full] incomplete ({last}) — retrying")

    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(
        f"Could not download Cora_full after {attempts} attempts on "
        f"{len(CORA_FULL_URLS)} mirrors ({last}).\nFetch it by hand from\n"
        f"    {CORA_FULL_URL}\nand save it as {path}."
    )


def _find_npz(data_root: str) -> str:
    """Wherever cora.npz already lives: the data root, or a manual raw/ drop."""
    for cand in (os.path.join(data_root, "cora.npz"),
                 os.path.join(data_root, "raw", "cora.npz")):
        if os.path.exists(cand):
            return cand
    return ""


def _ensure_npz(data_root: str, auto_download: bool) -> str:
    path = _find_npz(data_root) or os.path.join(data_root, "cora.npz")
    if os.path.exists(path):
        got = os.path.getsize(path)
        if got == CORA_FULL_BYTES:
            return path
        # Truncated from an earlier interrupted run. Re-fetching is what the
        # user wants here; only refuse when we are not allowed to.
        if not auto_download:
            raise RuntimeError(
                f"{path} is {got:,} bytes but should be {CORA_FULL_BYTES:,} — it "
                f"looks truncated, and auto-download is off. Delete it and rerun, "
                f"or fetch {CORA_FULL_URL} by hand."
            )
        print(f"  [Cora_full] {path} is truncated ({got:,} of "
              f"{CORA_FULL_BYTES:,} bytes) -> re-downloading")
        os.remove(path)
    elif not auto_download:
        raise FileNotFoundError(
            f"{path} not found and auto-download is off. Get it from\n"
            f"    {CORA_FULL_URL}\nand save it there (a copy in "
            f"{os.path.join(data_root, 'raw')} is picked up too)."
        )
    _download(path)
    return path


# --------------------------------------------------------------------- parse
def _read_npz(path: str) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Adjacency, attributes and labels from the graph2gauss .npz layout."""
    z = np.load(path, allow_pickle=True)
    A = sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]),
        shape=tuple(z["adj_shape"]),
    )
    X = sp.csr_matrix(
        (z["attr_data"], z["attr_indices"], z["attr_indptr"]),
        shape=tuple(z["attr_shape"]),
    )
    y = np.asarray(z["labels"]).reshape(-1)
    return A, X, y


SPLIT_MODES = ("word", "nnz", "degree", "random")


def _domain_score(
    X: sp.csr_matrix, A: sp.csr_matrix, how: str, seed: int = 0,
) -> np.ndarray:
    """The per-node quantity the six groups are cut on."""
    if how == "word":       # GOOD's word domain: sum of TF-IDF weights
        return np.asarray(X.sum(axis=1)).reshape(-1).astype(np.float64)
    if how == "nnz":        # literal count of selected words used
        return np.diff(X.indptr).astype(np.float64)
    if how == "degree":     # GOOD's other documented domain
        S = ((A + A.T) > 0).astype(np.int8)
        S.setdiag(0)
        S.eliminate_zeros()
        return np.asarray(S.sum(axis=1)).reshape(-1).astype(np.float64)
    if how == "random":     # structure-blind control
        return np.random.default_rng(seed).permutation(X.shape[0]).astype(np.float64)
    raise ValueError(f"unknown split mode {how!r}; expected one of {SPLIT_MODES}")


def _score_groups(score: np.ndarray, num_groups: int) -> List[np.ndarray]:
    """Cut nodes into `num_groups` near-equal blocks by descending `score`.

    Boundaries are pushed forward to the next change in value, so nodes sharing
    a score always land in the same graph — GOOD treats each distinct value as
    one indivisible domain. With the TF-IDF `word` score values are effectively
    unique, so this reduces to plain equal-size blocks; it matters for the
    integer-valued `nnz` and `degree` modes, where ties are common.
    """
    # Descending score; ties broken by node id so the split is deterministic.
    order = np.lexsort((np.arange(score.size), -score))
    sorted_score = score[order]
    n = order.size

    # Positions where the score changes: the only places a boundary may fall
    # without splitting a value across two graphs.
    change = np.flatnonzero(np.diff(sorted_score) != 0) + 1

    groups, start = [], 0
    for g in range(num_groups - 1):
        remaining = num_groups - 1 - g          # groups still to be filled after
        ideal = int(round((g + 1) * n / num_groups))
        # A boundary must leave at least one node for each remaining group.
        lo, hi = start + 1, n - remaining
        if lo > hi:                             # not enough nodes left to spread
            end = lo
        else:
            cand = change[(change >= lo) & (change <= hi)]
            # Snap to the *nearest* change point, not the next one: always
            # rounding up let one big tie block swallow several groups' worth of
            # nodes and leave later groups empty (degree and nnz have heavy
            # ties). Fall back to the raw cut only if no legal change point
            # exists in the window.
            end = int(cand[np.argmin(np.abs(cand - ideal))]) if cand.size else \
                int(min(max(ideal, lo), hi))
        groups.append(order[start:end])
        start = end
    groups.append(order[start:])
    return groups


def _induced(
    A: sp.csr_matrix, X: sp.csr_matrix, y: np.ndarray,
    nodes: np.ndarray, symmetrize: bool,
) -> Data:
    """Subgraph induced on `nodes`, renumbered to 0..len(nodes)-1."""
    nodes = np.sort(nodes)
    sub = A[nodes, :][:, nodes].tocoo()
    edge_index = torch.from_numpy(
        np.vstack([sub.row, sub.col]).astype(np.int64)
    )
    # Drop self-loops; GCNConv adds its own.
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]
    if symmetrize:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

    # Keep the raw TF-IDF weights. These are NOT a binary bag-of-words — the
    # matrix holds ~1.06M distinct float values — and PyG's CitationFull, GOOD
    # and hence the paper all feed the weights straight through. Binarising
    # here would train on a different feature matrix than the one being
    # compared against.
    x = torch.from_numpy(np.asarray(X[nodes].todense(), dtype=np.float32))
    y_t = torch.from_numpy(y[nodes].astype(np.int64))
    return Data(x=x, edge_index=edge_index, y=y_t, num_nodes=nodes.size)


def _build_all(
    data_root: str, symmetrize: bool, auto_download: bool, verbose: bool,
    split_by: str = "word",
) -> dict:
    key = (os.path.abspath(data_root), symmetrize, split_by)
    if key in _CACHE:
        return _CACHE[key]

    path = _ensure_npz(data_root, auto_download)
    A, X, y = _read_npz(path)
    if verbose:
        print(f"  [Cora_full] reading {path}")
        print(f"  [Cora_full] {A.shape[0]} papers, {X.shape[1]} attributes, "
              f"{len(np.unique(y))} classes; split_by={split_by}")

    score = _domain_score(X, A, split_by)
    groups = _score_groups(score, CORA_FULL_NUM_GRAPHS)
    graphs = {}
    for name, nodes in zip(CORA_FULL_DOMAINS, groups):
        g = _induced(A, X, y, nodes, symmetrize)
        g.score_range = (float(score[nodes].min()), float(score[nodes].max()))
        g.split_by = split_by
        graphs[name] = g
    _CACHE[key] = graphs
    return graphs


# ----------------------------------------------------------------------- api
def load_cora_full_domain(
    data_root: str, domain: str, symmetrize: bool = True,
    auto_download: bool = True, verbose: bool = False, split_by: str = "word",
) -> Data:
    if domain not in CORA_FULL_DOMAINS:
        raise ValueError(
            f"Unknown Cora_full domain '{domain}'. Choose from {CORA_FULL_DOMAINS}."
        )
    return _build_all(data_root, symmetrize, auto_download, verbose, split_by)[domain]


def load_sources_target(
    data_root: str, sources: List[str], target: str,
    symmetrize: bool = True, auto_download: bool = True, split_by: str = "word",
) -> Tuple[List[Data], Data]:
    """Same signature as the other loaders, so the FGW runner is symmetric."""
    graphs = _build_all(data_root, symmetrize, auto_download, True, split_by)
    for name in list(sources) + [target]:
        if name not in graphs:
            raise ValueError(
                f"Unknown Cora_full domain '{name}'. Choose from {CORA_FULL_DOMAINS}."
            )
    return [graphs[s] for s in sources], graphs[target]


def split_summary(data_root: str, symmetrize: bool = True) -> str:
    """Every split mode next to SelMAG's Table 3 averages, for verification.

    Prints all modes rather than only the active one, because the point is to
    show that none of them reproduces the paper's 3,697 edges — see the module
    docstring. The node column is identical for every mode by construction.
    """
    lines = []
    for mode in SPLIT_MODES:
        graphs = _build_all(data_root, symmetrize, True, False, mode)
        tot_n = tot_e = 0
        lines.append(f"split_by = {mode}")
        for name in CORA_FULL_DOMAINS:
            g = graphs[name]
            # report undirected edges (edge_index holds both directions)
            e = g.edge_index.size(1) // (2 if symmetrize else 1)
            lo, hi = g.score_range
            lines.append(f"  {name}: {g.num_nodes:5d} nodes  {e:6d} edges  "
                         f"score {lo:.3g}-{hi:.3g}")
            tot_n += g.num_nodes
            tot_e += e
        n = len(CORA_FULL_DOMAINS)
        lines.append(f"  average: {tot_n / n:7.1f} nodes  {tot_e / n:6.1f} edges"
                     f"   ({2 * tot_e / n:.0f} if counted doubled)")
        lines.append("")
    lines.append("SelMAG Table 3: 3,298 nodes, 3,697 edges")
    lines.append("")
    lines.append("Node counts match every mode (any 6-way partition of 19,793 "
                 "averages 3,298.8),")
    lines.append("so only the edge column is informative. The paper's prose says "
                 "word frequency,")
    lines.append("but only split_by=degree lands near 3,697 (within ~3%); "
                 "split_by=word is 42% low.")
    lines.append("Default is 'word' (what the text claims); consider reporting "
                 "'degree' alongside.")
    return "\n".join(lines)
