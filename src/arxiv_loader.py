"""Arxiv (ogbn-arxiv) loader for the FGW prototype-graph DA pipeline.

The OGB arXiv CS citation network [Hu et al., 2020]: 169,343 papers, a 128-dim
averaged word2vec embedding of title+abstract, 1,166,243 directed citations,
40 subject-area classes, and a publication year per node. SelMAG
(arXiv:2406.10425) Appendix A says it splits this into "6 disjoint graphs based
on the published year of each paper" and uses "the most recent one" as the
target. Table 3 lists 6 graphs averaging 28,223 nodes / 66,166 edges, and
169,343 / 6 = 28,223.83 — so the six graphs partition *every* node, as with
Cora_full.

How the six graphs are made
---------------------------
Every node lands in exactly one year band; each band induces a subgraph (an
edge survives only if both endpoints are inside), which is what makes the six
graphs disjoint. Bands are ordered oldest -> newest, so ``Y5`` is the most
recent and is the default target — the same "hold out the far end of the shift"
convention as Twitch (RU last) and Cora_full (W5 last).

The paper never states where the year cuts fall, and the choice matters a lot
here because ogbn-arxiv is extremely skewed in time: 71% of papers are from
2018 or later and 2020 alone holds 8,892. Two readings are implemented:

``year_tail`` (default)
    Equal-size node blocks, but with the last two bands merged so the target is
    2019+2020 rather than 2020 alone, and the resulting slack spent splitting
    2016-2017 back apart to keep six graphs:

        <=2013 | 2014-2015 | 2016 | 2017 | 2018 | 2019-2020

``year``
    The literal equal-size reading: nodes sorted by year and cut into six
    near-equal blocks, with boundaries snapped to a year change so no year is
    split across two graphs (GOOD treats each distinct domain value as
    indivisible). This yields

        <=2013 | 2014-2015 | 2016-2017 | 2018 | 2019 | 2020

    and its target, 2020, is unusable: 2020 papers overwhelmingly cite *older*
    papers, so the induced 2020 subgraph keeps only 1,200 of its citations and
    **81.5% of its nodes are isolated** (vs 35.6% for the ``year_tail`` target
    and 8-48% for every source band). A GNN on that target is scoring almost
    entirely on node features with no message passing, so ``year_tail`` is the
    default. Pass ``--split_by year`` for the literal cut, or ``--year_cuts``
    for any other band edges.

``degree`` and ``random`` are the same controls Cora_full carries: GOOD's other
documented arXiv domain, and a structure-blind partition.

Honest caveat on reproducing Table 3
------------------------------------
As with Cora_full, the node column confirms nothing — *every* 6-way partition
of 169,343 averages 28,223.8 — so only the edge column is informative, and no
year-based cut reproduces it:

    split_by=year_tail  (default)                    35,799 undirected / graph
    split_by=year       (literal equal-size)         38,700
    split_by=degree     (GOOD's other domain)        94,200
    split_by=random     (structure-blind control)    32,105
    SelMAG Table 3                                   66,166

Under the doubled convention Table 3 provably uses for Twitch (edge_index
columns after symmetrising, i.e. 2x the undirected count) the default lands at
71,599 against the paper's 66,166 — 8% high, and the closest any contiguous
year cut can get is 70,474 (6.5% high; bands <=2011 | 2012-14 | 2015-16 | 2017
| 2018 | 2019-20). Under the single-count convention the paper's number is 85%
*above* the default and unreachable by any balanced year cut. So the paper's
arXiv row, like its Cora_full row, is not reproducible from the stated recipe;
the doubled reading is the only one that is even close.

Worth knowing before quoting the agreement as evidence: doubled, a *random*
6-way partition gives 64,211, which is nearer 66,166 than any year cut is. The
edge column is simply too weak a check to identify the recipe either way.

Download policy
---------------
Three levels, each skipped once the level below it exists — so the first run
does everything and every later run does nothing but read:

1. ``<data_root>/arxiv.zip`` (83 MB) is fetched from the project's Google Drive
   copy via gdown, falling back to OGB's own host. Transfers resume from a
   partial ``.part`` and are retried across both sources; a truncated archive
   left by an interrupted run is detected by byte count and re-fetched rather
   than reported as corrupt. The archive is kept, so re-extracting never needs
   the network.
2. It is unpacked in place to ``<data_root>/arxiv/raw/*.csv.gz``.
3. Those CSVs are parsed once into ``<data_root>/cache/*.npy`` — the 76 MB
   gzipped feature matrix costs a few seconds to parse and ~10 ms to memory-map
   afterwards. This is the analogue of the Citation / Yelp
   ``processed_fgw.pt``, kept as ``.npy`` because it is shared across every
   ``--split_by`` instead of being one file per built graph.

Deriving the six graphs from the cache takes about a second, so there is no
per-split cache on top. ``data/arxiv/`` is gitignored; the whole tree is
reproducible from the Drive link.
"""

import os
import shutil
import urllib.error
import urllib.request
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data


# The copy this project uses (same bytes as OGB's, mirrored so the download
# works from regions where snap.stanford.edu is slow or blocked).
ARXIV_GDRIVE_ID = "15EmUoNw3LuGNl104lINurp8F-vHc34oU"
ARXIV_OGB_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
# Exact size of the archive. A truncated zip still looks like a zip to file(1)
# — only the central directory at the *end* is missing — so validating the
# length is the cheap way to catch a half-finished download early.
ARXIV_ZIP_BYTES = 83_058_288

ARXIV_NUM_NODES = 169_343
ARXIV_NUM_EDGES = 1_166_243        # directed, as stored in edge.csv.gz
ARXIV_NUM_GRAPHS = 6
ARXIV_DOMAINS = [f"Y{i}" for i in range(ARXIV_NUM_GRAPHS)]
ARXIV_TARGET = "Y5"                # most recent band
ARXIV_FEATURE_DIM = 128
ARXIV_NUM_CLASSES = 40

SPLIT_MODES = ("year_tail", "year", "degree", "random")
# Upper year of each of the first five bands; see the module docstring.
YEAR_TAIL_CUTS = (2013, 2015, 2016, 2017, 2018)

_RAW_FILES = ("node-feat.csv.gz", "node-label.csv.gz", "node_year.csv.gz",
              "edge.csv.gz")

# One build per process: `load_sources_target` asks for six graphs and we would
# otherwise re-read and re-split the arrays once per domain.
_CACHE: dict = {}


# ------------------------------------------------------------------ download
def _gdrive_get(tmp: str) -> int:
    """Pull the archive off Google Drive with gdown. Returns bytes on disk.

    gdown gets its own scratch path rather than `tmp` directly: handed an
    output file that already exists it prints "Skipping already downloaded
    file" and returns without transferring anything — even with `resume=True`,
    which only skips rather than continuing. Pointed at a partial `.part` it
    would therefore succeed silently and hand back a truncated archive forever.
    """
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "auto-download needs gdown:  pip install gdown\n"
            "(also listed in requirements.txt)."
        ) from exc
    scratch = tmp + ".gd"
    if os.path.exists(scratch):
        os.remove(scratch)
    print(f"  [arxiv] downloading from Google Drive (id={ARXIV_GDRIVE_ID})",
          flush=True)
    gdown.download(id=ARXIV_GDRIVE_ID, output=scratch, quiet=False)
    if not os.path.exists(scratch):
        return 0
    os.replace(scratch, tmp)
    return os.path.getsize(tmp)


def _resume_get(url: str, tmp: str, timeout: int) -> int:
    """Fetch `url` into `tmp`, continuing a partial file. Returns bytes on disk.

    A dropped connection leaves a valid prefix, so the retry asks for the rest
    with a Range header instead of starting over. Servers that ignore Range
    answer 200 and we restart the file rather than appending to it.
    """
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resumed = r.status == 206
        with open(tmp, "ab" if resumed else "wb") as f:
            if not resumed:
                f.truncate(0)
            shutil.copyfileobj(r, f, 1 << 20)
    return os.path.getsize(tmp)


def _fetch_zip(path: str, timeout: int = 600, attempts: int = 3) -> None:
    """Fetch arxiv.zip, retrying across both sources until the size matches.

    Google Drive (the project's mirror, and what the dataset was added from)
    goes first on a clean start; OGB's own host goes first when there is a
    partial file to finish, since only it can resume.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    last = ""
    for attempt in range(attempts):
        # OGB's host honours Range and `_resume_get` continues from whatever is
        # already on disk; gdown always restarts. So once a partial file exists,
        # finishing it beats going back to Drive for the whole 83 MB.
        partial = os.path.exists(tmp) and os.path.getsize(tmp) > 0
        for source in ("ogb", "drive") if partial else ("drive", "ogb"):
            try:
                if source == "drive":
                    got = _gdrive_get(tmp)
                else:
                    print(f"  [arxiv] downloading {ARXIV_OGB_URL}"
                          f"{f' (attempt {attempt + 1}/{attempts})' if attempt else ''}",
                          flush=True)
                    got = _resume_get(ARXIV_OGB_URL, tmp, timeout)
            # Broad on purpose: gdown raises its own hierarchy
            # (FileURLRetrievalError et al.) that shares no base class with
            # urllib's, and any failure here should fall through to the next
            # source rather than kill the run.
            except Exception as exc:                          # noqa: BLE001
                msg = str(exc).strip()
                last = msg.splitlines()[0] if msg else repr(exc)
                print(f"  [arxiv] {source} download failed ({last})")
                continue
            if got == ARXIV_ZIP_BYTES:
                os.replace(tmp, path)
                print(f"  [arxiv] saved {got / 1e6:.1f} MB -> {path}")
                return
            # Short read: keep the prefix so the next pass resumes from it.
            last = f"got {got:,} of {ARXIV_ZIP_BYTES:,} bytes"
            print(f"  [arxiv] incomplete ({last}) — retrying")

    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(
        f"Could not download ogbn-arxiv after {attempts} attempts on 2 sources "
        f"({last}). Fetch it by hand from\n"
        f"    https://drive.google.com/file/d/{ARXIV_GDRIVE_ID}/view\n"
        f"    (or  gdown {ARXIV_GDRIVE_ID} -O {path})\n"
        f"    (or  {ARXIV_OGB_URL})\n"
        f"and save it as {path}."
    )


def _find_raw_dir(data_root: str) -> str:
    """Locate the directory holding the raw CSVs, whatever the unpack layout."""
    for cand in (os.path.join(data_root, "arxiv", "raw"),
                 os.path.join(data_root, "raw"),
                 data_root):
        if all(os.path.exists(os.path.join(cand, f)) for f in _RAW_FILES):
            return cand
    return ""


def _ensure_raw(data_root: str, auto_download: bool) -> str:
    """Raw CSVs on disk, downloading and unpacking the archive if need be.

    The archive is kept next to the CSVs, so a later re-extract never needs the
    network again — the same "download once, reuse forever" contract the
    Citation and Yelp loaders give.
    """
    raw = _find_raw_dir(data_root)
    if raw:
        return raw

    zip_path = os.path.join(data_root, "arxiv.zip")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) != ARXIV_ZIP_BYTES:
        got = os.path.getsize(zip_path)
        if not auto_download:
            raise RuntimeError(
                f"{zip_path} is {got:,} bytes but should be {ARXIV_ZIP_BYTES:,} "
                f"— it looks truncated, and auto-download is off. Delete it and "
                f"rerun, or fetch it by hand from\n    {ARXIV_OGB_URL}"
            )
        # Truncated from an earlier interrupted run; _fetch_zip resumes from
        # the .part file rather than restarting the 83 MB transfer.
        print(f"  [arxiv] {zip_path} is truncated ({got:,} of "
              f"{ARXIV_ZIP_BYTES:,} bytes) -> re-downloading")
        os.replace(zip_path, zip_path + ".part")
    if not os.path.exists(zip_path):
        if not auto_download:
            raise FileNotFoundError(
                f"No ogbn-arxiv raw CSVs under {data_root} and auto-download is "
                f"off. Get the archive from\n"
                f"    https://drive.google.com/file/d/{ARXIV_GDRIVE_ID}/view\n"
                f"    (or  gdown {ARXIV_GDRIVE_ID} -O {zip_path})\n"
                f"    (or  {ARXIV_OGB_URL})\n"
                f"and unzip it there."
            )
        _fetch_zip(zip_path)

    print(f"  [arxiv] unpacking {zip_path} -> {data_root}", flush=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_root)
    except zipfile.BadZipFile as exc:
        os.remove(zip_path)
        raise RuntimeError(
            f"{zip_path} is not a readable zip ({exc}); it has been deleted. "
            f"Rerun to download it again."
        ) from exc

    raw = _find_raw_dir(data_root)
    if not raw:
        raise RuntimeError(
            f"{zip_path} unpacked but none of {_RAW_FILES} turned up under "
            f"{data_root}. Delete the archive and rerun to re-download."
        )
    return raw


# --------------------------------------------------------------------- parse
def _read_raw(data_root: str, auto_download: bool, verbose: bool) -> Dict[str, np.ndarray]:
    """Node features / labels / years / edges, parsed once and cached as .npy.

    ``node-feat.csv.gz`` is 169,343 x 128 floats behind gzip; parsing it costs a
    few seconds that would otherwise be paid on every run and every seed, and
    memory-mapping the ``.npy`` afterwards costs nothing. The cache is ~87 MB
    for the features and a few MB for the rest.
    """
    cache_dir = os.path.join(data_root, "cache")
    names = {"x": "node_feat.npy", "y": "node_label.npy",
             "year": "node_year.npy", "edge": "edge.npy"}
    paths = {k: os.path.join(cache_dir, v) for k, v in names.items()}
    if all(os.path.exists(p) for p in paths.values()):
        if verbose:
            print(f"  [arxiv] reusing parsed cache in {cache_dir} "
                  f"(no download, no re-parse)")
        return {k: np.load(p, mmap_mode="r" if k == "x" else None)
                for k, p in paths.items()}

    raw = _ensure_raw(data_root, auto_download)
    if verbose:
        print(f"  [arxiv] parsing raw CSVs from {raw} (one-off)", flush=True)

    def _csv(name, dtype):
        return pd.read_csv(os.path.join(raw, name), header=None,
                           dtype=dtype).to_numpy()

    out = {
        "x": _csv("node-feat.csv.gz", np.float32),
        "y": _csv("node-label.csv.gz", np.int64).reshape(-1),
        "year": _csv("node_year.csv.gz", np.int64).reshape(-1),
        "edge": _csv("edge.csv.gz", np.int64),
    }

    n, f = out["x"].shape
    if n != ARXIV_NUM_NODES or f != ARXIV_FEATURE_DIM:
        raise RuntimeError(
            f"unexpected ogbn-arxiv feature matrix {n} x {f}; expected "
            f"{ARXIV_NUM_NODES} x {ARXIV_FEATURE_DIM}. Is {raw} really the "
            f"ogbn-arxiv release?"
        )
    if out["edge"].shape != (ARXIV_NUM_EDGES, 2):
        raise RuntimeError(
            f"unexpected edge list {out['edge'].shape}; expected "
            f"({ARXIV_NUM_EDGES}, 2)."
        )

    os.makedirs(cache_dir, exist_ok=True)
    for k, p in paths.items():
        np.save(p, out[k])
    if verbose:
        print(f"  [arxiv] cached parsed arrays under {cache_dir}")
    return out


# --------------------------------------------------------------------- split
def _adjacency(edge: np.ndarray) -> sp.csr_matrix:
    n = ARXIV_NUM_NODES
    return sp.csr_matrix(
        (np.ones(edge.shape[0], dtype=np.int8), (edge[:, 0], edge[:, 1])),
        shape=(n, n),
    )


def _score_groups(score: np.ndarray, num_groups: int) -> List[np.ndarray]:
    """Cut nodes into `num_groups` near-equal blocks by ascending `score`.

    Boundaries are pushed to the next change in value so nodes sharing a score
    always land in the same graph — GOOD treats each distinct domain value as
    one indivisible domain, and here that is what keeps a publication year from
    straddling two graphs. Mirrors the rule in `src/cora_full_loader.py`; ties
    matter far more here, since `year` takes only 36 distinct values over
    169,343 nodes.
    """
    order = np.lexsort((np.arange(score.size), score))   # ascending, id-stable
    sorted_score = score[order]
    n = order.size
    change = np.flatnonzero(np.diff(sorted_score) != 0) + 1

    groups, start = [], 0
    for g in range(num_groups - 1):
        remaining = num_groups - 1 - g
        ideal = int(round((g + 1) * n / num_groups))
        lo, hi = start + 1, n - remaining
        if lo > hi:
            end = lo
        else:
            cand = change[(change >= lo) & (change <= hi)]
            # Nearest change point, not the next one: rounding up would let one
            # big tie block (2019 alone is 39,711 nodes) swallow several groups'
            # worth of nodes and leave later groups empty.
            end = int(cand[np.argmin(np.abs(cand - ideal))]) if cand.size else \
                int(min(max(ideal, lo), hi))
        groups.append(order[start:end])
        start = end
    groups.append(order[start:])
    return groups


def _year_cut_groups(year: np.ndarray, cuts: Sequence[int]) -> List[np.ndarray]:
    """Six bands from five inclusive upper-year boundaries, oldest first."""
    cuts = list(cuts)
    if len(cuts) != ARXIV_NUM_GRAPHS - 1:
        raise ValueError(
            f"need {ARXIV_NUM_GRAPHS - 1} year cuts, got {len(cuts)}: {cuts}"
        )
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"year cuts must be strictly increasing: {cuts}")

    groups, lo = [], -np.inf
    for hi in list(cuts) + [np.inf]:
        groups.append(np.flatnonzero((year > lo) & (year <= hi)))
        lo = hi
    empty = [i for i, g in enumerate(groups) if g.size == 0]
    if empty:
        raise ValueError(f"year cuts {cuts} leave band(s) {empty} empty")
    return groups


def _make_groups(
    year: np.ndarray, A: sp.csr_matrix, split_by: str,
    year_cuts: Sequence[int] = None, seed: int = 0,
) -> List[np.ndarray]:
    if split_by not in SPLIT_MODES:
        raise ValueError(f"unknown split mode {split_by!r}; expected one of {SPLIT_MODES}")
    if year_cuts is not None:
        return _year_cut_groups(year, year_cuts)
    if split_by == "year_tail":
        return _year_cut_groups(year, YEAR_TAIL_CUTS)
    if split_by == "year":
        return _score_groups(year.astype(np.float64), ARXIV_NUM_GRAPHS)
    if split_by == "degree":            # GOOD's other documented arXiv domain
        S = ((A + A.T) > 0).astype(np.int8)
        S.setdiag(0)
        S.eliminate_zeros()
        deg = np.asarray(S.sum(axis=1)).reshape(-1).astype(np.float64)
        return _score_groups(deg, ARXIV_NUM_GRAPHS)
    # random: structure-blind control
    perm = np.random.default_rng(seed).permutation(year.size)
    return [np.sort(g) for g in np.array_split(perm, ARXIV_NUM_GRAPHS)]


def _induced(
    A: sp.csr_matrix, X: np.ndarray, y: np.ndarray, year: np.ndarray,
    nodes: np.ndarray, symmetrize: bool,
) -> Data:
    """Subgraph induced on `nodes`, renumbered to 0..len(nodes)-1."""
    nodes = np.sort(nodes)
    sub = A[nodes, :][:, nodes].tocoo()
    edge_index = torch.from_numpy(np.vstack([sub.row, sub.col]).astype(np.int64))
    # Drop self-loops; GCNConv adds its own.
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if symmetrize:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

    g = Data(
        x=torch.from_numpy(np.ascontiguousarray(X[nodes], dtype=np.float32)),
        edge_index=edge_index,
        y=torch.from_numpy(y[nodes].astype(np.int64)),
        num_nodes=int(nodes.size),
    )
    g.year_range = (int(year[nodes].min()), int(year[nodes].max()))
    return g


def _build_all(
    data_root: str, symmetrize: bool, auto_download: bool, verbose: bool,
    split_by: str = "year_tail", year_cuts: Sequence[int] = None,
) -> Dict[str, Data]:
    key = (os.path.abspath(data_root), symmetrize, split_by,
           tuple(year_cuts) if year_cuts is not None else None)
    if key in _CACHE:
        return _CACHE[key]

    raw = _read_raw(data_root, auto_download, verbose)
    X, y, year, edge = raw["x"], raw["y"], raw["year"], raw["edge"]
    A = _adjacency(edge)
    if verbose:
        print(f"  [arxiv] {X.shape[0]:,} papers, {X.shape[1]} attributes, "
              f"{len(np.unique(y))} classes, years {year.min()}-{year.max()}; "
              f"split_by={split_by}"
              f"{'' if year_cuts is None else f' cuts={list(year_cuts)}'}")

    groups = _make_groups(year, A, split_by, year_cuts)
    graphs = {}
    for name, nodes in zip(ARXIV_DOMAINS, groups):
        g = _induced(A, X, y, year, nodes, symmetrize)
        g.split_by = split_by
        graphs[name] = g
    _CACHE[key] = graphs
    return graphs


# ----------------------------------------------------------------------- api
def load_arxiv_domain(
    data_root: str, domain: str, symmetrize: bool = True,
    auto_download: bool = True, verbose: bool = False,
    split_by: str = "year_tail", year_cuts: Sequence[int] = None,
) -> Data:
    if domain not in ARXIV_DOMAINS:
        raise ValueError(
            f"Unknown arXiv domain '{domain}'. Choose from {ARXIV_DOMAINS}."
        )
    return _build_all(data_root, symmetrize, auto_download, verbose,
                      split_by, year_cuts)[domain]


def load_sources_target(
    data_root: str, sources: List[str], target: str,
    symmetrize: bool = True, auto_download: bool = True,
    split_by: str = "year_tail", year_cuts: Sequence[int] = None,
) -> Tuple[List[Data], Data]:
    """Same signature as the other loaders, so the FGW runner is symmetric."""
    graphs = _build_all(data_root, symmetrize, auto_download, True,
                        split_by, year_cuts)
    for name in list(sources) + [target]:
        if name not in graphs:
            raise ValueError(
                f"Unknown arXiv domain '{name}'. Choose from {ARXIV_DOMAINS}."
            )
    return [graphs[s] for s in sources], graphs[target]


def split_summary(
    data_root: str, symmetrize: bool = True, year_cuts: Sequence[int] = None,
) -> str:
    """Every split mode next to SelMAG's Table 3 averages, for verification.

    Prints all modes rather than only the active one, because the point is to
    show that none of them reproduces the paper's 66,166 edges — see the module
    docstring. The node column is identical for every mode by construction, so
    the isolated-node column is the one to read: it is what rules out the
    literal `year` cut, whose 2020-only target is 81% isolated nodes.
    """
    lines = []
    for mode in SPLIT_MODES:
        graphs = _build_all(data_root, symmetrize, True, False, mode,
                            year_cuts if mode == "year_tail" else None)
        tot_n = tot_e = 0
        lines.append(f"split_by = {mode}"
                     + ("  (default)" if mode == SPLIT_MODES[0] else ""))
        for name in ARXIV_DOMAINS:
            g = graphs[name]
            # report undirected edges (edge_index holds both directions)
            e = g.edge_index.size(1) // (2 if symmetrize else 1)
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes)
            iso = 100.0 * float((deg == 0).sum()) / g.num_nodes
            lo, hi = g.year_range
            lines.append(f"  {name}: {g.num_nodes:6,} nodes  {e:6,} edges  "
                         f"avg deg {2 * e / g.num_nodes:4.2f}  isolated {iso:4.1f}%  "
                         f"years {lo}-{hi}")
            tot_n += g.num_nodes
            tot_e += e
        k = len(ARXIV_DOMAINS)
        lines.append(f"  average: {tot_n / k:8,.0f} nodes  {tot_e / k:6,.0f} edges"
                     f"   ({2 * tot_e / k:,.0f} if counted doubled)")
        lines.append("")

    lines.append("SelMAG Table 3: 28,223 nodes, 66,166 edges")
    lines.append("")
    lines.append("Node counts match every mode (any 6-way partition of 169,343 "
                 "averages 28,223.8),")
    lines.append("so only the edge column is informative. No year cut reproduces "
                 "66,166: the default")
    lines.append("is 8% high under the doubled convention Table 3 uses for Twitch "
                 "(71,599 vs 66,166)")
    lines.append("and 46% low under the single count. The closest any contiguous "
                 "year cut can get is")
    lines.append("70,474 doubled, with bands <=2011|2012-14|2015-16|2017|2018|"
                 "2019-20 —")
    lines.append("  --split_by year_tail --year_cuts 2011 2014 2016 2017 2018")
    lines.append("")
    lines.append("The literal equal-size cut (--split_by year) makes 2020 the "
                 "target on its own:")
    lines.append("81.5% of its nodes are isolated, so message passing carries "
                 "almost nothing there.")
    return "\n".join(lines)
