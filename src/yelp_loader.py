"""Yelp POI-graph loader for the FGW prototype-graph DA pipeline.

Reproduces the *Yelp* dataset described in the SelMAG paper (Appendix A):

  * One graph per **city**. Six cities of different scales are used —
    Madison, Glendale, Gilbert, Las Vegas, Toronto, Phoenix — with
    **Phoenix as the target** graph.
  * A **node** is a point-of-interest (POI / business).
  * An **edge** is a *co-review* relationship: two POIs are connected when at
    least ``min_common_reviewers`` distinct users have reviewed both of them.
  * **Node features** are obtained by averaging the pre-trained **GLOVE** word
    embeddings of all the review text a POI received (an F-dim vector, F = the
    GLOVE dimension, e.g. 300 for ``glove.6B.300d``).
  * **Labels** are 5 POI categories: {Food, Shop, Home Service, Health
    Service, Finance}, mapped from Yelp's top-level business categories.

All six city graphs share the same GLOVE feature space and the same 5 classes,
so — exactly like the Citation setting — a single shared encoder transfers
across them and the loader exposes the same
``YELP_CITIES / YELP_FEATURE_DIM / YELP_NUM_CLASSES / load_sources_target`` API
as ``src.citation_loader`` and ``src.data_loader``.

Download policy:

  * On first use, if the raw Yelp dump is missing, it is **auto-downloaded from
    a public Google Drive** via gdown — a ``.tgz`` holding the old-round
    ``business.json`` + ``review.json`` (the SelMAG cities). Drive works from
    regions where Kaggle is IP-blocked (403). Needs ``pip install gdown``. Pass
    ``auto_download=False`` to require a manually placed dump; ``gdrive_id=`` to
    point at a different Drive file. ``data/yelp/README.md`` has the manual step.
  * The raw JSON is read straight from wherever it lands — an extracted ``.json``,
    a ``.zip``, or a ``.tar[.gz]`` — and the multi-GB ``review.json`` is
    **streamed**, never fully extracted to disk.
  * The stream runs **once**: every city graph is parsed and cached to
    ``data/yelp/<City>/processed_fgw.pt``. Every later call just ``torch.load``s
    that cache — no re-download, no re-stream, no re-parse.

Version caveat:
    The Yelp *Open Dataset* changed which metropolitan areas it ships over the
    years. The SelMAG cities (esp. Las Vegas and Toronto) are only in the older
    rounds — which is why the default download points at a Drive-hosted copy of
    that old round rather than the current Yelp/Kaggle release. If a requested
    city yields 0 businesses the loader raises a clear error.
"""

import io
import json
import os
import re
import shutil
import tarfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------- constants
# Folder name (no spaces) -> the set of accepted `city` strings (normalised to
# lowercase, single-spaced) found in the Yelp business records.
YELP_CITY_ALIASES: Dict[str, set] = {
    "Madison": {"madison"},
    "Glendale": {"glendale"},
    "Gilbert": {"gilbert"},
    "LasVegas": {"las vegas"},
    "Toronto": {"toronto"},
    "Phoenix": {"phoenix"},
}
YELP_CITIES = list(YELP_CITY_ALIASES.keys())
YELP_TARGET_CITY = "Phoenix"        # the paper's held-out target graph

YELP_FEATURE_DIM = 300              # GLOVE dim (glove.6B.300d); re-derived from
                                    # the vector file at build time.
YELP_NUM_CLASSES = 5               # {Food, Shop, Home Service, Health, Finance}

# Class index -> the Yelp top-level categories that map onto it. Order matters:
# the first class (top to bottom) whose category set intersects a business's
# categories wins, so overlaps (e.g. a shop inside a restaurant) resolve
# deterministically in the paper's class order.
YELP_CLASS_CATEGORIES: List[Tuple[str, set]] = [
    ("Food",          {"restaurants", "food", "nightlife", "bars",
                        "coffee & tea", "bakeries", "food trucks"}),
    ("Shop",          {"shopping"}),
    ("Home Service",  {"home services", "local services"}),
    ("Health Service", {"health & medical"}),
    ("Finance",       {"financial services"}),
]
YELP_CLASS_NAMES = [name for name, _ in YELP_CLASS_CATEGORIES]

_WORD_RE = re.compile(r"[a-z]+")


# ---------------------------------------------------------------- raw files
# Accepted basenames for each JSON, tolerating the official
# ``yelp_academic_dataset_<kind>.json`` name and the shorter ``<kind>.json``
# used by some mirrors.
_MEMBER_NAMES = {
    "business": ("yelp_academic_dataset_business.json", "business.json"),
    "review": ("yelp_academic_dataset_review.json", "review.json"),
}


def _search_dirs(data_root: str, extra_dirs: Optional[List[str]] = None) -> List[str]:
    # local raw/ + data_root take precedence over any auto-downloaded location.
    return [os.path.join(data_root, "raw"), data_root] + list(extra_dirs or [])


def _locate_member(
    data_root: str, kind: str, extra_dirs: Optional[List[str]] = None,
) -> Optional[Tuple[str, ...]]:
    """Find the business/review JSON as a plain file *or* inside an archive.

    Searches ``data_root/raw``, ``data_root`` and any ``extra_dirs``. Returns a
    descriptor and never extracts the (multi-GB) file:
      ``("file", path)``               – an already-extracted .json
      ``("zip", archive, member)``     – a member of a .zip (e.g. a Kaggle dump)
      ``("tar", archive, member)``     – a member of a .tar / .tar.gz / .tgz
    or ``None`` if nothing matches. Extracted files win over archives so a
    manual unzip always takes precedence.
    """
    names = _MEMBER_NAMES[kind]
    dirs = _search_dirs(data_root, extra_dirs)
    # 1) already-extracted plain files
    for d in dirs:
        for name in names:
            p = os.path.join(d, name)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return ("file", p)
    # 2) inside a .zip (match by basename anywhere in the archive)
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            if fn.lower().endswith(".zip") and zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist():
                        if os.path.basename(member) in names:
                            return ("zip", path, member)
    # 3) inside a .tar / .tar.gz / .tgz (auto-detects gzip)
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            if fn.lower().endswith((".tar", ".tar.gz", ".tgz")):
                # Iterate lazily and stop at the first match: getnames() would
                # decompress the whole (gzip) archive just to list it.
                try:
                    with tarfile.open(path, "r:*") as tf:
                        for member in tf:
                            if os.path.basename(member.name) in names:
                                return ("tar", path, member.name)
                except tarfile.TarError:
                    continue
    return None


@contextmanager
def _open_member(
    data_root: str, kind: str, extra_dirs: Optional[List[str]] = None,
) -> Iterator[Iterator[str]]:
    """Yield a line iterator over the business/review JSON, streaming it out of
    a .zip / .tar[.gz] when it is not already extracted — so the 5 GB
    ``review.json`` never has to land on disk."""
    desc = _locate_member(data_root, kind, extra_dirs)
    if desc is None:
        raise FileNotFoundError(_download_hint(data_root))
    if desc[0] == "file":
        _, path = desc
        print(f"  [yelp] reading {kind} from {path}")
        with open(path, "r", encoding="utf-8") as f:
            yield f
    elif desc[0] == "zip":
        _, archive, member = desc
        print(f"  [yelp] streaming {kind} from {archive}::{member}")
        with zipfile.ZipFile(archive) as zf, zf.open(member) as raw:
            yield io.TextIOWrapper(raw, encoding="utf-8")
    else:  # tar
        _, archive, member = desc
        print(f"  [yelp] streaming {kind} from {archive}::{member}")
        with tarfile.open(archive, "r:*") as tf:
            raw = tf.extractfile(member)
            if raw is None:
                raise FileNotFoundError(
                    f"could not read '{member}' from {archive}")
            yield io.TextIOWrapper(raw, encoding="utf-8")


def _have_raw(data_root: str, extra_dirs: Optional[List[str]] = None) -> bool:
    return all(_locate_member(data_root, k, extra_dirs) is not None
               for k in ("business", "review"))


# ----------------------------------------------------- Google-Drive auto-download
# The SelMAG-cities Yelp dump (old round, Phoenix / Las Vegas / Toronto / …),
# uploaded to a public Google Drive as a .tgz that holds
# yelp_academic_dataset_business.json + ..._review.json. Kaggle is IP-blocked
# from some regions; Drive + gdown works where Kaggle returns 403.
YELP_GDRIVE_ID = "1wzwSjAtQJaQa8hWT9WIHQrTH4m-zk4gO"
_GDRIVE_OUT = "yelp_dataset.tgz"


def _gdrive_download(data_root: str, file_id: str) -> None:
    """Download the Yelp .tgz from Google Drive into ``<data_root>/raw/`` (once).

    Uses gdown (in requirements.txt). The archive lands in ``raw/`` — already a
    search dir — so the loader then streams business.json / review.json straight
    out of the .tgz without ever fully extracting it. Skips if already present.
    """
    try:
        import gdown
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "auto-download needs gdown:  pip install gdown\n"
            "(also listed in requirements.txt). Or place the dump manually — "
            "see data/yelp/README.md."
        ) from e
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out = os.path.join(raw_dir, _GDRIVE_OUT)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    print(f"  [yelp] downloading Yelp dump from Google Drive (id={file_id}) ...",
          flush=True)
    gdown.download(id=file_id, output=out, quiet=False)
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        raise RuntimeError(
            f"Google-Drive download failed (empty {out}). Check the link is "
            f"public and your network, or place the dump manually — see "
            f"{os.path.join(data_root, 'README.md')}.")


def _ensure_raw(
    data_root: str, auto_download: bool, gdrive_id: Optional[str],
) -> List[str]:
    """Make sure the raw JSON is reachable; auto-download it if not.

    Returns a list of extra search dirs (always empty here — the download lands
    in ``raw/`` — kept for signature symmetry with the archive helpers). Raises
    with the manual instructions if the data is missing and ``auto_download`` is
    off (or the download did not contain the expected files)."""
    if _have_raw(data_root):
        return []
    if not auto_download:
        raise FileNotFoundError(_download_hint(data_root))
    _gdrive_download(data_root, gdrive_id or YELP_GDRIVE_ID)
    if not _have_raw(data_root):
        raise FileNotFoundError(
            "download finished but business.json / review.json were not found "
            "inside it. The Drive file should be a .tgz containing "
            "yelp_academic_dataset_business.json and ..._review.json.\n"
            + _download_hint(data_root))
    return []


def _download_hint(data_root: str) -> str:
    raw = os.path.join(data_root, "raw")
    return (
        f"Raw Yelp data not found under {raw} (looked for extracted .json, and "
        f"inside any .zip / .tar[.gz] there or in {data_root}).\n\n"
        f"By default the loader auto-downloads the SelMAG-cities dump (old Yelp "
        f"round: Phoenix / Las Vegas / Toronto / Gilbert / Glendale / Madison) "
        f"from Google Drive via gdown. If that failed, fetch it by hand and drop "
        f"the .tgz in {raw} (leave it as-is — the loader streams from it):\n\n"
        f"  pip install gdown\n"
        f"  gdown {YELP_GDRIVE_ID} -O {os.path.join(raw, _GDRIVE_OUT)}\n"
        f"  # or:  gdown 'https://drive.google.com/uc?id={YELP_GDRIVE_ID}' "
        f"-O {os.path.join(raw, _GDRIVE_OUT)}\n\n"
        f"Any extracted *.json / .zip / .tar[.gz] in {raw} also works.\n"
        f"See {os.path.join(data_root, 'README.md')} for details."
    )


# GLOVE zip mirrored on Hugging Face (stanfordnlp/glove) — reachable from
# regions where nlp.stanford.edu is blocked/slow. Holds glove.6B.{50,100,200,300}d.
YELP_GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"


def _default_glove_path(data_root: str) -> str:
    return os.path.join(data_root, "glove", "glove.6B.300d.txt")


def _glove_hint(glove_path: str) -> str:
    glove_dir = os.path.dirname(glove_path) or "."
    return (
        f"GLOVE embedding file not found at {glove_path}.\n"
        f"POI features are averaged GLOVE word vectors, so the vectors are "
        f"needed once to build the graphs. Fetch them by hand (macOS has no "
        f"wget — use curl; the Hugging Face mirror works where Stanford is "
        f"blocked):\n\n"
        f"  mkdir -p {glove_dir}\n"
        f"  curl -L {YELP_GLOVE_URL} \\\n"
        f"    -o {os.path.join(glove_dir, 'glove.6B.zip')}\n"
        f"  unzip {os.path.join(glove_dir, 'glove.6B.zip')} -d {glove_dir}\n\n"
        f"That yields glove.6B.{{50,100,200,300}}d.txt; pass a different one "
        f"with `glove_path=` (or the --glove_path flag) to change the dim."
    )


def _download_glove(glove_path: str, url: Optional[str] = None) -> None:
    """Download the GLOVE zip and extract just the needed dim into ``glove_path``.

    Streams ``glove.6B.zip`` (~862 MB) from the Hugging Face mirror, then pulls
    only the single ``glove.6B.<dim>d.txt`` member matching ``glove_path``'s
    basename — so we don't unpack all four dimensions. The zip is kept so a later
    dim change doesn't re-download."""
    import urllib.request

    url = url or YELP_GLOVE_URL
    glove_dir = os.path.dirname(glove_path) or "."
    os.makedirs(glove_dir, exist_ok=True)
    member = os.path.basename(glove_path)          # e.g. glove.6B.300d.txt
    zip_path = os.path.join(glove_dir, os.path.basename(url.split("?")[0]))

    if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 0):
        print(f"  [yelp] downloading GLOVE from {url} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                chunk = 1 << 20  # 1 MB
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    out.write(buf)
                    done += len(buf)
                    if done % (50 * chunk) < chunk:
                        tot = f"/{total/1e6:.0f}" if total else ""
                        print(f"    ... {done/1e6:.0f}{tot} MB", flush=True)
        except Exception as e:  # noqa: BLE001 - surface a clean, actionable hint
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(
                f"GLOVE download from {url} failed ({e}). If Hugging Face is "
                f"blocked, download glove.6B.zip by hand into {glove_dir} "
                f"(see data/yelp/README.md) — or pass --glove_url with a mirror."
            ) from e

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if os.path.basename(n) == member]
        if not names:
            raise RuntimeError(
                f"'{member}' not found in {zip_path}. Members: {zf.namelist()}. "
                f"Pass --glove_path matching one of them."
            )
        with zf.open(names[0]) as src, open(glove_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"  [yelp] GLOVE ready at {glove_path}", flush=True)


# ------------------------------------------------------------- parsing bits
def _city_lookup() -> Dict[str, str]:
    """Map each accepted (normalised) city string -> its folder name."""
    out: Dict[str, str] = {}
    for folder, aliases in YELP_CITY_ALIASES.items():
        for a in aliases:
            out[a] = folder
    return out


def _norm_city(raw: str) -> str:
    return re.sub(r"\s+", " ", (raw or "").strip().lower())


def _category_to_class(categories: Optional[str]) -> Optional[int]:
    """Map a Yelp comma-separated category string to a class index (or None)."""
    if not categories:
        return None
    toks = {c.strip().lower() for c in categories.split(",")}
    for idx, (_name, cat_set) in enumerate(YELP_CLASS_CATEGORIES):
        if toks & cat_set:
            return idx
    return None


def _load_glove(glove_path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """Load GLOVE vectors into ``{word: vec}`` and return the dimension."""
    print(f"  [yelp] loading GLOVE vectors from {glove_path} ...", flush=True)
    vecs: Dict[str, np.ndarray] = {}
    dim = None
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if dim is None:
                dim = vec.shape[0]
            vecs[word] = vec
    if dim is None:
        raise RuntimeError(f"no vectors parsed from {glove_path}")
    print(f"  [yelp] loaded {len(vecs):,} GLOVE vectors (dim={dim})")
    return vecs, dim


def _build_and_cache_all(
    data_root: str,
    glove_path: str,
    min_common_reviewers: int,
    max_user_degree: int,
    symmetrize: bool,
    auto_download: bool = True,
    gdrive_id: Optional[str] = None,
    glove_url: Optional[str] = None,
) -> None:
    """Stream the raw Yelp dump once and cache every city graph.

    Writes ``<data_root>/<City>/processed_fgw.pt`` for each of ``YELP_CITIES``.
    Auto-downloads the raw dump (.tgz with the paper's cities) from Google Drive
    via gdown, and the GLOVE vectors from the Hugging Face mirror, if either is
    missing; ``gdrive_id`` / ``glove_url`` override the defaults.
    """
    extra_dirs = _ensure_raw(data_root, auto_download, gdrive_id)
    if not os.path.exists(glove_path):
        if auto_download:
            _download_glove(glove_path, glove_url)
        else:
            raise FileNotFoundError(_glove_hint(glove_path))

    glove, dim = _load_glove(glove_path)
    city_of = _city_lookup()

    # --- pass 1: businesses -> keep those in our cities with a mappable class.
    # business_id -> (folder, local_idx); per-folder label list.
    print("  [yelp] scanning businesses ...", flush=True)
    biz_folder: Dict[str, str] = {}
    biz_local: Dict[str, int] = {}
    labels: Dict[str, List[int]] = {c: [] for c in YELP_CITIES}
    with _open_member(data_root, "business", extra_dirs) as f:
        for line in f:
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            folder = city_of.get(_norm_city(b.get("city", "")))
            if folder is None:
                continue
            cls = _category_to_class(b.get("categories"))
            if cls is None:
                continue
            bid = b["business_id"]
            biz_folder[bid] = folder
            biz_local[bid] = len(labels[folder])
            labels[folder].append(cls)
    counts = {c: len(labels[c]) for c in YELP_CITIES}
    print(f"  [yelp] kept POIs per city: {counts}")

    # --- feature accumulators (running GLOVE sum + token count) per business.
    feat_sum: Dict[str, np.ndarray] = {
        bid: np.zeros(dim, dtype=np.float64) for bid in biz_folder
    }
    tok_count: Dict[str, int] = defaultdict(int)
    # per city: user -> set(local business idx) reviewed in that city.
    user_biz: Dict[str, Dict[str, set]] = {c: defaultdict(set) for c in YELP_CITIES}

    # --- pass 2: reviews -> features + co-review user sets.
    print("  [yelp] streaming reviews (this is the big one) ...", flush=True)
    n_seen = 0
    with _open_member(data_root, "review", extra_dirs) as rf:
        for line in rf:
            n_seen += 1
            if n_seen % 1_000_000 == 0:
                print(f"    ... {n_seen:,} reviews scanned", flush=True)
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = r.get("business_id")
            folder = biz_folder.get(bid)
            if folder is None:
                continue
            # feature: accumulate GLOVE embeddings of the review tokens.
            acc = feat_sum[bid]
            cnt = 0
            for tok in _WORD_RE.findall((r.get("text") or "").lower()):
                v = glove.get(tok)
                if v is not None:
                    acc += v
                    cnt += 1
            tok_count[bid] += cnt
            # co-review: remember which POIs this user touched in this city.
            uid = r.get("user_id")
            if uid is not None:
                user_biz[folder][uid].add(biz_local[bid])
    print(f"  [yelp] finished streaming {n_seen:,} reviews")

    # --- build + cache each city graph.
    for folder in YELP_CITIES:
        n = counts[folder]
        if n == 0:
            print(f"  [yelp][WARN] '{folder}' has 0 POIs — skipping "
                  f"(likely absent in this Yelp release; see README).")
            continue

        # features
        x = np.zeros((n, dim), dtype=np.float32)
        # labels
        y = np.asarray(labels[folder], dtype=np.int64)
        # map local idx -> business_id to fill features
        # (rebuild the reverse index cheaply)
        for bid, fol in biz_folder.items():
            if fol != folder:
                continue
            li = biz_local[bid]
            c = tok_count[bid]
            if c > 0:
                x[li] = (feat_sum[bid] / c).astype(np.float32)

        # edges: count common reviewers per unordered POI pair.
        pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
        for _uid, biz_set in user_biz[folder].items():
            if len(biz_set) < 2 or len(biz_set) > max_user_degree:
                continue
            items = sorted(biz_set)
            for a_i in range(len(items)):
                ia = items[a_i]
                for b_i in range(a_i + 1, len(items)):
                    pair_count[(ia, items[b_i])] += 1

        edges = [p for p, c in pair_count.items() if c >= min_common_reviewers]
        if edges:
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            if symmetrize:
                ei = torch.cat([ei, ei.flip(0)], dim=1)
        else:
            ei = torch.empty(2, 0, dtype=torch.long)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=ei,
            y=torch.from_numpy(y),
            num_nodes=n,
        )
        city_dir = os.path.join(data_root, folder)
        os.makedirs(city_dir, exist_ok=True)
        cache_path = os.path.join(city_dir, "processed_fgw.pt")
        torch.save(data, cache_path)
        print(f"  [yelp] {folder}: {n} POIs, {ei.size(1)} edges -> cached {cache_path}")


# --------------------------------------------------------------- public API
def load_yelp_city(
    data_root: str,
    city: str,
    glove_path: Optional[str] = None,
    min_common_reviewers: int = 1,
    max_user_degree: int = 100,
    symmetrize: bool = True,
    auto_download: bool = True,
    gdrive_id: Optional[str] = None,
    glove_url: Optional[str] = None,
) -> Data:
    """Load one Yelp city graph, building + caching all cities on first use.

    ``<data_root>/<city>/processed_fgw.pt`` is the cache. If it exists it is
    loaded directly; otherwise the raw dump is downloaded (from Google Drive via
    gdown, when ``auto_download`` and it is missing), streamed once, every city
    graph is built and cached, and the requested one is returned.
    """
    if city not in YELP_CITY_ALIASES:
        raise ValueError(f"Unknown Yelp city '{city}'. Choose from {YELP_CITIES}.")

    cache_path = os.path.join(data_root, city, "processed_fgw.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    glove_path = glove_path or _default_glove_path(data_root)
    print(f"  [{city}] cache missing -> building all Yelp city graphs "
          f"(one-time, streams the raw dump)")
    _build_and_cache_all(
        data_root, glove_path, min_common_reviewers, max_user_degree, symmetrize,
        auto_download=auto_download, gdrive_id=gdrive_id, glove_url=glove_url,
    )
    if not os.path.exists(cache_path):
        raise RuntimeError(
            f"'{city}' graph was not produced — it is likely absent in this "
            f"Yelp release. See {os.path.join(data_root, 'README.md')} for the "
            f"version caveat (Las Vegas / Toronto need an older round)."
        )
    return torch.load(cache_path, weights_only=False)


def load_sources_target(
    data_root: str,
    sources: List[str],
    target: str,
    glove_path: Optional[str] = None,
    min_common_reviewers: int = 1,
    max_user_degree: int = 100,
    symmetrize: bool = True,
    auto_download: bool = True,
    gdrive_id: Optional[str] = None,
    glove_url: Optional[str] = None,
) -> Tuple[List[Data], Data]:
    """Load several source city graphs and one target city graph.

    Same signature contract as ``src.citation_loader.load_sources_target`` so
    the FGW runner is symmetric across the Twitch / Citation / Yelp settings.
    """
    kwargs = dict(
        glove_path=glove_path,
        min_common_reviewers=min_common_reviewers,
        max_user_degree=max_user_degree,
        symmetrize=symmetrize,
        auto_download=auto_download,
        gdrive_id=gdrive_id,
        glove_url=glove_url,
    )
    src_graphs = [load_yelp_city(data_root, s, **kwargs) for s in sources]
    tgt_graph = load_yelp_city(data_root, target, **kwargs)
    return src_graphs, tgt_graph
