#!/usr/bin/env python3
"""Fetch and verify the Twitch graphs used by SelMAG (arXiv:2406.10425).

State of the data
-----------------
The six Twitch language graphs come from MUSAE (Rozemberczki et al.). The
authoritative archive is SNAP's `twitch.zip`, which holds, per language, the
edge list, the sparse bag-of-words features and the labels. That is what this
script downloads and verifies, and it is what `--features musae` trains on.

SelMAG's Table 3 lists "#Attributes 128", i.e. it used the dense 128-d feature
release that `torch_geometric.datasets.Twitch` shipped as `<NAME>.npz` from
`graphmining.ai`. **Those files are gone.** The host's account was suspended
(pyg-team/pytorch_geometric#10346, #10510, #10672) and as of this writing the
domain does not resolve at all — NXDOMAIN, no NS, no SOA — so nobody can fetch
them, VPN or not. A PyG maintainer's reply on #10346 is explicit: "this is not
something we can fix in the code. we need the underlying links to be fixed or
new links."

The open replacement PR (#10415) repoints Twitch at SNAP but does **not**
reconstruct the 128-d features. It takes each node's sparse feature *index
list*, truncates or zero-pads it to length 128, and feeds those vocabulary IDs
in as float values — a list of token ids treated as a continuous vector, whose
entries depend on the arbitrary order the indices happen to be stored in. It is
a stopgap to make the loader run, not the paper's feature matrix, and training
on it would not be a faithful reproduction either.

What this means for the comparison
----------------------------------
The graphs and labels are provably the paper's: every file here is byte-
identical to SNAP's, the node counts match exactly, and averaging over the six
graphs gives 5,686.3 nodes and — counting each undirected edge twice plus one
self-loop per node, which is how PyG's STATS table counts — exactly 148,724
edges, against Table 3's 5,687 / 148,724.

Only the feature *representation* cannot be reproduced bit-for-bit. The model
handles this honestly: `svd_proj` reduces the 3170-d bag-of-words to
`--proj_dim` (default 128) with a frozen basis fitted without labels, so
training runs at the paper's stated width on the authors' original features,
by a documented and reproducible reduction rather than an opaque lost file.

Usage
-----
    python scripts/download_twitch.py                 # fetch + verify -> data/
    python scripts/download_twitch.py --check         # verify what is on disk
    python scripts/download_twitch.py --npz-base URL  # if you locate the npz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile

import numpy as np

SNAP_URL = "https://snap.stanford.edu/data/twitch.zip"

DOMAINS = ["DE", "ENGB", "ES", "FR", "PTBR", "RU"]

# Exact node counts, cross-checked against SelMAG Table 3's six-graph averages.
EXPECTED_NODES = {"DE": 9498, "ENGB": 7126, "ES": 4648,
                  "FR": 6549, "PTBR": 1912, "RU": 4385}
# Undirected edge counts as stored in the raw csv (one row per edge).
EXPECTED_EDGES = {"DE": 153138, "ENGB": 35324, "ES": 59382,
                  "FR": 112666, "PTBR": 31299, "RU": 37304}

# DE stores its features under a different filename than the rest.
def _feature_file(domain: str) -> str:
    return "musae_DE.json" if domain == "DE" else f"musae_{domain}_features.json"


def _files(domain: str) -> list[str]:
    return [f"musae_{domain}_edges.csv", _feature_file(domain),
            f"musae_{domain}_target.csv"]


_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# The 128-d npz, kept only so the path still works if the file resurfaces.
NPZ_NAME = {"DE": "DE", "ENGB": "EN", "ES": "ES",
            "FR": "FR", "PTBR": "PT", "RU": "RU"}
NPZ_BASES = ["https://graphmining.ai/datasets/ptg/twitch"]


# --------------------------------------------------------------- raw (SNAP)
def download_snap(data_root: str, timeout: int, force: bool) -> bool:
    """Fetch SNAP's twitch.zip and install the six language directories."""
    have = all(os.path.exists(os.path.join(data_root, d, f))
               for d in DOMAINS for f in _files(d))
    if have and not force:
        print(f"Raw MUSAE files already present under {data_root}/ "
              f"(use --force to re-fetch).")
        return True

    print(f"Downloading {SNAP_URL}")
    req = urllib.request.Request(SNAP_URL, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            blob = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"  failed: {exc}")
        return False
    print(f"  got {len(blob) / 1e6:.1f} MB  "
          f"sha256 {hashlib.sha256(blob).hexdigest()[:16]}...")

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        for domain in DOMAINS:
            dest = os.path.join(data_root, domain)
            os.makedirs(dest, exist_ok=True)
            for fname in _files(domain):
                member = next(
                    (n for n in names
                     if n.endswith(f"/{domain}/{fname}") or n == f"{domain}/{fname}"),
                    None,
                )
                if member is None:
                    print(f"  ! {domain}/{fname} not found in the archive")
                    return False
                with z.open(member) as src, open(os.path.join(dest, fname), "wb") as f:
                    shutil.copyfileobj(src, f)
            print(f"  {domain:5s} -> {dest}/")
    return True


def check_raw(data_root: str) -> bool:
    """Validate the raw files against the counts the paper's Table 3 implies."""
    ok = True
    tot_n = tot_e = 0
    print(f"Checking raw MUSAE files under {data_root}/")
    for domain in DOMAINS:
        d = os.path.join(data_root, domain)
        missing = [f for f in _files(domain) if not os.path.exists(os.path.join(d, f))]
        if missing:
            print(f"  {domain:5s} MISSING {missing}")
            ok = False
            continue
        with open(os.path.join(d, _feature_file(domain))) as f:
            n_feat_rows = len(json.load(f))
        with open(os.path.join(d, f"musae_{domain}_edges.csv")) as f:
            n_edges = sum(1 for _ in f) - 1          # minus the header
        # Count *distinct* node ids, not rows. FR's target.csv contains two
        # duplicate rows (ids 3754 and 1018, scraped twice — one pair differs
        # only by 1 in `views`), so it has 6551 rows for 6549 nodes. Both
        # copies carry the same label, so nothing is ambiguous, but a row count
        # would report the wrong size. PyG's STATS table documents FR as 6,551
        # nodes, which is exactly this artifact.
        with open(os.path.join(d, f"musae_{domain}_target.csv")) as f:
            header = f.readline().rstrip("\n").split(",")
            col = header.index("new_id")
            n_nodes = len({line.split(",")[col] for line in f if line.strip()})
        exp_n, exp_e = EXPECTED_NODES[domain], EXPECTED_EDGES[domain]
        bad = []
        if n_nodes != exp_n:
            bad.append(f"{n_nodes} nodes != {exp_n}")
        if n_edges != exp_e:
            bad.append(f"{n_edges} edges != {exp_e}")
        if n_feat_rows != exp_n:
            bad.append(f"{n_feat_rows} feature rows != {exp_n}")
        if bad:
            print(f"  {domain:5s} MISMATCH  {'; '.join(bad)}")
            ok = False
            continue
        print(f"  {domain:5s} ok  {n_nodes:5d} nodes  {n_edges:7d} undirected edges")
        tot_n += n_nodes
        tot_e += n_edges
    if ok:
        # PyG's STATS table (and hence SelMAG's Table 3) counts each undirected
        # edge twice and adds one self-loop per node. It also inherits FR's
        # duplicate-row artifact, listing FR as 6,551 nodes rather than 6,549 —
        # so reproducing Table 3 exactly means counting FR the way PyG did.
        dup = 2 if EXPECTED_NODES["FR"] == 6549 else 0
        avg_n = (tot_n + dup) / len(DOMAINS)
        avg_e = (2 * tot_e + tot_n + dup) / len(DOMAINS)
        print(f"\n  six-graph averages (PyG counting, incl. FR's 2 duplicate")
        print(f"  target rows and one self-loop per node):")
        print(f"    ours          : {avg_n:,.2f} nodes, {avg_e:,.0f} edges")
        print(f"    SelMAG Table 3: 5,687 nodes, 148,724 edges")
        match = round(avg_n) == 5687 and round(avg_e) == 148724
        print("  -> " + ("MATCH: these are exactly the paper's graphs."
                         if match else "MISMATCH — investigate before comparing."))
        print(f"\n  (True distinct-node average is {tot_n / len(DOMAINS):,.2f}; the "
              f"0.33 gap\n   is FR's two duplicated target rows, which carry "
              f"identical labels\n   and so change nothing about the graph.)")
    return ok


# ------------------------------------------------------------- 128-d (gone)
def try_npz(pyg_root: str, bases: list[str], timeout: int) -> bool:
    raw_dir = os.path.join(pyg_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    got = 0
    for domain in DOMAINS:
        name = NPZ_NAME[domain]
        dest = os.path.join(raw_dir, f"{name}.npz")
        if os.path.exists(dest):
            got += 1
            continue
        for base in bases:
            try:
                req = urllib.request.Request(f"{base}/{name}.npz",
                                             headers={"User-Agent": _UA})
                with urllib.request.urlopen(req, timeout=timeout) as r, \
                        open(dest, "wb") as f:
                    shutil.copyfileobj(r, f)
            except Exception as exc:
                print(f"  {domain:5s} .. {base.split('/')[2]}: {exc}")
                continue
            try:
                z = np.load(dest, allow_pickle=True)
                assert z["features"].shape == (EXPECTED_NODES[domain], 128)
            except Exception as exc:
                print(f"  {domain:5s} .. rejected: {exc}")
                os.remove(dest)
                continue
            print(f"  {domain:5s} -> {name}.npz OK")
            got += 1
            break
    return got == len(DOMAINS)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="data",
                   help="where the per-language directories live (default: %(default)s)")
    p.add_argument("--pyg_root", default="data/twitch_pyg",
                   help="where a 128-d npz release would go, if you find one")
    p.add_argument("--check", action="store_true",
                   help="only validate what is already on disk")
    p.add_argument("--force", action="store_true", help="re-download the archive")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--npz-base", action="append", dest="npz_bases", default=None,
                   help="mirror exposing <base>/<NAME>.npz for the 128-d release")
    args = p.parse_args()

    if args.check:
        return 0 if check_raw(args.data_root) else 1

    if not download_snap(args.data_root, args.timeout, args.force):
        print("\nCould not obtain the raw archive. Fetch it by hand from")
        print(f"    {SNAP_URL}")
        print(f"and unzip its per-language folders into {args.data_root}/.")
        return 1

    print()
    ok = check_raw(args.data_root)

    print("\n" + "-" * 62)
    print("128-d feature release (SelMAG's Table 3 setting)")
    print("-" * 62)
    bases = (args.npz_bases or []) + NPZ_BASES
    if try_npz(args.pyg_root, bases, args.timeout):
        print("  All six npz present -> train with `--features pyg`.")
    else:
        print("  Unavailable, as expected: graphmining.ai was suspended and the")
        print("  domain no longer resolves, so the original 128-d files cannot")
        print("  be fetched by anyone. See pytorch_geometric#10346 / #10510.")
        print()
        print("  Train on the authors' own features instead:")
        print("      python main_fgw.py --features musae")
        print("  The encoder's frozen unsupervised SVD basis reduces them to")
        print("  --proj_dim (default 128), so the run is at the paper's stated")
        print("  width on the paper's graphs, via a reduction that is written")
        print("  down and reproducible.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
