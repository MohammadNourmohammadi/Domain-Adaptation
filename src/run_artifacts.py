"""Per-run output directory: log, metrics, history and paper figures.

Every run of any of the five runners lands in its own numbered directory under
``--out_dir`` (default ``runs/``)::

    runs/
      index.csv                      one row per run — the paper's results table
      run_001_twitch_RU/
        log.txt                      the complete stdout of the run
        config.json                  every hyper-parameter + the exact argv
        history.csv                  per-eval-epoch metrics and loss components
        metrics.json                 final per-seed and aggregated scores
        figures/
          run_001_twitch_RU_target_curves.png
          run_001_twitch_RU_loss_components.png
          ...

The run id is incremental and never reused: `allocate_run_dir` scans for the
highest existing `run_NNN_` and claims the next one with an exclusive `mkdir`,
so two runs started at the same moment cannot collide.

Every figure filename carries the full run id, so figures from any set of runs
can be copied into one folder (a paper's `figures/`, say) without collision.

`history.csv` and `metrics.json` are also the *table view* of every figure: the
data-viz rule is that no value may be reachable only by reading a chart, and for
a static paper figure the CSV is that twin.
"""

import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import List, Optional


_RUN_RE = re.compile(r"^run_(\d+)")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")


def allocate_run_dir(root: str, dataset: str, target: str) -> str:
    """Claim the next free `run_NNN_<dataset>_<target>/` under `root`."""
    os.makedirs(root, exist_ok=True)
    used = [
        int(m.group(1))
        for name in os.listdir(root)
        if (m := _RUN_RE.match(name)) and os.path.isdir(os.path.join(root, name))
    ]
    nxt = max(used, default=0) + 1
    while True:
        name = f"run_{nxt:03d}_{_slug(dataset)}_{_slug(target)}"
        path = os.path.join(root, name)
        try:
            # Exclusive: whoever wins the mkdir owns the id.
            os.mkdir(path)
            os.makedirs(os.path.join(path, "figures"), exist_ok=True)
            return path
        except FileExistsError:
            nxt += 1


def run_id(run_dir: str) -> str:
    return os.path.basename(os.path.normpath(run_dir))


class Tee:
    """Mirror stdout/stderr into `path` while still printing to the terminal.

    Used as a context manager around the whole of `main()`, so the saved log is
    the run header, the training trace and the final tables verbatim — the same
    text you would have copied out of the terminal.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._stdout = None
        self._stderr = None

    class _Fan:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
            return len(data)

        def flush(self):
            for s in self.streams:
                s.flush()

        def isatty(self):
            return False

    def __enter__(self):
        self._file = open(self.path, "w", buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self._Fan(self._stdout, self._file)
        sys.stderr = self._Fan(self._stderr, self._file)
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        if self._file:
            self._file.close()
        return False


def save_json(path: str, obj) -> None:
    def _default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def save_history_csv(path: str, per_seed_history: List[List[dict]]) -> None:
    """One row per (seed, eval epoch). The table view of the curve figures."""
    rows = []
    for seed, hist in per_seed_history:
        for rec in hist:
            rows.append({"seed": seed, **rec})
    if not rows:
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def append_index(root: str, row: dict) -> None:
    """Append one summary row to `runs/index.csv`, creating it if needed."""
    path = os.path.join(root, "index.csv")
    exists = os.path.exists(path)
    old_cols = []
    if exists:
        with open(path, newline="") as f:
            r = csv.reader(f)
            old_cols = next(r, [])
    cols = list(old_cols) + [c for c in row if c not in old_cols]
    if cols != old_cols and exists:
        # A new run introduced a column: rewrite with the widened header so the
        # file stays a valid single-schema CSV for pandas / a paper table.
        with open(path, newline="") as f:
            existing = list(csv.DictReader(f))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(existing)
        exists = True
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow(row)


def report_and_save(
    cfg,
    seeds: List[int],
    per_seed: List[dict],
    target,
    dataset: str,
    root: Optional[str] = None,
    make_figures: bool = True,
    run_dir: Optional[str] = None,
) -> dict:
    """Print the final results block, then write every artefact for this run.

    Shared by all five runners so the console table, `metrics.json`, `index.csv`
    and the figures are always the same numbers.
    """
    import statistics

    from .utils import majority_baseline

    print("\n" + "=" * 60)
    print(f"  Final results — {cfg.target_domain}, {len(per_seed)} run(s)")
    print("=" * 60)
    base = majority_baseline(target.y, cfg.num_classes)
    print(f"  majority class : ACC {base['acc']:.4f}  AUROC {base['auc']:.4f}  "
          f"MacroF {base['f1']:.4f}   (no-skill reference)")
    agg = {"majority_acc": float(base["acc"]), "majority_f1": float(base["f1"])}
    for key, label in (("acc", "ACC"), ("auc", "AUROC"), ("f1", "MacroF")):
        vals = [r[key] for r in per_seed]
        agg[f"{key}_mean"] = float(statistics.mean(vals))
        agg[f"{key}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        if len(vals) > 1:
            print(f"  {label:<14}: {agg[f'{key}_mean']:.4f} "
                  f"+- {agg[f'{key}_std']:.4f}   "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")
        else:
            print(f"  {label:<14}: {vals[0]:.4f}")
    print("=" * 60)

    for rec in per_seed:                    # so the curve figure can draw it
        rec.setdefault("majority_f1", float(base["f1"]))

    if run_dir is not None and root is not None:
        finalize(run_dir, root, cfg, seeds, per_seed, agg, dataset,
                 make_figures=make_figures)
    return agg


def finalize(
    run_dir: str,
    root: str,
    cfg,
    seeds: List[int],
    per_seed: List[dict],
    aggregate: dict,
    dataset: str,
    baseline: Optional[dict] = None,
    make_figures: bool = True,
) -> None:
    """Write config/history/metrics, render the figures, update the index."""
    rid = run_id(run_dir)
    cfg_dump = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    save_json(os.path.join(run_dir, "config.json"), {
        "run": rid,
        "dataset": dataset,
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "seeds": seeds,
        "config": cfg_dump,
    })
    save_history_csv(
        os.path.join(run_dir, "history.csv"),
        [(s["seed"], s.get("history", [])) for s in per_seed],
    )
    # The complete per-seed record, including the training trace. `metrics.json`
    # stays the short human-readable summary; this is what `scripts/replot.py`
    # reads to rebuild every figure without re-running the training.
    save_json(os.path.join(run_dir, "trace.json"), {
        "run": rid,
        "dataset": dataset,
        "config": cfg_dump,
        "aggregate": aggregate,
        "per_seed": per_seed,
    })
    save_json(os.path.join(run_dir, "metrics.json"), {
        "run": rid,
        "dataset": dataset,
        "target": cfg.target_domain,
        "sources": list(cfg.source_domains),
        "seeds": seeds,
        "per_seed": [
            {k: v for k, v in s.items() if k not in ("history", "events",
                                                     "selection", "support")}
            for s in per_seed
        ],
        "aggregate": aggregate,
        "baseline": baseline,
    })

    n_fig = 0
    if make_figures:
        try:
            from .plots import render_all
            n_fig = render_all(run_dir, rid, cfg, per_seed, aggregate,
                               dataset, baseline)
        except Exception as exc:                       # never lose a run to a plot
            print(f"  [figures] skipped: {type(exc).__name__}: {exc}")

    row = {
        "run": rid,
        "dataset": dataset,
        "target": cfg.target_domain,
        "sources": "+".join(cfg.source_domains),
        "seeds": len(seeds),
        "epochs": cfg.epochs,
        "acc": round(aggregate.get("acc_mean", float("nan")), 4),
        "acc_std": round(aggregate.get("acc_std", 0.0), 4),
        "auroc": round(aggregate.get("auc_mean", float("nan")), 4),
        "auroc_std": round(aggregate.get("auc_std", 0.0), 4),
        "macro_f1": round(aggregate.get("f1_mean", float("nan")), 4),
        "macro_f1_std": round(aggregate.get("f1_std", 0.0), 4),
        "majority_f1": round(aggregate.get("majority_f1", float("nan")), 4),
        "knn_augment": getattr(cfg, "knn_augment", 0),
        "class_weighted_ce": getattr(cfg, "class_weighted_ce", False),
        "stratified": getattr(cfg, "source_label_stratified", False),
        "selection": getattr(cfg, "model_selection", ""),
        "no_da": getattr(cfg, "no_da", False),
        "when": datetime.now().isoformat(timespec="seconds"),
        "note": getattr(cfg, "note", ""),
    }
    append_index(root, row)

    print("\n" + "-" * 60)
    print(f"  Saved to {run_dir}")
    print("-" * 60)
    print(f"  log.txt      the full console output of this run")
    print(f"  config.json  every hyper-parameter + argv")
    print(f"  history.csv  per-epoch metrics (the table view of the curves)")
    print(f"  metrics.json final scores, per seed and aggregated")
    if n_fig:
        print(f"  figures/     {n_fig} PNGs, each named {rid}_<figure>.png")
    print(f"  {os.path.join(root, 'index.csv')} updated — one row per run")
    print("-" * 60)
