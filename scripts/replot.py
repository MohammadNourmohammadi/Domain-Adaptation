"""Rebuild a run's figures from its saved trace — no retraining.

Every run writes `trace.json` (the full per-seed record: metrics, training
history, phase events, selection weights, ego-graph support). This script
re-renders the figures from it, so tweaking a caption, a colour or a limit for
the paper costs a second rather than a 300-epoch rerun.

    python scripts/replot.py                     # every run under runs/
    python scripts/replot.py runs/run_003_*      # named runs
    python scripts/replot.py --out_dir results   # a different root
"""

import argparse
import glob
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plots import render_all              # noqa: E402


def replot(run_dir: str) -> int:
    trace_path = os.path.join(run_dir, "trace.json")
    if not os.path.exists(trace_path):
        print(f"  {os.path.basename(run_dir)}: no trace.json "
              f"(run predates it) — skipped")
        return 0
    with open(trace_path) as f:
        t = json.load(f)
    cfg = types.SimpleNamespace(**t["config"])
    n = render_all(run_dir, t["run"], cfg, t["per_seed"], t["aggregate"],
                   t["dataset"])
    print(f"  {t['run']}: {n} figures")
    return n


def main():
    p = argparse.ArgumentParser(description="rebuild run figures from trace.json")
    p.add_argument("runs", nargs="*", help="run directories (default: all)")
    p.add_argument("--out_dir", default="runs", help="root holding the runs")
    args = p.parse_args()

    dirs = args.runs or sorted(
        d for d in glob.glob(os.path.join(args.out_dir, "run_*"))
        if os.path.isdir(d)
    )
    if not dirs:
        print(f"no runs found under {args.out_dir}/")
        return
    total = sum(replot(d) for d in dirs)
    print(f"\n{total} figures rebuilt across {len(dirs)} run(s)")


if __name__ == "__main__":
    main()
