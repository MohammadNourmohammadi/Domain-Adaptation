"""Paper figures for one run.

Every figure is written as a 300 dpi `.png` named ``<run_id>_<figure>.png``, so
figures from different runs can be copied into one directory (a paper's
`figures/` folder, say) without any of them overwriting another.

Design rules applied (they are not cosmetic — they are what makes a figure
readable in print and to a colour-blind reader):

* **Categorical hues in fixed slot order, never cycled.** The eight-slot order
  below is validated: worst adjacent CVD ΔE 9.1 and worst adjacent
  normal-vision ΔE 19.6 on the light surface, i.e. every neighbouring pair stays
  distinguishable under protanopia/deuteranopia and in greyscale print.
* **One y-axis per plot, never two.** Two scales on one frame invent a
  correlation the data does not contain. Where two quantities live on different
  scales they get separate panels.
* **Secondary encoding everywhere colour carries meaning**: distinct marker
  shapes and dash patterns per series, plus a legend and selective end-labels.
  Three of the palette slots sit below 3:1 contrast on white, so the "relief
  rule" applies — labels and the CSV twin (`history.csv`, `metrics.json`) carry
  every value that the colour alone would otherwise encode.
* **Sequential = one hue light→dark** (the confusion matrix). Never a rainbow.
* Hairline solid grid, no top/right spines, no value printed on every point.
"""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")                       # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# --------------------------------------------------------------- design tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = [(None, None), (5, 2), (1, 1.6), (7, 2, 1.5, 2),
          (3, 1.5), (9, 2), (2, 1.2), (6, 2, 2, 2)]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
PHASE_BAND = {"warmup": "#f2f1ec", "adapt": "#e8eef7", "refine": "#f6eee9"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.dpi": 120,
})


def _style(ax, title=None, xlabel=None, ylabel=None, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def _save(fig, run_dir, rid, name):
    out = os.path.join(run_dir, "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{rid}_{name}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 1


def _markevery(n, target=12):
    return max(1, n // target)


def _legend_below(ax, ncol=3, y=-0.26):
    """Legend outside the plot area.

    Inside-the-axes legends collided with the curves and the annotations on
    every real run; below the x-label there is nothing to collide with, and it
    keeps the plotting area free for the data.
    """
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     handlelength=2.4, columnspacing=1.6)


def _note(ax, xy, text, color=MUTED, xytext=(4, 4), ha="left", va="bottom"):
    """Annotation on an opaque surface patch so it never sits on a mark."""
    return ax.annotate(
        text, xy, textcoords="offset points", xytext=xytext, ha=ha, va=va,
        fontsize=7.5, color=color, zorder=6,
        bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                  edgecolor="none", alpha=0.85),
    )


def _phase_bands(ax, events, xmax):
    """Shade warm-up / adapt / refine and label each band once."""
    bounds, order = [], ["warmup", "adapt", "refine"]
    start = 1
    for ev in events:
        bounds.append((start, ev["epoch"], ev.get("from", "")))
        start = ev["epoch"]
    bounds.append((start, xmax, events[-1].get("to", "") if events else "warmup"))
    for lo, hi, phase in bounds:
        if hi <= lo or phase not in PHASE_BAND:
            continue
        ax.axvspan(lo, hi, color=PHASE_BAND[phase], zorder=0, linewidth=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], f" {phase} ",
                ha="center", va="top", fontsize=7.5, color=MUTED)
    _ = order


# ------------------------------------------------------------------- figures
def fig_target_curves(run_dir, rid, seed_runs, cfg):
    """Target-graph scores against epoch, with the training phases shaded."""
    hist = seed_runs[0].get("history") or []
    if len(hist) < 2:
        return 0
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    me = _markevery(len(ep))
    for i, (key, label) in enumerate(
        [("tgt_acc", "Accuracy"), ("tgt_f1", "Macro-F1"), ("tgt_auc", "AUROC")]
    ):
        y = [h.get(key, float("nan")) for h in hist]
        ax.plot(ep, y, color=SERIES[i], marker=MARKERS[i], markevery=me,
                dashes=DASHES[i], label=label, zorder=3)
        ax.annotate(f"{y[-1]:.3f}", (ep[-1], y[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8, color=SERIES[i])
    _style(ax, f"Target-graph performance — {cfg.target_domain}",
           "epoch", "score")
    ax.set_xlim(min(ep), max(ep) * 1.10)
    _phase_bands(ax, seed_runs[0].get("events") or [], max(ep))

    maj = seed_runs[0].get("majority_f1")
    if maj is not None:
        ax.axhline(maj, color=MUTED, linewidth=1.0, dashes=(2, 2), zorder=1)
        _note(ax, (max(ep), maj), "majority-class Macro-F1",
              xytext=(-2, 4), ha="right")
    sel = seed_runs[0].get("best_epoch")
    if sel:
        ax.axvline(sel, color=INK_2, linewidth=1.0, zorder=2)
        _note(ax, (sel, ax.get_ylim()[1]), f"reported\nepoch {sel}", INK_2,
              xytext=(4, -4), va="top")
    _legend_below(ax, ncol=3)
    return _save(fig, run_dir, rid, "target_curves")


def fig_loss_components(run_dir, rid, seed_runs, cfg):
    """Each loss term against epoch. Two panels — never two y-scales on one."""
    hist = seed_runs[0].get("history") or []
    if len(hist) < 2:
        return 0
    ep = [h["epoch"] for h in hist]
    sup = [("L_cls", "L_cls (source CE)"), ("L_proto", "L_proto"),
           ("L_margin", "L_margin")]
    tgt = [("L_align", "L_align"), ("L_ent", "L_ent (IM)"), ("L_pl", "L_pl")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True)
    me = _markevery(len(ep))
    for ax, group, title, slot0 in (
        (axes[0], sup, "Supervised terms (sources)", 0),
        (axes[1], tgt, "Transfer terms (target)", 3),
    ):
        drawn = 0
        for i, (key, label) in enumerate(group):
            y = [h.get(key, float("nan")) for h in hist]
            if not np.isfinite(y).any() or np.allclose(np.nan_to_num(y), 0):
                continue
            s = slot0 + i
            ax.plot(ep, y, color=SERIES[s], marker=MARKERS[s], markevery=me,
                    dashes=DASHES[s], label=label, zorder=3)
            drawn += 1
        _style(ax, title, "epoch", "loss")
        if drawn:
            _legend_below(ax, ncol=min(drawn, 3), y=-0.20)
        else:
            # A short run can end before adaptation starts, so every target
            # term is identically zero. Say so rather than leaving a blank
            # frame that reads as a rendering bug.
            ax.text(0.5, 0.5, "never active in this run\n(warm-up did not end)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, color=MUTED)
            ax.set_yticks([])
    ch = seed_runs[0].get("proto_chance")
    if ch:
        axes[0].axhline(ch, color=MUTED, linewidth=1.0, dashes=(2, 2))
        _note(axes[0], (ep[-1], ch), f"ln C = {ch:.2f} (chance)",
              xytext=(-2, 4), ha="right")
    fig.suptitle(f"Loss components — {cfg.target_domain}", x=0.008, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, run_dir, rid, "loss_components")


def fig_selection_signal(run_dir, rid, seed_runs, cfg):
    """What the checkpoint criterion sees vs. what it is trying to maximise.

    Both series are scores in [0, 1], so they share one axis honestly. The gap
    between the chosen epoch and the (unusable) target oracle is the figure's
    point: it quantifies what honest model selection costs.
    """
    hist = seed_runs[0].get("history") or []
    if len(hist) < 2:
        return 0
    ep = [h["epoch"] for h in hist]
    val = [h.get("src_val", float("nan")) for h in hist]
    tgt = [h.get("tgt_f1", float("nan")) for h in hist]
    if not np.isfinite(val).any():
        return 0
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    me = _markevery(len(ep))
    lbl = seed_runs[0].get("val_label", "held-out source score")
    ax.plot(ep, val, color=SERIES[0], marker=MARKERS[0], markevery=me,
            dashes=DASHES[0], label=f"{lbl} (selection signal)", zorder=3)
    ax.plot(ep, tgt, color=SERIES[1], marker=MARKERS[1], markevery=me,
            dashes=DASHES[1], label="target Macro-F1 (never seen)", zorder=3)
    _style(ax, "Model selection uses no target label", "epoch", "score")
    ax.set_xlim(min(ep), max(ep) * 1.06)
    sel, orc = seed_runs[0].get("best_epoch"), seed_runs[0].get("oracle_epoch")
    for e, color, text, dy in ((orc, SERIES[1], "oracle", 12),
                               (sel, SERIES[0], "selected", -16)):
        if not e:
            continue
        j = min(range(len(ep)), key=lambda i: abs(ep[i] - e))
        ax.scatter([ep[j]], [tgt[j]], s=90, facecolor=SURFACE,
                   edgecolor=color, linewidth=2.0, zorder=5)
        right = ep[j] > 0.6 * max(ep)
        _note(ax, (ep[j], tgt[j]), f"{text}: ep {e}, F1 {tgt[j]:.3f}", color,
              xytext=(-8 if right else 8, dy), ha="right" if right else "left",
              va="center")
    _legend_below(ax, ncol=2)
    return _save(fig, run_dir, rid, "selection_signal")


def fig_source_weights(run_dir, rid, seed_runs, cfg):
    """Learned per-domain transferability s_global over training."""
    sel = seed_runs[0].get("selection") or []
    if len(sel) < 2:
        return 0
    ep = [s["epoch"] for s in sel]
    names = list(cfg.source_domains)
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for i, name in enumerate(names[:8]):
        y = [s["weights"][i] for s in sel]
        ax.plot(ep, y, color=SERIES[i], marker=MARKERS[i],
                dashes=DASHES[i], label=name, zorder=3)
        ax.annotate(f" {name}", (ep[-1], y[-1]), fontsize=8, va="center",
                    color=SERIES[i])
    ax.axhline(1.0, color=MUTED, linewidth=1.0, dashes=(2, 2), zorder=1)
    _note(ax, (ep[0], 1.0), "uniform (no selection)", xytext=(2, 4))
    _style(ax, "Source-domain transferability $s_{global}$", "epoch",
           "weight (mean 1)")
    ax.set_xlim(min(ep), max(ep) * 1.12)
    _legend_below(ax, ncol=min(len(names), 5))
    return _save(fig, run_dir, rid, "source_weights")


def fig_ego_support(run_dir, rid, seed_runs, cfg):
    """How much of each k-node ego-graph PPR actually reaches.

    The motivating diagnostic: on an induced-subgraph split most seeds sit in a
    component smaller than k, so most of the ego-graph is padding.
    """
    sup = seed_runs[0].get("support") or {}
    if not sup:
        return 0
    names = list(sup.keys())
    k = sup[names[0]].get("k", cfg.ego_size)
    mean = [sup[n]["mean_support"] for n in names]
    full = [100 * sup[n]["frac_full"] for n in names]
    iso = [100 * sup[n]["frac_singleton"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    y = np.arange(len(names))
    axes[0].barh(y, mean, height=0.6, color=SERIES[0], zorder=3)
    axes[0].axvline(k, color=MUTED, linewidth=1.0, dashes=(2, 2), zorder=4)
    _note(axes[0], (k, len(names) - 0.5), f"k = {k}", xytext=(-3, 0),
          ha="right", va="center")
    for yi, v in zip(y, mean):
        axes[0].annotate(f"{v:.1f}", (v, yi), textcoords="offset points",
                         xytext=(4, 0), va="center", fontsize=8, color=INK_2)
    axes[0].set_yticks(y, names)
    axes[0].set_xlim(0, k * 1.15)
    _style(axes[0], "Mean reached nodes per ego-graph", "nodes", ygrid=False)
    axes[0].grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_axisbelow(True)

    w = 0.38
    axes[1].barh(y + w / 2, full, height=w, color=SERIES[0],
                 label="fully filled", zorder=3)
    axes[1].barh(y - w / 2, iso, height=w, color=SERIES[1],
                 label="isolated (support 1)", zorder=3)
    for yi, v in zip(y, full):
        axes[1].annotate(f"{v:.0f}%", (v, yi + w / 2),
                         textcoords="offset points", xytext=(4, 0),
                         va="center", fontsize=8, color=INK_2)
    for yi, v in zip(y, iso):
        axes[1].annotate(f"{v:.0f}%", (v, yi - w / 2),
                         textcoords="offset points", xytext=(4, 0),
                         va="center", fontsize=8, color=INK_2)
    axes[1].set_yticks(y, names)
    axes[1].set_xlim(0, 112)
    _style(axes[1], "Share of ego-graphs", "% of seed nodes", ygrid=False)
    axes[1].grid(axis="x", color=GRID, linewidth=0.6)
    axes[1].set_axisbelow(True)
    _legend_below(axes[1], ncol=2, y=-0.20)
    fig.tight_layout()
    return _save(fig, run_dir, rid, "ego_support")


def fig_final_scores(run_dir, rid, seed_runs, aggregate, cfg):
    """Per-graph final scores; the target is the highlighted bar."""
    per = seed_runs[0].get("source_scores") or {}
    if not per:
        return 0
    src_names = list(per.keys())
    names = src_names + [cfg.target_domain]
    n_seed = len(seed_runs)
    metrics = [("acc", "Accuracy"), ("auc", "AUROC"), ("f1", "Macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=True)
    x = np.arange(len(names))
    for ax, (key, label) in zip(axes, metrics):
        # Sources are averaged over the same seeds as the target, so every bar
        # in the panel summarises the same set of runs.
        cols = [[s["source_scores"][n][key] for s in seed_runs] for n in src_names]
        vals = [float(np.mean(c)) for c in cols]
        errs = [float(np.std(c, ddof=1)) if n_seed > 1 else 0.0 for c in cols]
        vals.append(aggregate.get(f"{key}_mean", float("nan")))
        errs.append(aggregate.get(f"{key}_std", 0.0))
        colors = [SERIES[0]] * len(src_names) + [SERIES[1]]
        bars = ax.bar(x, vals, width=0.66, color=colors, zorder=3)
        if n_seed > 1:
            ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK_2,
                        elinewidth=1.2, capsize=3, zorder=4)
        for b, v, e in zip(bars, vals, errs):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v + e),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_ylim(0, 1.12)
        _style(ax, label)
    axes[0].set_ylabel("score")
    handles = [Line2D([], [], color=SERIES[0], linewidth=6, label="source"),
               Line2D([], [], color=SERIES[1], linewidth=6,
                      label=f"target ({cfg.target_domain})")]
    axes[1].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.30), ncol=2)
    fig.suptitle(
        "Final scores per graph"
        + (f" — mean ± sd over {n_seed} seeds" if n_seed > 1 else ""),
        x=0.008, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, run_dir, rid, "final_scores")


def fig_per_class_f1(run_dir, rid, seed_runs, cfg):
    """Per-class F1 on the target, sorted — where macro-F1 is actually lost."""
    pc = seed_runs[0].get("per_class_f1")
    support = seed_runs[0].get("per_class_support")
    if pc is None or len(pc) < 2:
        return 0
    pc = np.asarray(pc, dtype=float)
    order = np.argsort(-pc)
    width = min(max(6.6, 0.14 * len(pc) + 2), 8.4)
    fig, ax = plt.subplots(figsize=(width, 3.4))
    ax.bar(np.arange(len(pc)), pc[order], width=0.75, color=SERIES[0], zorder=3)
    macro = float(np.mean(pc))
    ax.axhline(macro, color=SERIES[1], linewidth=1.6, dashes=(4, 2), zorder=4)
    # Left for the macro line, right for the zero-class count: the two notes
    # sat on top of each other when both were right-aligned.
    _note(ax, (0, macro), f"macro-F1 = {macro:.3f}", SERIES[1], xytext=(2, 5))
    zero = int((pc == 0).sum())
    if zero:
        _note(ax, (len(pc) - 1, 0), f"{zero} of {len(pc)} classes score 0",
              INK_2, xytext=(-2, 10), ha="right")
    if len(pc) <= 12:
        ax.set_xticks(np.arange(len(pc)),
                      [f"c{c}" + (f"\nn={support[c]}" if support else "")
                       for c in order], fontsize=7.5)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"the {len(pc)} target classes, sorted by F1")
    _style(ax, f"Per-class F1 on {cfg.target_domain}", ylabel="F1")
    ax.set_ylim(0, 1.05)
    return _save(fig, run_dir, rid, "per_class_f1")


def fig_class_prior(run_dir, rid, seed_runs, cfg):
    """Source vs target label distribution — the label shift the run faces."""
    sp = seed_runs[0].get("src_prior")
    tp = seed_runs[0].get("tgt_true_prior")
    if sp is None or tp is None:
        return 0
    sp, tp = np.asarray(sp, float), np.asarray(tp, float)
    n = len(sp)
    fig, ax = plt.subplots(figsize=(min(max(6.6, 0.15 * n + 2), 8.4), 3.3))
    x = np.arange(n)
    if n <= 20:
        w = 0.38
        ax.bar(x - w / 2, sp, width=w, color=SERIES[0], label="source (pooled)",
               zorder=3)
        ax.bar(x + w / 2, tp, width=w, color=SERIES[1], label="target (true)",
               zorder=3)
        ax.set_xticks(x, [f"c{i}" for i in range(n)], fontsize=7.5)
    else:
        order = np.argsort(-tp)
        ax.plot(np.arange(n), sp[order], color=SERIES[0], marker=MARKERS[0],
                markevery=_markevery(n), dashes=DASHES[0],
                label="source (pooled)", zorder=3)
        ax.plot(np.arange(n), tp[order], color=SERIES[1], marker=MARKERS[1],
                markevery=_markevery(n), dashes=DASHES[1],
                label="target (true)", zorder=3)
        ax.set_xlabel(f"the {n} classes, sorted by target frequency")
    tv = 0.5 * float(np.abs(sp - tp).sum())
    _style(ax, f"Label shift — total variation {tv:.3f}", ylabel="class share")
    ax.legend(loc="upper right")
    return _save(fig, run_dir, rid, "class_prior")


def fig_confusion(run_dir, rid, seed_runs, cfg):
    """Row-normalised confusion on the target. Sequential single hue."""
    cm = seed_runs[0].get("confusion")
    if cm is None:
        return 0
    cm = np.asarray(cm, dtype=float)
    if cm.shape[0] > 12:                     # unreadable past ~12 classes
        return 0
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row == 0, 1, row))
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(0.62 * n + 2.6, 0.62 * n + 2.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            if v < 0.005:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > 0.55 else INK_2)
    ax.set_xticks(range(n), [f"c{i}" for i in range(n)], fontsize=8)
    ax.set_yticks(range(n), [f"c{i}  (n={int(row[i, 0])})" for i in range(n)],
                  fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Target confusion — {cfg.target_domain} (row-normalised)",
                 loc="left", color=INK, pad=10)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, color=MUTED)
    fig.tight_layout()
    return _save(fig, run_dir, rid, "confusion")


def fig_seed_spread(run_dir, rid, seed_runs, aggregate, cfg):
    """Per-seed target scores — the variance a single number hides."""
    if len(seed_runs) < 2:
        return 0
    seeds = [s["seed"] for s in seed_runs]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    x = np.arange(len(seeds))
    w = 0.26
    for i, (key, label) in enumerate([("acc", "Accuracy"), ("auc", "AUROC"),
                                      ("f1", "Macro-F1")]):
        vals = [s[key] for s in seed_runs]
        ax.bar(x + (i - 1) * w, vals, width=w, color=SERIES[i], label=label,
               zorder=3)
        ax.axhline(aggregate.get(f"{key}_mean", np.nan), color=SERIES[i],
                   linewidth=1.0, dashes=(3, 2), zorder=2)
    ax.set_xticks(x, [f"seed {s}" for s in seeds])
    ax.set_ylim(0, 1.05)
    _style(ax, "Per-seed target scores (dashed = mean)", ylabel="score")
    ax.legend(loc="lower right", ncol=3)
    return _save(fig, run_dir, rid, "seed_spread")


# ------------------------------------------------------------------- driver
def render_all(run_dir, rid, cfg, per_seed: List[dict], aggregate: dict,
               dataset: str, baseline: Optional[dict] = None) -> int:
    n = 0
    n += fig_target_curves(run_dir, rid, per_seed, cfg)
    n += fig_loss_components(run_dir, rid, per_seed, cfg)
    n += fig_selection_signal(run_dir, rid, per_seed, cfg)
    n += fig_source_weights(run_dir, rid, per_seed, cfg)
    n += fig_ego_support(run_dir, rid, per_seed, cfg)
    n += fig_final_scores(run_dir, rid, per_seed, aggregate, cfg)
    n += fig_per_class_f1(run_dir, rid, per_seed, cfg)
    n += fig_class_prior(run_dir, rid, per_seed, cfg)
    n += fig_confusion(run_dir, rid, per_seed, cfg)
    n += fig_seed_spread(run_dir, rid, per_seed, aggregate, cfg)
    return n
