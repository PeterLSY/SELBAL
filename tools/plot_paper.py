"""Paper-style accuracy-vs-training-budget figure.

x = training units (full-set-equivalent epochs), per the budget-axis table in
experiments.md: each evolution run starts at its pretraining cost and adds 20
units per generation (10 offspring x 2 mutation epochs x full-set).

Curves:
  * four SESiL evolution runs -- solid line (population mean) + translucent
    min-max band -- placed on the budget axis by (start, +20/gen).
  * joint-training full baseline -- red dashed, its actual per-epoch test-acc
    curve over x = 1..520 (a real training curve, not a horizontal line).
  * joint-training 19-epoch baseline -- short dark marker at its end-point,
    kept minimal so it does not clutter the left edge.

exp2c/exp2ref auto-appear once their gen25 CSVs exist (they read live).

    python tools/plot_paper.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plot_evolution import find_run_csvs, parse_csv  # noqa: E402

PER_GEN = 20.0  # full-set-equivalent epochs added per generation

# (path, base_label, start_units, color). Final/running suffix is appended live.
EVO = [
    ("runs_weak/seed0/permute", "SESiL weak-supervised (19.2u)", 19.2, "tab:blue"),
    ("runs_half/seed0/permute", "SESiL half-budget (9.0u)", 9.0, "tab:green"),
    ("runs_exp2a/seed0/permute", "SESiL SSL E3+S2 (9.0u)", 9.0, "tab:orange"),
    ("runs_exp2b/seed0/permute", "SESiL SSL E6+S1 (9.0u)", 9.0, "tab:purple"),
    ("runs_exp2c/seed0/permute", "SESiL SSL E7.5+S0.5 (9.0u)", 9.0, "tab:brown"),
    ("runs_exp2d/seed0/permute", "SESiL SSL E9+S0.1 (9.3u, min-cal)", 9.3, "tab:cyan"),
    ("runs_exp2ref/seed0/permute", "SESiL SSL E50+S1 (53u)", 53.0, "tab:pink"),
]


def run_done(path):
    """True if the run wrote its 'Done.' marker (all generations complete)."""
    log = path + ".log"
    if not os.path.exists(log):
        return False
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        return any(line.startswith("Done.") for line in f)


def smooth(y, w=5):
    """Centered moving average; returns (valid_slice_indices, smoothed_y)."""
    if len(y) < w:
        return np.arange(len(y)), y
    kern = np.ones(w) / w
    sm = np.convolve(y, kern, mode="valid")
    half = w // 2
    idx = np.arange(half, half + len(sm))
    return idx, sm


def evo_series(path, start):
    d = find_run_csvs(path)
    if not d:
        return None
    g = np.array(sorted(d))
    mean = np.array([np.array([r["joint"] for r in parse_csv(d[k])[0]]).mean() for k in g])
    lo = np.array([np.array([r["joint"] for r in parse_csv(d[k])[0]]).min() for k in g])
    hi = np.array([np.array([r["joint"] for r in parse_csv(d[k])[0]]).max() for k in g])
    x = start + PER_GEN * (g - 1)
    return x, mean, lo, hi


def plot_evo_curves(ax, include_ref=False):
    """Draw the evolution runs. exp2ref (53u over-budget) is off the matched-budget
    comparison, so it is excluded from the main figure and included only in the
    honesty figure (include_ref=True), where the wider x-range has room for it."""
    for path, base, start, color in EVO:
        if "exp2ref" in path and not include_ref:
            continue
        s = evo_series(path, start)
        if s is None:
            continue
        x, mean, lo, hi = s
        maxgen = len(mean)
        if run_done(path):
            label = f"{base}, final {mean[-1]:.3f}"
        else:
            label = f"{base} (running gen{maxgen}={mean[-1]:.3f})"
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, "-", color=color, lw=2, label=label)


def joint_full_curve():
    h = json.load(open("runs_baseline/joint_e520_seed0/history.json"))
    bx = np.array([e["epoch"] for e in h["epochs"]])
    by = np.array([e["test_acc"] for e in h["epochs"]])
    idx, bys = smooth(by, w=5)
    return bx[idx], bys


def make_main():
    """Figure A (for the professor): initialization strategies under matched budget.
    joint-full demoted to a faint upper-bound reference; no 19ep marker; y focused.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bx, bys = joint_full_curve()
    ax.plot(bx, bys, "--", color="0.6", lw=1.2, alpha=0.8,
            label="joint training (full 520ep) — upper-bound ref")
    plot_evo_curves(ax, include_ref=True)
    ax.set_xlabel("Training units (full-set-equivalent epochs)")
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.set_title("SESiL: Initialization Strategies under Matched Budget")
    ax.set_ylim(0.05, 0.90)
    ax.set_xlim(0, 540)
    ax.grid(alpha=0.25)
    leg = ax.legend(loc="lower right", fontsize=7.5, framealpha=0.6)
    leg.get_frame().set_edgecolor("0.7")
    fig.text(0.5, 0.005,
             "exp2d (E=9) uses S=0.1 subset-epoch = 3 batches/expert = 0.3u finetune, "
             "the minimal-calibration config tested; the strict S->0 (zero-finetune) "
             "endpoint is not measured (see exp2e).",
             ha="center", va="bottom", fontsize=6, color="0.4")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = "figures/paper_main.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def make_honesty():
    """Figure B (honesty appendix): full picture, y 0-1, with the caveat caption."""
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    bx, bys = joint_full_curve()
    ax.plot(bx, bys, "--", color="red", lw=1.8, alpha=0.9,
            label="joint training (full, 520 ep)")
    h19 = json.load(open("runs_baseline/joint_e19_seed0/history.json"))
    y19 = h19["best"]["test_acc"]
    ax.axhline(y19, color="0.25", alpha=0.3, lw=1, ls=":", zorder=1)
    ax.plot([19], [y19], marker="D", ms=7, color="0.25", zorder=5,
            label=f"joint training (19 ep, budget-matched) = {y19:.3f}")
    plot_evo_curves(ax, include_ref=True)
    ax.set_xlabel("Training units (full-set-equivalent epochs)")
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.set_title("CIFAR-10 C3: Evolutionary Merging vs Joint-Training Baselines")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 530)
    ax.grid(alpha=0.25)
    leg = ax.legend(loc="lower right", fontsize=7.5, framealpha=0.6)
    leg.get_frame().set_edgecolor("0.7")
    fig.text(0.5, 0.012,
             "Under matched compute, joint training dominates; the evolutionary "
             "merging paradigm earns its keep where data cannot be centralized. "
             "Within the paradigm,\nthe initialization strategy matters substantially "
             "— SSL shared init outperforms supervised init at equal budget.",
             ha="center", va="bottom", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = "figures/paper_honesty.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def main():
    make_main()
    make_honesty()


if __name__ == "__main__":
    main()
