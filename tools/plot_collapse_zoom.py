"""Gen 1-8 zoom of all six runs, aligned on generation.

Shows the shared claim: trough depths are similar but recovery slopes diverge.
Colors follow the paper figures (SSL warm, supervised cool, strong grey).

    python tools/plot_collapse_zoom.py
"""

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

from plot_evolution import find_run_csvs, find_generation_csvs, parse_csv  # noqa: E402

GMAX = 8
RUNS = [
    ("strong", "hist", "permute", "0.5"),
    ("exp1 (weak-sup)", "run", "runs_weak/seed0/permute", "tab:blue"),
    ("exp1b (half-sup)", "run", "runs_half/seed0/permute", "tab:green"),
    ("exp2a (SSL E3S2)", "run", "runs_exp2a/seed0/permute", "tab:orange"),
    ("exp2b (SSL E6S1)", "run", "runs_exp2b/seed0/permute", "tab:purple"),
    ("exp2c (SSL E7.5S0.5)", "run", "runs_exp2c/seed0/permute", "tab:brown"),
    ("exp2ref (SSL E50S1)", "run", "runs_exp2ref/seed0/permute", "tab:red"),
]


def series(kind, arg):
    d = find_generation_csvs("./csvs", arg) if kind == "hist" else find_run_csvs(arg)
    g = [k for k in sorted(d) if k <= GMAX]
    m = [np.array([r["joint"] for r in parse_csv(d[k])[0]]).mean() for k in g]
    return np.array(g), np.array(m)


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label, kind, arg, color in RUNS:
        g, m = series(kind, arg)
        ls = "--" if label == "strong" else "-"
        ax.plot(g, m, ls, marker="o", color=color, lw=2, ms=5, label=label)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population-mean Joint")
    ax.set_title("Generations 1-8: collapse and recovery across initializations")
    ax.set_xlim(0.8, GMAX + 0.2)
    ax.set_ylim(0.08, 0.75)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    out = "figures/collapse_recovery_zoom.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
