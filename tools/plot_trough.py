"""Early-generation zoom: SSL shared init vs supervised inits.

Focuses on gen 1-8 to show the merge-damage trough at gen 2 and the recovery
slope, annotating the gen-2 trough of each run. The claim being visualised:
a shared SSL initialisation puts every expert in one feature space, so the
first merge damages less (shallower trough) and recovers faster (steeper slope)
than independently-pretrained experts.

    python tools/plot_trough.py
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

from plot_evolution import find_run_csvs, parse_csv  # noqa: E402

RUNS = [
    ("runs_exp2b/seed0/permute", "SSL E6+S1 (9.0u)", "tab:red"),
    ("runs_half/seed0/permute", "half budget (9.0u)", "tab:green"),
    ("runs_weak/seed0/permute", "weak init (19.2u)", "tab:orange"),
]
GMAX = 8


def series(path):
    d = find_run_csvs(path)
    gens, mean = [], []
    for g in sorted(d):
        if g > GMAX:
            break
        pop, _ = parse_csv(d[g])
        gens.append(g)
        mean.append(np.array([r["joint"] for r in pop]).mean())
    return np.array(gens), np.array(mean)


def main():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for path, label, color in RUNS:
        g, m = series(path)
        ax.plot(g, m, "o-", color=color, lw=2, ms=5, label=label)
        # annotate the gen-2 trough
        i2 = np.where(g == 2)[0]
        if len(i2):
            y = m[i2[0]]
            ax.annotate(
                f"gen2 = {y:.3f}",
                xy=(2, y), xytext=(2.35, y - 0.03),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )
        # slope gen1->gen5 label near gen 5
        i1 = np.where(g == 1)[0]
        i5 = np.where(g == 5)[0]
        if len(i1) and len(i5):
            slope = (m[i5[0]] - m[i1[0]]) / 4
            ax.annotate(
                f"+{slope:.3f}/gen" if slope >= 0 else f"{slope:.3f}/gen",
                xy=(5, m[i5[0]]), xytext=(5.1, m[i5[0]] + 0.02),
                fontsize=8, color=color, fontweight="bold",
            )

    ax.set_xlabel("Generation")
    ax.set_ylabel("CIFAR-10 accuracy (population mean, Joint)")
    ax.set_title("Merge-damage trough & recovery: shared SSL init vs supervised")
    ax.set_xlim(0.7, GMAX + 0.3)
    ax.set_ylim(0, 0.75)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = "figures/exp2b_trough_zoom.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
