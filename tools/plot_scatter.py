"""Scatter: expert start-point accuracy vs evolution end-point (gen25).

The claim: the population's start-point classification accuracy does NOT predict
where evolution lands. If it did, the points would trend up-right; instead the
strongest start (strong init 0.93) lands BELOW the weakest SSL start (exp2b
0.73) at gen25. What predicts the end-point is backbone feature quality /
mergeability, not start-point accuracy.

Start-point = mean 3-class subset accuracy of the 10 experts right after
pretraining (pretrain-summary 'mean' row). End-point = gen25 population-mean
Joint (read live from CSV, so exp2c/exp2ref auto-appear once their gen25 lands).

    python tools/plot_scatter.py
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

# (label, start-point mean, run path or None, family, estimated?)
# family: 'ssl' | 'sup' ; estimated flags a start-point we didn't fully measure.
POINTS = [
    ("exp2a (E3,S2)", 0.7663, "runs_exp2a/seed0/permute", "ssl", False),
    ("exp2b (E6,S1)", 0.7312, "runs_exp2b/seed0/permute", "ssl", False),
    ("exp2c (E7.5,S0.5)", 0.6273, "runs_exp2c/seed0/permute", "ssl", False),
    ("exp2d (E9,S0.1)", 0.5571, "runs_exp2d/seed0/permute", "ssl", False),
    ("exp2ref (E50,S1)", 0.8260, "runs_exp2ref/seed0/permute", "ssl", False),
    ("exp1 (weak 0.75)", 0.7529, "runs_weak/seed0/permute", "sup", False),
    ("exp1b (half)", 0.6435, "runs_half/seed0/permute", "sup", False),
    ("strong (~0.93)", 0.93, None, "sup", True),  # gen25 from csvs; start estimated
]
STRONG_GEN25 = None  # filled from csvs


def gen25_of(path):
    d = find_run_csvs(path)
    if 25 not in d:
        return None
    return np.array([r["joint"] for r in parse_csv(d[25])[0]]).mean()


def main():
    strong = find_generation_csvs("./csvs", "permute")
    strong_g25 = (
        np.array([r["joint"] for r in parse_csv(strong[25])[0]]).mean()
        if 25 in strong else None
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    colors = {"ssl": "tab:red", "sup": "tab:blue"}
    seen_family = set()

    for label, start, path, fam, est in POINTS:
        if start is None:
            continue  # start-point not known yet (exp2ref pending)
        y = strong_g25 if path is None else gen25_of(path)
        if y is None:
            continue  # gen25 not ready yet
        lab = None
        flabel = "SSL shared init" if fam == "ssl" else "supervised init"
        if fam not in seen_family:
            lab = flabel
            seen_family.add(fam)
        if est:
            # estimated start-point (value approximate, training budget unknown)
            ax.scatter(start, y, s=130, facecolors="none", edgecolors=colors[fam],
                       marker="s", linewidths=1.6, zorder=3, label=lab)
            tag = f"{label} est.\n({start:.2f} est., {y:.3f})"
        else:
            ax.scatter(start, y, s=110, c=colors[fam], marker="o",
                       edgecolors="black", linewidths=0.7, zorder=3, label=lab)
            tag = f"{label}\n({start:.3f}, {y:.3f})"
        # labels below-right for high points, above-right for low, to dodge overlap
        below = y > 0.842
        ax.annotate(tag, xy=(start, y),
                    xytext=(start + 0.007, y + (-0.010 if below else 0.004)),
                    fontsize=7.5, va="top" if below else "bottom")

    ax.set_xlabel("Expert start-point: mean 3-class accuracy after pretraining")
    ax.set_ylabel("Evolution end-point: gen25 population-mean Joint")
    ax.set_title("Start-point accuracy does NOT predict evolution end-point")
    ax.grid(alpha=0.3)
    ax.annotate("strongest start, NOT best end (and its budget is unknown)",
                xy=(0.93, strong_g25 or 0.84), xytext=(0.66, 0.818),
                fontsize=8, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.60, 0.99)
    ax.set_ylim(top=(max(0.856, (strong_g25 or 0.84) + 0.02)))
    # Honesty caption: precise claim + not-independent-samples caveat.
    fig.text(0.5, 0.012,
             "Start-point predicts end-point NEITHER within nor across paradigms. "
             "Within SSL the end-point tracks SSL-pretrain budget E, not start "
             "(exp2a starts\nabove exp2c yet ends below; start is confounded by "
             "finetune S). Across paradigms the strongest start (supervised ~0.93) "
             "is overtaken.\nDescriptive scatter, not a regression — points are not "
             "independent (shared SSL trajectory / shared class split); 'strong' "
             "start estimated, budget unrecorded.",
             ha="center", va="bottom", fontsize=6.2, color="0.35")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = "figures/startpoint_vs_final.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    print("(square = estimated start-point; SSL=red, supervised=blue)")


if __name__ == "__main__":
    main()
