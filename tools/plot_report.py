"""Report figures per the professor's rules (2026-07-29). Display only.

Rules implemented:
  1. Vertical dashed line at the phase change: left = pretrain (population
     total budget), right = evolution. Both SESiL runs in Fig 1 pretrain with
     a FIXED per-agent budget (no early stop), so the line sits at their
     shared population pretrain total (9 ep).
  2. x-axis = "epochs", one consistent definition for every method:
     1 epoch = one pass over the full 50k CIFAR-10 train set (what the docs
     call a full-set-equivalent unit). A 3-class-subset pass (15k) = 0.3 ep.
     Evolution adds 20 ep/generation (10 offspring x 2 full-set ep).
  3. One visual element per entity: baseline = red dashed line ONLY (its real
     per-epoch test-acc curve); each SESiL run = one solid mean line. No
     min-max bands, no diamonds, no extra hlines.
     SSL budget labels use "SESiL SSL [x]ep + [y]ep": x = core (SSL) budget,
     y = TOTAL finetune budget across the 10 agents (= 3S in the old ledger).
  4. One figure per experiment:
       Fig 1 (fig1_10c3.png)          : baseline + SESiL-origin + one default
                                        SSL-SESiL.
       Fig 2 (fig2_budget_sweep.png)  : equal-budget sweep (core + finetune = 9).
       Fig 2b (fig2_corners.png)      : budget grid corners (default, low-low,
                                        low-high; high-low = default).
       Fig 2c (fig2_finetune_sweep.png): core FIXED at 6 ep, finetune varies
                                        (impact of finetuning).
       Fig 2d (fig2_ssl_sweep.png)    : finetune FIXED at 1.5 ep, core varies
                                        (impact of the unsupervised learning).
     In the sweep figures the pretrain totals differ per run, so the phase
     line is drawn at each run's own pretrain total (rule 1 applied per run).

    python tools/plot_report.py
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

PER_GEN = 20.0   # epochs added per generation (10 offspring x 2 full-set ep)
PHASE = 9.0      # population pretrain total for the matched runs (ep)
MIN_GENS = 25    # every run in these figures is a 25-gen run; a run with fewer
                 # generations on disk is still training and must NOT be drawn
                 # (its legend would print an in-progress value as "final")

# Fig 1 entities. SESiL-origin = exp1b: fixed 3 subset-ep per agent, no early
# stop (the professor's phase-1 spec). Default SSL-SESiL = exp2d: core 9 ep +
# total finetune 0.3 ep -- the best allocation found in the sweep (Fig 2), per
# "assigning respect to previous results"; just one run, no tuning in Fig 1.
FIG1 = [
    # SESiL as published: partial zipping to node 21, offspring = head_models[0].
    ("runs_half/seed0/permute", "SESiL-origin (pretrain 9ep), partial merge as published", 9.0, "tab:green"),
    ("runs_exp2d/seed0/permute", "SESiL-foundation SSL 9ep + 0.3ep, partial merge as published", 9.3, "tab:purple"),
    # The same two SESiL runs with the merger applied to the WHOLE network and the
    # merged network saved (globa.run --merge sesil-full); nothing else changed.
    ("runs_phase4_origin_full/seed0/sesil-full_permute", "SESiL-origin, permute, whole-network merge", 9.0, "tab:olive"),
    ("runs_phase4_sesilfull_permute/seed0/sesil-full_permute", "SESiL-foundation, permute (= wavg), whole-network merge", 9.3, "tab:blue"),
    ("runs_phase4_sesilfull_zipit/seed0/sesil-full_zipit", "SESiL-foundation, zipit, whole-network merge", 9.3, "tab:cyan"),
    # GLOBA crossover arms (globa/run.py), same gen-0 population, seed and
    # mutation as exp2d; whole-network merge + label-aware head.
    ("runs_phase4_avg_label/seed0/globa_average_label", "GLOBA average + label head", 9.3, "tab:orange"),
    ("runs_phase4_sf_label/seed0/globa_single-full_label", "GLOBA single-full + label head", 9.3, "tab:red"),
]

# Fig 2: the budget-distribution sweep, labels "core x ep + total finetune y ep".
FIG2 = [
    ("runs_exp2a/seed0/permute", "SESiL SSL 3ep + 6ep", 9.0, "tab:orange"),
    ("runs_exp2b/seed0/permute", "SESiL SSL 6ep + 3ep", 9.0, "tab:purple"),
    ("runs_exp2c/seed0/permute", "SESiL SSL 7.5ep + 1.5ep", 9.0, "tab:brown"),
    ("runs_exp2d/seed0/permute", "SESiL SSL 9ep + 0.3ep", 9.3, "tab:cyan"),
]

# Fig 2b: grid corners. high-core/low-finetune = 9+0.3 = the default (exp2d).
FIG2_CORNERS = [
    ("runs_exp2i/seed0/permute", "SESiL SSL 3ep + 0.3ep (low core, low ft)", 3.3, "tab:pink"),
    ("runs_exp2a/seed0/permute", "SESiL SSL 3ep + 6ep (low core, high ft)", 9.0, "tab:orange"),
    ("runs_exp2d/seed0/permute", "SESiL SSL 9ep + 0.3ep (high core, low ft = default)", 9.3, "tab:cyan"),
]

# Fig 2c: SSL core FIXED at 6 ep, total finetune varies (impact of finetuning).
FIG2_FT = [
    ("runs_exp2f/seed0/permute", "SESiL SSL 6ep + 0.5ep", 6.5, "tab:blue"),
    ("runs_exp2g/seed0/permute", "SESiL SSL 6ep + 1ep", 7.0, "tab:green"),
    ("runs_exp2h/seed0/permute", "SESiL SSL 6ep + 1.5ep", 7.5, "tab:olive"),
    ("runs_exp2b/seed0/permute", "SESiL SSL 6ep + 3ep", 9.0, "tab:purple"),
]

# Fig 2d: total finetune FIXED at 1.5 ep, SSL core varies (impact of the
# unsupervised learning). exp2j/exp2k are skipped until their runs finish.
FIG2_SSL = [
    ("runs_exp2j/seed0/permute", "SESiL SSL 3ep + 1.5ep", 4.5, "tab:blue"),
    ("runs_exp2h/seed0/permute", "SESiL SSL 6ep + 1.5ep", 7.5, "tab:olive"),
    ("runs_exp2c/seed0/permute", "SESiL SSL 7.5ep + 1.5ep", 9.0, "tab:brown"),
    ("runs_exp2k/seed0/permute", "SESiL SSL 9ep + 1.5ep", 10.5, "tab:green"),
]


def smooth(y, w=5):
    if len(y) < w:
        return np.arange(len(y)), y
    sm = np.convolve(y, np.ones(w) / w, mode="valid")
    half = w // 2
    return np.arange(half, half + len(sm)), sm


def evo_mean(path, start):
    d = find_run_csvs(path)
    if not d:
        return None
    g = np.array(sorted(d))
    vals = [np.array([r["joint"] for r in parse_csv(d[k])[0]]) for k in g]
    mean = np.array([v.mean() for v in vals])
    lo = np.array([v.min() for v in vals])
    hi = np.array([v.max() for v in vals])
    return start + PER_GEN * (g - 1), mean, lo, hi


def baseline_curve():
    h = json.load(open("runs_baseline/joint_e520_seed0/history.json"))
    bx = np.array([e["epoch"] for e in h["epochs"]])
    by = np.array([e["test_acc"] for e in h["epochs"]])
    idx, bys = smooth(by, w=5)
    return bx[idx], bys


def phase_lines(ax, starts):
    """Rule 1, per run: dashed line where that run's pretrain ends."""
    for x in sorted(set(starts)):
        ax.axvline(x, color="0.4", ls="--", lw=1.2)
    if starts:
        ax.text(max(starts), ax.get_ylim()[1], "  pretrain | evolution",
                va="top", ha="left", fontsize=8, color="0.35")


def draw(runs, out, title, with_baseline):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if with_baseline:
        bx, by = baseline_curve()
        ax.plot(bx, by, "--", color="red", lw=1.6,
                label="baseline: joint training")
    starts = []
    for path, label, start, color in runs:
        s = evo_mean(path, start)
        if s is None or len(s[0]) < MIN_GENS:
            continue
        x, mean, lo, hi = s
        starts.append(start)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, "-", color=color, lw=2,
                label=f"{label}, final {mean[-1]:.3f}")
    ax.set_xlabel("epochs")
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 520)
    ax.set_ylim(0, 1.0)
    phase_lines(ax, starts)
    ax.grid(alpha=0.25)
    leg = ax.legend(loc="lower right", fontsize=8.5, framealpha=0.6)
    leg.get_frame().set_edgecolor("0.7")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def main():
    draw(FIG1, "figures/fig1_10c3.png",
         "CIFAR-10 10C3, 10 agents", with_baseline=True)
    draw(FIG2, "figures/fig2_budget_sweep.png",
         "SSL pretrain budget distribution (core + finetune = 9 ep)",
         with_baseline=False)
    draw(FIG2_CORNERS, "figures/fig2_corners.png",
         "SSL budget grid corners", with_baseline=False)
    draw(FIG2_FT, "figures/fig2_finetune_sweep.png",
         "Impact of finetuning (SSL core fixed at 6 ep)",
         with_baseline=False)
    draw(FIG2_SSL, "figures/fig2_ssl_sweep.png",
         "Impact of unsupervised learning (finetune fixed at 1.5 ep)",
         with_baseline=False)


if __name__ == "__main__":
    main()
