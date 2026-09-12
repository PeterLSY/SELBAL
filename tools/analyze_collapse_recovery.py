"""Collapse & recovery analysis of the merge-damage trough.

Offline, CSV-only. For each completed run it summarises the gen-2-ish collapse
(the first-merge damage) and how the population recovers from it.

Metrics (per the task spec 1-4, plus a few natural closers):
  1. start        : gen1 population-mean Joint.
  2. trough_gen / trough_mean : generation of the lowest population-mean, and
     that mean. trough_min / trough_min_gen : the single lowest individual Joint
     anywhere in the run, and the generation it occurred in.
  3. collapse_depth : start - trough_mean (absolute drop) and trough_mean/start
     (fraction of start retained at the trough).
  4. recovery_gen : first generation AFTER the trough whose population-mean is
     back at or above `start`. recovery_span = recovery_gen - trough_gen.
  5. recovery_rate : mean per-generation slope from the trough to recovery_gen.
  6. half_ceiling_gen : first generation reaching 90% of this run's own gen25
     final -- how fast it enters the end-game.
  (+) final       : last-generation population-mean, and net_gain = final - start.

    python tools/analyze_collapse_recovery.py            # prints markdown table
    python tools/analyze_collapse_recovery.py --md out.md
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plot_evolution import find_run_csvs, find_generation_csvs, parse_csv  # noqa: E402

# (label, kind, arg) -- kind 'run' uses find_run_csvs, 'hist' uses csvs root.
RUNS = [
    ("strong", "hist", "permute"),
    ("exp1", "run", "runs_weak/seed0/permute"),
    ("exp1b", "run", "runs_half/seed0/permute"),
    ("exp2a", "run", "runs_exp2a/seed0/permute"),
    ("exp2b", "run", "runs_exp2b/seed0/permute"),
    ("exp2c", "run", "runs_exp2c/seed0/permute"),
    ("exp2ref", "run", "runs_exp2ref/seed0/permute"),
]


def load_series(kind, arg):
    """Return {gen: (mean, min, [all member joints])} for a run."""
    gen_map = find_generation_csvs("./csvs", arg) if kind == "hist" else find_run_csvs(arg)
    out = {}
    for g in sorted(gen_map):
        pop, _ = parse_csv(gen_map[g])
        if not pop:
            continue
        joints = np.array([r["joint"] for r in pop])
        out[g] = (joints.mean(), joints.min(), joints)
    return out


def analyse(series):
    gens = sorted(series)
    means = np.array([series[g][0] for g in gens])
    start = means[0]

    # trough by population mean
    ti = int(np.argmin(means))
    trough_gen = gens[ti]
    trough_mean = means[ti]

    # global single-member minimum, anywhere
    tmin, tmin_gen = np.inf, None
    for g in gens:
        m = series[g][1]
        if m < tmin:
            tmin, tmin_gen = m, g

    depth_abs = start - trough_mean
    retain = trough_mean / start if start else float("nan")

    # first gen after the trough back at/above start
    recovery_gen = None
    for g in gens:
        if g > trough_gen and series[g][0] >= start:
            recovery_gen = g
            break
    recovery_span = (recovery_gen - trough_gen) if recovery_gen is not None else None

    # 5. mean per-generation slope from trough to recovery point
    if recovery_gen is not None and recovery_span:
        rec_mean = series[recovery_gen][0]
        recovery_rate = (rec_mean - trough_mean) / recovery_span
    else:
        recovery_rate = None

    final = means[-1]

    # 6. first gen reaching 90% of this run's own final
    ceil90 = 0.9 * final
    half_ceiling_gen = next((g for g in gens if series[g][0] >= ceil90), None)

    return {
        "start": start,
        "trough_gen": trough_gen,
        "trough_mean": trough_mean,
        "trough_min": tmin,
        "trough_min_gen": tmin_gen,
        "depth_abs": depth_abs,
        "retain": retain,
        "recovery_gen": recovery_gen,
        "recovery_span": recovery_span,
        "recovery_rate": recovery_rate,
        "half_ceiling_gen": half_ceiling_gen,
        "final": final,
        "net_gain": final - start,
        "last_gen": gens[-1],
    }


def markdown_table(rows):
    cols = [
        ("run", "{label}"),
        ("start", "{start:.4f}"),
        ("trough_gen", "{trough_gen}"),
        ("trough_mean", "{trough_mean:.4f}"),
        ("trough_min (gen)", "{trough_min:.4f} (g{trough_min_gen})"),
        ("depth_abs", "{depth_abs:.4f}"),
        ("retain", "{retain:.3f}"),
        ("recov_gen", "{recovery_gen}"),
        ("recov_span", "{recovery_span}"),
        ("recov_rate", "{recovery_rate:.4f}"),
        ("90%_gen", "{half_ceiling_gen}"),
        ("final", "{final:.4f}"),
        ("net_gain", "{net_gain:+.4f}"),
    ]
    head = "| " + " | ".join(h for h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        d = dict(r)
        for k in ("recovery_gen", "recovery_span", "recovery_rate", "half_ceiling_gen"):
            if d.get(k) is None:
                d[k] = "n/a"
        cells = []
        for _, fmt in cols:
            try:
                cells.append(fmt.format(**d))
            except (ValueError, KeyError):
                cells.append(str(d.get(fmt.strip("{}").split(":")[0], "")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="figures/collapse_recovery_table.md")
    args = ap.parse_args()

    rows = []
    for label, kind, arg in RUNS:
        series = load_series(kind, arg)
        if not series:
            print(f"{label}: no data")
            continue
        r = analyse(series)
        r["label"] = label
        rows.append(r)

    table = markdown_table(rows)
    print(table)

    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("# Collapse & recovery of the merge-damage trough\n\n")
        f.write("Offline CSV analysis of six completed runs. `retain` = "
                "trough_mean/start (fraction kept at the trough); `recov_span` = "
                "generations from trough back to start level; `recov_rate` = mean "
                "slope over that span; `90%_gen` = first gen at 90% of the run's "
                "own final.\n\n")
        f.write(table + "\n")
    print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
