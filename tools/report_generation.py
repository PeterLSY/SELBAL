"""Matched-generation snapshot of one generation, vs the historical runs.

Compares like with like: generation N of the current run against generation N
of the reference runs, never against their final generation. Per the rule in
experiments.md, comparing a 25-generation run's tail to an 11-generation run's
tail would read "ran longer" as "performed better".

    python tools/report_generation.py --gen 5
    python tools/report_generation.py --gen 11 --with-wavg
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plot_evolution import find_generation_csvs, find_run_csvs, parse_csv  # noqa: E402


def stats(csv_path):
    pop, _ = parse_csv(csv_path)
    if not pop:
        return None
    j = np.array([r["joint"] for r in pop])
    p = np.array([r["per_task_avg"] for r in pop])
    return {
        "n": len(pop),
        "mean": j.mean(),
        "min": j.min(),
        "max": j.max(),
        "spread": j.max() - j.min(),
        "pta": p.mean(),
    }


def line(label, s):
    if s is None:
        print(f"{label:<26} (no data)")
        return
    print(
        f"{label:<26} n={s['n']:<3} Joint mean {s['mean']:.4f}  "
        f"min {s['min']:.4f}  max {s['max']:.4f}  "
        f"spread {s['spread']:.4f}  PTA {s['pta']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--run", default="runs_weak/seed0/permute")
    ap.add_argument("--run-label", default="exp1 weak init 0.7529")
    ap.add_argument("--with-wavg", action="store_true",
                    help="also compare against hist_wavg_11gen at the same gen")
    args = ap.parse_args()

    g = args.gen
    print(f"===== Generation {g} (matched-generation) =====")

    cur_map = find_run_csvs(args.run)
    cur = stats(cur_map[g]) if g in cur_map else None
    line(args.run_label, cur)

    ref_map = find_generation_csvs("./csvs", "permute")
    ref = stats(ref_map[g]) if g in ref_map else None
    line("hist_permute strong ~0.93", ref)

    wav = None
    if args.with_wavg:
        w_map = find_generation_csvs("./csvs", "wavg")
        wav = stats(w_map[g]) if g in w_map else None
        line("hist_wavg strong ~0.93", wav)

    print()
    if cur and ref:
        print(f"  delta vs hist_permute : {cur['mean'] - ref['mean']:+.4f} "
              f"(ratio {cur['mean'] / ref['mean']:.2f})")
    if cur and wav:
        print(f"  delta vs hist_wavg    : {cur['mean'] - wav['mean']:+.4f} "
              f"(ratio {cur['mean'] / wav['mean']:.2f})")
    if cur and ref:
        print(f"  spread  weak {cur['spread']:.4f}  vs  strong {ref['spread']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
