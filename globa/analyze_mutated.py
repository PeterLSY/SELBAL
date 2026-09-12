"""Stage 0b analysis: H0 on post-mutation children (merge_dataset_exp2d_globa_mutated.csv).

    python -m globa.analyze_mutated [--csv merge_dataset_exp2d_globa_mutated.csv] [--pre merge_dataset_exp2d_globa_v2.csv]

Prints: pre/post means, pre->post rank stability, Spearman of the GLOBA
diagnostics against post-mutation outcomes, and the same table split by the
number of shared classes.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

FEATS = ["type_A", "type_B", "type_C", "type_E", "type_D_plus", "type_D_minus",
         "typed_frac_mean", "norm_ratio_med", "mean_parent", "pre_joint", "pre_per_task"]
TGTS = ["post_joint", "post_per_task", "excess_post"]


def shared_count(row):
    a = {int(x) for x in str(row.labels_a).split("_")}
    b = {int(x) for x in str(row.labels_b).split("_")}
    return len(a & b)


def table(df, feats, tgts, title):
    print(f"\n{title}  (n={len(df)})")
    print(f"  {'feature':16s} " + " ".join(f"{t:>16s}" for t in tgts))
    for f in feats:
        if f not in df or df[f].nunique() < 3:
            continue
        cells = []
        for t in tgts:
            r, p = spearmanr(df[f], df[t])
            cells.append(f"{r:+.3f} ({p:.2f})")
        print(f"  {f:16s} " + " ".join(f"{c:>16s}" for c in cells))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.analyze_mutated")
    ap.add_argument("--csv", default="merge_dataset_exp2d_globa_mutated.csv")
    ap.add_argument("--pre", default="merge_dataset_exp2d_globa_v2.csv",
                    help="Phase 2 v2 table (pre-mutation regimes incl. saved baseline), optional")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    df["mean_parent"] = 0.5 * (df.a_per_task + df.b_per_task)
    df["excess_post"] = df.post_per_task - df.mean_parent
    df["excess_pre"] = df.pre_per_task - df.mean_parent
    df["gain"] = df.post_per_task - df.pre_per_task
    df["shared"] = df.apply(shared_count, axis=1)
    print(f"{len(df)} pairs, regime {df.regime.iloc[0]}, mean time {df.time_s.mean():.0f}s")

    print("\n=== means ===")
    print(f"  parents per_task (mean of A,B)  {df.mean_parent.mean():.4f}")
    print(f"  pre   joint {df.pre_joint.mean():.4f}   per_task {df.pre_per_task.mean():.4f}   excess {df.excess_pre.mean():+.4f}")
    print(f"  post  joint {df.post_joint.mean():.4f}   per_task {df.post_per_task.mean():.4f}   excess {df.excess_post.mean():+.4f}")
    print(f"  post > mean parent (per_task): {int((df.excess_post > 0).sum())}/{len(df)}")
    print(f"  post > max parent  (per_task): {int((df.post_per_task > df[['a_per_task','b_per_task']].max(axis=1)).sum())}/{len(df)}")
    r, p = spearmanr(df.pre_per_task, df.post_per_task)
    print(f"  rank stability pre->post per_task: rho {r:+.3f} (p {p:.3f});"
          f"  joint: rho {spearmanr(df.pre_joint, df.post_joint)[0]:+.3f}")

    print("\n=== by shared classes ===")
    g = df.groupby("shared")
    print(f"  {'shared':>6s} {'n':>3s} {'parents':>8s} {'pre_pt':>8s} {'post_pt':>8s} {'post_j':>8s} {'excess':>8s}")
    for k, d in g:
        print(f"  {k:>6d} {len(d):>3d} {d.mean_parent.mean():8.4f} {d.pre_per_task.mean():8.4f} "
              f"{d.post_per_task.mean():8.4f} {d.post_joint.mean():8.4f} {d.excess_post.mean():+8.4f}")

    print("\n=== type shares (mean) ===")
    print("  " + "  ".join(f"{t.replace('type_','')}={df[t].mean():.3f}" for t in FEATS[:6]))

    table(df, FEATS, TGTS, "=== H0 on POST-mutation outcomes, Spearman rho (p) ===")
    table(df, FEATS[:8], ["gain"], "=== diagnostics vs mutation gain (post - pre per_task) ===")
    # partial: residualise post on mean_parent, then correlate
    b = np.polyfit(df.mean_parent, df.post_per_task, 1)
    df["post_resid"] = df.post_per_task - np.polyval(b, df.mean_parent)
    table(df, FEATS[:8] + ["pre_per_task"], ["post_resid"], "=== after linear control for parent quality ===")

    if args.pre and os.path.exists(args.pre):
        pre = pd.read_csv(args.pre)
        saved = pre[pre.regime.str.startswith("saved")] if "regime" in pre else pd.DataFrame()
        if len(saved):
            m = df.merge(saved[["parent_a", "parent_b", "joint", "per_task"]].rename(
                columns={"joint": "saved_joint", "per_task": "saved_per_task"}), on=["parent_a", "parent_b"])
            print(f"\n=== SESiL saved child (= parent A, pre-mutation) on the same pairs (n={len(m)}) ===")
            print(f"  saved per_task {m.saved_per_task.mean():.4f} joint {m.saved_joint.mean():.4f}")
            print(f"  GLOBA post-mutation per_task {m.post_per_task.mean():.4f} joint {m.post_joint.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
