"""Stage 0b, the outcome the proposal names explicitly: per-class RETENTION.

retention(c) = child_acc(c) - best_parent_acc(c), split into classes both
parents know (shared) and classes only one knows (disjoint). The proposal's
expected result was "D- energy the strongest predictor of shared-skill
damage"; this script tests exactly that, pre- and post-mutation.

    python -m globa.analyze_retention
"""

from __future__ import annotations

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

FEATS = ["type_A", "type_B", "type_C", "type_E", "type_D_plus", "type_D_minus", "typed_frac_mean", "norm_ratio_med", "mean_parent"]


def retention(row, child_col):
    a = {int(x) for x in str(row.labels_a).split("_")}
    b = {int(x) for x in str(row.labels_b).split("_")}
    pa = np.array(json.loads(row.a_per_class), dtype=float)
    pb = np.array(json.loads(row.b_per_class), dtype=float)
    ch = np.array([np.nan if v is None else v for v in json.loads(row[child_col])], dtype=float)
    shared, disjoint = sorted(a & b), sorted(a ^ b)
    best = np.maximum(pa, pb)
    r_sh = float(np.mean(ch[shared] - best[shared])) if shared else np.nan
    r_dj = float(np.mean(ch[disjoint] - best[disjoint])) if disjoint else np.nan
    return pd.Series({"ret_shared": r_sh, "ret_disjoint": r_dj, "ret_all": float(np.nanmean(ch[sorted(a | b)] - best[sorted(a | b)])),
                      "n_shared": len(shared), "worst_disjoint": float(np.min(ch[disjoint] - best[disjoint])) if disjoint else np.nan})


def table(df, tgts, title):
    d = df.dropna(subset=tgts)
    print(f"\n{title}  (n={len(d)})")
    print(f"  {'feature':16s} " + " ".join(f"{t:>16s}" for t in tgts))
    for f in FEATS:
        cells = []
        for t in tgts:
            r, p = spearmanr(d[f], d[t]); cells.append(f"{r:+.3f} ({p:.2f})")
        print(f"  {f:16s} " + " ".join(f"{c:>16s}" for c in cells))


def run(df, child_col, label):
    df = df.copy()
    df["mean_parent"] = 0.5 * (df.a_per_task + df.b_per_task)
    df = pd.concat([df, df.apply(lambda r: retention(r, child_col), axis=1)], axis=1)
    print(f"\n==================== {label} ====================")
    print(f"  retention (child - best parent), mean over pairs:  shared {df.ret_shared.mean():+.4f}  disjoint {df.ret_disjoint.mean():+.4f}  all {df.ret_all.mean():+.4f}")
    print(f"  worst single disjoint class per pair, mean {df.worst_disjoint.mean():+.4f}, min {df.worst_disjoint.min():+.4f}")
    g = df.groupby("n_shared")[["ret_shared", "ret_disjoint", "ret_all"]].mean()
    print("  by #shared classes:\n" + g.to_string(float_format=lambda x: f"{x:+.4f}"))
    r, p = spearmanr(df.n_shared, df.ret_all); print(f"  rho(#shared, ret_all) = {r:+.3f} (p {p:.2f})   [proposal pilot: -0.35]")
    table(df, ["ret_shared", "ret_disjoint", "ret_all"], "  types vs retention, Spearman rho (p)")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.analyze_retention")
    ap.add_argument("--pre", default="merge_dataset_exp2d_globa_v2.csv")
    ap.add_argument("--post", default="merge_dataset_exp2d_globa_mutated.csv")
    ap.add_argument("--permute", default="merge_dataset_exp2d_permute.csv")
    args = ap.parse_args(argv)

    pre = pd.read_csv(args.pre)
    for regime in ["average@0.80@label", "single-full@0.80@label"]:
        run(pre[pre.regime == regime], "child_per_class", f"PRE-mutation, {regime}")
    if os.path.exists(args.post):
        post = pd.read_csv(args.post)
        run(post, "pre_per_class", f"PRE-mutation (0b run), {post.regime.iloc[0]}")
        run(post, "post_per_class", f"POST-mutation, {post.regime.iloc[0]}")
    if os.path.exists(args.permute):
        pm = pd.read_csv(args.permute)
        if "child_per_class" in pm and "labels_a" in pm:
            pm = pm.copy()
            pm["mean_parent"] = 0.5 * (pm.a_per_task + pm.b_per_task)
            pm = pd.concat([pm, pm.apply(lambda r: retention(r, "child_per_class"), axis=1)], axis=1)
            print(f"\n==================== SESiL permute merged trunk (pre-mutation, exp2d) ====================")
            print(f"  retention shared {pm.ret_shared.mean():+.4f}  disjoint {pm.ret_disjoint.mean():+.4f}  all {pm.ret_all.mean():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
