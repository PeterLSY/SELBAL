"""Stage 0b, joint (multivariate) test: do the six type shares ADD anything to
parent quality when used together, as the proposal's G(A,B) would?

Nested linear models over the 45 pairs
    M0:  outcome ~ 1 + mean_parent
    M1:  outcome ~ 1 + mean_parent + type_A + type_B + type_C + type_E + type_D_plus
         (type_D_minus dropped: the six shares sum to 1)
    M2:  M1 + typed_frac_mean + norm_ratio_med
reported with R^2, the nested F-test p-value for the added block, and the
leave-one-out cross-validated RMSE (the honest number with n = 45).

    python -m globa.analyze_joint
"""

from __future__ import annotations

import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

TYPES5 = ["type_A", "type_B", "type_C", "type_E", "type_D_plus"]


def fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float((resid ** 2).sum())


def loo_rmse(X, y):
    n = len(y); errs = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        beta, _ = fit(X[m], y[m]); errs.append(float(y[i] - X[i] @ beta))
    return float(np.sqrt(np.mean(np.square(errs))))


def add_cos(df, population, core_path):
    """cos(tau_A, tau_B) over all conv layers, the similarity of the two parents' updates."""
    import torch
    core = torch.load(core_path, map_location="cpu"); core = core.get("state_dict", core)
    ids = sorted(set(df.parent_a) | set(df.parent_b))
    sds = {}
    for i in ids:
        sd = torch.load(os.path.join(population, i, "resnet20x4_v0.pth.tar"), map_location="cpu")
        sds[i] = sd.get("state_dict", sd)
    keys = [k for k in core if k.endswith("weight") and sds[ids[0]][k].dim() == 4]
    tau = {i: torch.cat([(sds[i][k].double() - core[k].double()).flatten() for k in keys]) for i in ids}
    df["cos"] = [float(tau[a] @ tau[b] / (tau[a].norm() * tau[b].norm())) for a, b in zip(df.parent_a, df.parent_b)]
    return df


def nested(df, outcome, label, base=("mean_parent",)):
    y = df[outcome].to_numpy(float); n = len(y)
    one = np.ones((n, 1))
    X0 = np.hstack([one, df[list(base)].to_numpy(float)])
    X1 = np.hstack([X0, df[TYPES5].to_numpy(float)])
    X2 = np.hstack([X1, df[["typed_frac_mean", "norm_ratio_med"]].to_numpy(float)])
    sst = float(((y - y.mean()) ** 2).sum())
    rows = []
    prev = None
    for name, X in [("M0 " + "+".join(base), X0), ("M1 + 5 type shares", X1), ("M2 + typed_frac, norm_ratio", X2)]:
        beta, sse = fit(X, y); p = X.shape[1]
        r2 = 1 - sse / sst
        if prev is None:
            fp = float("nan")
        else:
            sse_prev, p_prev = prev
            df1, df2 = p - p_prev, n - p
            F = ((sse_prev - sse) / df1) / (sse / df2)
            fp = float(1 - stats.f.cdf(F, df1, df2))
        rows.append((name, p, r2, fp, loo_rmse(X, y)))
        prev = (sse, p)
    print(f"\n=== {label}: outcome = {outcome}  (n={n}) ===")
    print(f"  {'model':30s} {'k':>3s} {'R^2':>7s} {'F-test p (added block)':>24s} {'LOO RMSE':>9s}")
    for name, p, r2, fp, rm in rows:
        print(f"  {name:30s} {p:>3d} {r2:7.3f} {fp:>24.3f} {rm:9.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.analyze_joint")
    ap.add_argument("--pre", default="merge_dataset_exp2d_globa_v2.csv")
    ap.add_argument("--post", default="merge_dataset_exp2d_globa_mutated.csv")
    ap.add_argument("--gen10", default="merge_dataset_exp2d_gen10_globa.csv")
    args = ap.parse_args(argv)

    pre = pd.read_csv(args.pre); pre = pre[pre.regime == "average@0.80@label"].copy()
    pre["mean_parent"] = 0.5 * (pre.a_per_task + pre.b_per_task)
    nested(pre, "child_per_task", "PRE-mutation, gen 0")
    if os.path.exists(args.post):
        post = pd.read_csv(args.post); post["mean_parent"] = 0.5 * (post.a_per_task + post.b_per_task)
        nested(post, "post_per_task", "POST-mutation, gen 0")
    if os.path.exists(args.gen10):
        g = pd.read_csv(args.gen10); g = g[g.regime == "average@0.80@label"].copy()
        g["mean_parent"] = 0.5 * (g.a_per_task + g.b_per_task)
        nested(g, "child_per_task", "PRE-mutation, gen 10 (all agents 10-class)")
        g = add_cos(g, "runs_exp2d/seed0/permute/gen_10", "runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
        nested(g, "child_per_task", "PRE-mutation, gen 10, base = parent quality + cos(tau_A, tau_B)", base=("mean_parent", "cos"))
        nested(pre.assign(cos=add_cos(pre.copy(), "runs_exp2d/seed0/population", "runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")["cos"]),
               "child_per_task", "PRE-mutation, gen 0, base = parent quality + cos", base=("mean_parent", "cos"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
