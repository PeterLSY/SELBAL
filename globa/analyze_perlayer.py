"""Stage 0b, per-layer variant (proposal 5.2: "per layer and aggregated").

The aggregated type shares carry no signal. Before closing H0 the per-layer
shares get the same test: for each analysed layer, Spearman of that layer's
type-D-, type-A, type-D+ share and norm ratio against the realised outcome
(pre-mutation child per-task and excess from the Phase 2 v2 table; post-
mutation from the 0b table). Reports the strongest layer per feature with a
multiple-comparison note (21 layers x 4 features x 3 outcomes = 252 tests;
~13 would clear p < .05 by chance).

CPU only, float64; no evaluation is rerun.

    python -m globa.analyze_perlayer
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from globa.collect import classes_of, load_sd  # noqa: E402
from globa.operator import MergeConfig, merge  # noqa: E402

FEATS = ["D_minus", "A", "D_plus", "norm_ratio"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.analyze_perlayer")
    ap.add_argument("--population", default="runs_exp2d/seed0/population")
    ap.add_argument("--core", default="runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
    ap.add_argument("--pre", default="merge_dataset_exp2d_globa_v2.csv")
    ap.add_argument("--post", default="merge_dataset_exp2d_globa_mutated.csv")
    ap.add_argument("--out", default="merge_dataset_exp2d_globa_perlayer.csv")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    core = load_sd(args.core)
    cfg = MergeConfig.preset("average", eta=0.80, head="label")
    ids = sorted(d for d in os.listdir(args.population) if os.path.isdir(os.path.join(args.population, d)))
    sds = {i: load_sd(os.path.join(args.population, i, "resnet20x4_v0.pth.tar")) for i in ids}

    if os.path.exists(args.out):
        long = pd.read_csv(args.out)
        print(f"loaded {args.out} ({len(long)} rows)")
    else:
        rows, t0 = [], time.time()
        for n, (a, b) in enumerate(itertools.combinations(ids, 2), 1):
            _, st = merge(sds[a], sds[b], core, cfg, classes_of(a), classes_of(b), return_stats=True)
            for layer, s in st.items():
                rows.append({"parent_a": a, "parent_b": b, "layer": layer, "norm_ratio": s["norm_ratio"],
                             "w": s["norm_p"] ** 2 + s["norm_q"] ** 2, **{t: s["energy_frac"][t] for t in s["energy_frac"]}})
            print(f"[{n}/45] {a} x {b} ({time.time() - t0:.0f}s)", flush=True)
        long = pd.DataFrame(rows); long.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    pre = pd.read_csv(args.pre); pre = pre[pre.regime == "average@0.80@label"].copy()
    pre["mean_parent"] = 0.5 * (pre.a_per_task + pre.b_per_task)
    pre["excess_pre"] = pre.child_per_task - pre.mean_parent
    out = pre[["parent_a", "parent_b", "child_per_task", "excess_pre", "mean_parent"]]
    if os.path.exists(args.post):
        post = pd.read_csv(args.post)
        post["excess_post"] = post.post_per_task - 0.5 * (post.a_per_task + post.b_per_task)
        out = out.merge(post[["parent_a", "parent_b", "post_per_task", "excess_post"]], on=["parent_a", "parent_b"], how="left")
    tgts = [c for c in ["child_per_task", "excess_pre", "post_per_task", "excess_post"] if c in out]

    layers = list(dict.fromkeys(long.layer))
    print(f"\n{len(layers)} layers, {len(out)} pairs, features {FEATS}, outcomes {tgts}")
    res = []
    for layer in layers:
        d = long[long.layer == layer].merge(out, on=["parent_a", "parent_b"])
        for f in FEATS:
            if d[f].nunique() < 3:
                continue
            for t in tgts:
                r, p = spearmanr(d[f], d[t]); res.append({"layer": layer, "feature": f, "outcome": t, "rho": r, "p": p})
    res = pd.DataFrame(res)
    n_tests = len(res); n_sig = int((res.p < 0.05).sum())
    print(f"\n{n_tests} tests, {n_sig} with p<.05 (chance expectation {0.05 * n_tests:.1f}); Bonferroni threshold p<{0.05 / n_tests:.5f}: "
          f"{int((res.p < 0.05 / n_tests).sum())} pass")
    print("\nstrongest layer per feature x outcome:")
    best = res.loc[res.groupby(["feature", "outcome"]).rho.apply(lambda s: s.abs().idxmax())]
    print(best.sort_values(["feature", "outcome"]).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nmean |rho| over layers, per feature x outcome:")
    print(res.assign(a=res.rho.abs()).pivot_table(index="feature", columns="outcome", values="a").to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nper-layer type shares (mean over pairs), depth order:")
    sh = long.groupby("layer", sort=False)[["A", "B", "C", "E", "D_plus", "D_minus", "norm_ratio"]].mean()
    print(sh.to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
