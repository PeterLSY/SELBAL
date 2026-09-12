"""Stage -1 v3: v2's regime, with the two gaps that decide whether the v2
numbers mean anything.

  * SEEDS      every v2 number was one seed; single-full beat average by
               0.01-0.02, which is within plausible seed noise. v3 sweeps
               seeds and reports mean +- sd.
  * MUTATION   v2 scored the child right after the merge. SESiL fine-tunes the
               child on the union of the parents' classes (2 epochs, Adam
               1e-3) before it is saved. v3 applies the same step and scores
               again, so the question "how much of the offline difference
               survives mutation" is answered in the sandbox.
  * copyA      SESiL's de-facto crossover (save_offspring writes head_models[0]
               = parent A) as a recipe: child = A, then mutated on A|B. This,
               not "A alone", is the baseline a merge has to beat.

    python -m globa.sandbox_v3 --seeds 0 1 2 3 4 --steps 800 3200 --mutation-steps 300

300 mutation steps at batch 128 ~ 2 epochs over 18-36k union images, matching
SESiL's 2 epochs. Nothing here touches sesil/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from globa.operator import TYPES  # noqa: E402
from globa.sandbox import A_DIGITS, B_DIGITS, MLP, SYMMETRIC, acc_on, assemble, decompose_layer, energy_shares, mnist, subset, train  # noqa: E402
from globa.sandbox_v2 import BACKBONE, linear_probe, make_specialist, merge_head, pretrain_backbone  # noqa: E402

# recipe -> (backbone rule, head rule). 'copyA' is SESiL as it actually runs.
RECIPES = {
    "copyA":          ("A",           "A"),
    "avg+avghead":    ("avg",         "avg"),
    "avg+label":      ("avg",         "label"),
    "single+label":   ("sym_single1", "label"),
    "sum+label":      ("sum",         "label"),
}


def build_child(core, sd_a, sd_b, decs, recipe, digits_a, digits_b):
    bb, hd = RECIPES[recipe]
    sd = {}
    for k in BACKBONE:
        if bb == "A":
            sd[k] = sd_a[k].float()
        else:
            sd[k] = (core[k] + assemble(decs[k], bb)).float()
    for k in ("fc1.bias", "fc2.bias"):
        sd[k] = (sd_a[k] if bb == "A" else 0.5 * (sd_a[k] + sd_b[k])).float()
    if hd == "A":
        sd.update({k: sd_a[k].float() for k in ("fc3.weight", "fc3.bias")})
    else:
        sd.update({k: v.float() for k, v in merge_head(sd_a, sd_b, digits_a, digits_b, hd).items()})
    return sd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.sandbox_v3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, nargs="+", default=[800, 3200])
    ap.add_argument("--pretext-steps", type=int, default=3000)
    ap.add_argument("--pretext", choices=["rotation", "dae", "random"], default="rotation")
    ap.add_argument("--mutation-steps", type=int, default=300)
    ap.add_argument("--mode", default="hybrid_t")
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--out", default="sandbox_stage_minus1_v3.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    dev = args.device
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = mnist(dev)

    rows = []
    child = MLP().to(dev)
    for seed in args.seeds:
        torch.manual_seed(seed)
        backbone_sd = pretrain_backbone(args.pretext, Xtr, args.pretext_steps, dev, seed)
        if seed == args.seeds[0]:
            print(f"pretext={args.pretext}: frozen linear probe on 10 digits = "
                  f"{linear_probe(backbone_sd, Xtr, Ytr, Xte, Yte, dev):.4f}")
        core = {k: v.double().cpu() for k, v in backbone_sd.items()}
        for steps in args.steps:
            XA, YA = subset(Xtr, Ytr, A_DIGITS)
            A = train(make_specialist(backbone_sd, dev, seed * 100 + 11), XA, YA, steps, 1e-3, 128, seed * 100 + 1)
            sd_a = {k: v.detach().double().cpu() for k, v in A.state_dict().items()}
            for k_shared, bd in B_DIGITS.items():
                XB, YB = subset(Xtr, Ytr, bd)
                B = train(make_specialist(backbone_sd, dev, seed * 100 + 21 + k_shared), XB, YB, steps, 1e-3, 128, seed * 100 + 2 + k_shared)
                sd_b = {k: v.detach().double().cpu() for k, v in B.state_dict().items()}
                union = tuple(sorted(set(A_DIGITS) | set(bd)))
                XU, YU = subset(Xtr, Ytr, union)
                decs = {k: decompose_layer(sd_a[k] - core[k], sd_b[k] - core[k], args.mode, args.eta, args.eta) for k in BACKBONE}
                sh = {t: 0.0 for t in TYPES}; e_tot = 0.0
                for k in BACKBONE:
                    s, _, e = energy_shares(decs[k]); e_tot += e
                    for t in TYPES: sh[t] += s[t] * e
                for recipe in RECIPES:
                    sd = build_child(core, sd_a, sd_b, decs, recipe, set(A_DIGITS), set(bd))
                    child.load_state_dict(sd); child.eval()
                    pre = acc_on(child, Xte, Yte, union)
                    pre_a, pre_b = acc_on(child, Xte, Yte, A_DIGITS), acc_on(child, Xte, Yte, bd)
                    # SESiL mutation: fine-tune the child on the union of the parents' classes
                    train(child, XU, YU, args.mutation_steps, 1e-3, 128, seed * 100 + 50 + k_shared)
                    post = acc_on(child, Xte, Yte, union)
                    post_a, post_b = acc_on(child, Xte, Yte, A_DIGITS), acc_on(child, Xte, Yte, bd)
                    rows.append({"seed": seed, "steps": steps, "shared": k_shared, "recipe": recipe,
                                 "A_on_union": acc_on(A, Xte, Yte, union), "B_on_union": acc_on(B, Xte, Yte, union),
                                 "pre": pre, "pre_A": pre_a, "pre_B": pre_b,
                                 "post": post, "post_A": post_a, "post_B": post_b,
                                 **{f"E_{t}": sh[t] / e_tot for t in TYPES}})
            print(f"seed {seed} steps {steps} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows -> {args.out}   ({time.time()-t0:.0f}s)\n")

    # ---- report: mean +- sd over seeds, pre and post mutation -------------------
    g = df.groupby(["steps", "shared", "recipe"])
    agg = g[["pre", "post"]].agg(["mean", "std"])
    for steps in args.steps:
        for k_shared in sorted(df.shared.unique()):
            d = df[(df.steps == steps) & (df.shared == k_shared)]
            pa, pb = d.A_on_union.mean(), d.B_on_union.mean()
            print(f"steps={steps} shared={k_shared}  parents on A|B {pa:.3f}/{pb:.3f}   "
                  f"E: " + " ".join(f"{t[0] if t[0]!='D' else t.replace('_plus','+').replace('_minus','-')}={d[f'E_{t}'].mean():.2f}" for t in TYPES))
            print(f"    {'recipe':14s} {'pre-mutation':>20s} {'post-mutation':>20s}")
            for recipe in RECIPES:
                a = agg.loc[(steps, k_shared, recipe)]
                print(f"    {recipe:14s} {a[('pre','mean')]:9.3f} ± {a[('pre','std')]:.3f}    "
                      f"{a[('post','mean')]:9.3f} ± {a[('post','std')]:.3f}")
            print()

    # ---- paired deltas vs copyA and vs avg+label, post-mutation ------------------
    print("=== post-mutation, paired over (seed, steps, shared): recipe - reference ===")
    piv = df.pivot_table(index=["seed", "steps", "shared"], columns="recipe", values="post")
    for ref in ("copyA", "avg+label"):
        for recipe in RECIPES:
            if recipe == ref: continue
            d = piv[recipe] - piv[ref]
            print(f"  {recipe:14s} - {ref:10s}: {d.mean():+.4f} ± {d.std():.4f}   wins {int((d>0).sum())}/{len(d)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
