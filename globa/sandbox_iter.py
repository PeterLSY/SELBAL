"""Stage -1: what happens to the decomposition when merging is ITERATED.

Spec section 7.7 left open: tau = theta - theta_core with theta_core fixed at
generation 0. After several generations the two parents share far more recent
ancestry than that core, and the whole population's collective drift shows up
in every layer as same-sign overlap (D+). This script runs a small evolution
in the sandbox and watches the decomposition under two definitions of the core:

  fixed    theta_core = the generation-0 backbone (what Phase 4 does)
  popmean  theta_core = mean of the current parents' backbones (refreshed each
           generation)

Population of 4 MLP specialists (digit sets {0,1,2} {2,3,4} {4,5,6} {6,7,8}),
each generation forms 2 pairs (rotating), each pair yields 2 children through
the PRODUCTION operator (globa.operator.merge, label head), children are
mutated on their class union (30 steps), population size stays 4. Recipes
average and single-full, plus copyA (SESiL's de-facto crossover) as control.

    python -m globa.sandbox_iter --gens 6 --seeds 0 1 2
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

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from globa.operator import TYPES, MergeConfig, merge  # noqa: E402
from globa.sandbox import MLP, acc_on, mnist, subset, train  # noqa: E402
from globa.sandbox_v2 import make_specialist, pretrain_backbone  # noqa: E402

DIGITS = [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)]
ALL9 = tuple(range(9))
W = ("fc1.weight", "fc2.weight")


def sd_of(m):
    return {k: v.detach().double().cpu() for k, v in m.state_dict().items()}


def drift(sd, core0):
    return float(torch.sqrt(sum(((sd[k] - core0[k]) ** 2).sum() for k in W)))


def agg_stats(stats):
    w_sum, sh, ratios, typed = 0.0, {t: 0.0 for t in TYPES}, [], []
    for s in stats.values():
        w = s["norm_p"] ** 2 + s["norm_q"] ** 2
        for t in TYPES: sh[t] += s["energy_frac"][t] * w
        w_sum += w; ratios.append(s["norm_ratio"]); typed += [s["typed_frac_p"], s["typed_frac_q"]]
    out = {f"E_{t}": sh[t] / w_sum for t in TYPES}
    out["single_cells"] = sum(out[f"E_{t}"] for t in ("A", "B", "C", "E"))
    out["norm_ratio"] = sum(ratios) / len(ratios); out["typed_frac"] = sum(typed) / len(typed)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.sandbox_iter")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--mutation-steps", type=int, default=30)
    ap.add_argument("--pretext", default="dae")
    ap.add_argument("--out", default="sandbox_stage_minus1_iter.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    dev = args.device; t0 = time.time()
    Xtr, Ytr, Xte, Yte = mnist(dev)
    subsets = {d: subset(Xtr, Ytr, d) for d in DIGITS}

    rows = []
    probe = MLP().to(dev)
    for seed in args.seeds:
        bb = pretrain_backbone(args.pretext, Xtr, 3000, dev, seed)
        core0 = {k: v.double().cpu() for k, v in bb.items()}
        gen0 = []
        for i, d in enumerate(DIGITS):
            m = train(make_specialist(bb, dev, seed * 100 + 10 + i), *subsets[d], args.steps, 1e-3, 128, seed * 100 + i)
            gen0.append((sd_of(m), set(d)))

        for core_kind in ("fixed", "popmean"):
            for recipe in ("average", "single-full", "copyA"):
                pop = [(deepcopy(sd), set(c)) for sd, c in gen0]
                for g in range(1, args.gens + 1):
                    pairs = [(0, 1), (2, 3)] if g % 2 else [(1, 2), (3, 0)]
                    if core_kind == "fixed":
                        core = core0
                    else:
                        core = {k: sum(pop[i][0][k] for i in range(4)) / 4 for k in W}
                    new_pop = []
                    for (i, j) in pairs:
                        (sa, ca), (sb, cb) = pop[i], pop[j]
                        union = tuple(sorted(ca | cb))
                        rec = {"seed": seed, "core": core_kind, "recipe": recipe, "gen": g, "pair": f"{i}{j}",
                               "n_classes": len(union),
                               "tauA": drift(sa, core), "tauB": drift(sb, core),          # w.r.t. the core USED
                               "driftA0": drift(sa, core0), "driftB0": drift(sb, core0)}    # w.r.t. gen-0 core
                        if recipe == "copyA":
                            child_sd = {k: v.clone() for k, v in sa.items()}
                        else:
                            cfg = MergeConfig.preset(recipe, head="label", head_prefix="fc3.")
                            child_sd, stats = merge(sa, sb, core, cfg, ca, cb, return_stats=True)
                            rec.update(agg_stats(stats))
                        rec["child_drift0_pre"] = drift(child_sd, core0)
                        XU, YU = subset(Xtr, Ytr, union)
                        for c in range(2):                     # two children per couple, mutated apart
                            probe.load_state_dict({k: v.float() for k, v in child_sd.items()})
                            train(probe, XU, YU, args.mutation_steps, 1e-3, 128, seed * 1000 + g * 10 + i * 2 + c)
                            csd = sd_of(probe)
                            r = dict(rec, child=c, acc_own=acc_on(probe, Xte, Yte, union), acc_all9=acc_on(probe, Xte, Yte, ALL9),
                                     child_drift0_post=drift(csd, core0))
                            rows.append(r)
                            new_pop.append((csd, set(union)))
                    pop = new_pop
                print(f"seed {seed} core={core_kind:7s} recipe={recipe:11s} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows -> {args.out}\n")
    m = df[df.recipe != "copyA"]
    print("=== decomposition over generations (mean over seeds, pairs) ===")
    print(f"{'core':8s} {'recipe':12s} {'gen':>3s} {'|tau| wrt core':>14s} {'drift wrt core0':>15s} {'D+':>5s} {'D-':>5s} {'E':>5s} {'single':>6s} {'typed':>6s} {'ratio':>6s} {'acc9 post':>10s}")
    for (ck, rc), d0 in m.groupby(["core", "recipe"]):
        for g, d in d0.groupby("gen"):
            print(f"{ck:8s} {rc:12s} {g:3d} {0.5*(d.tauA+d.tauB).mean():14.2f} {0.5*(d.driftA0+d.driftB0).mean():15.2f} "
                  f"{d.E_D_plus.mean():5.2f} {d.E_D_minus.mean():5.2f} {d.E_E.mean():5.2f} {d.single_cells.mean():6.2f} "
                  f"{d.typed_frac.mean():6.2f} {d.norm_ratio.mean():6.2f} {d.acc_all9.mean():10.3f}")
        print()
    print("=== accuracy on all 9 digits after mutation, by generation (mean over seeds, children) ===")
    piv = df.pivot_table(index="gen", columns=["core", "recipe"], values="acc_all9")
    print(piv.round(3).to_string())
    print("\n=== class coverage: n_classes of children by generation ===")
    print(df.groupby("gen").n_classes.mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
