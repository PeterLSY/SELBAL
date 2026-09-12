"""Invariants from docs/globa_operator_spec.md section 5 (v2), on real
checkpoints. Exit code 1 if any test fails.

    python -m globa.check
    python -m globa.check --population runs_exp2d/seed0/population --pair 0 1

Needs only torch, the checkpoints, and mapping.json (for the class sets).
Nothing here touches sesil/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy

import torch

from globa.operator import TYPES, MergeConfig, as_matrix, decompose, is_analysable, merge


def load_sd(path: str):
    sd = torch.load(path, map_location="cpu")
    return sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd


def members(pop: str):
    return sorted(d for d in os.listdir(pop) if os.path.isdir(os.path.join(pop, d)))


def classes_of(model_id: str, mapping_file="mapping.json"):
    with open(mapping_file) as f:
        m = json.load(f)
    v = m[model_id]
    if isinstance(v, str):
        return set(int(x) for x in v.split("_"))
    return set(int(x) for x in v)


def bitwise_equal(x: dict, y: dict) -> bool:
    return set(x) == set(y) and all(torch.equal(x[k], y[k]) for k in x)


def analysable_keys(sd, core, cfg):
    return [k for k in sd if is_analysable(k, sd[k], core, cfg)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="runs_exp2d/seed0/population")
    ap.add_argument("--core", default="runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
    ap.add_argument("--pair", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--ckpt", default="resnet20x4_v0.pth.tar")
    args = ap.parse_args()

    ids = members(args.population)
    a_id, b_id = ids[args.pair[0]], ids[args.pair[1]]
    A = load_sd(os.path.join(args.population, a_id, args.ckpt))
    B = load_sd(os.path.join(args.population, b_id, args.ckpt))
    core = load_sd(args.core)
    ca, cb = classes_of(a_id), classes_of(b_id)
    print(f"pair {a_id} {sorted(ca)}  x  {b_id} {sorted(cb)}   core {os.path.basename(args.core)}\n")

    results = []

    def record(tid, ok, detail=""):
        results.append((tid, ok))
        print(f"  {tid:3s} {'PASS' if ok else 'FAIL'}  {detail}")

    dflt = MergeConfig()          # average, eta .80, svd .90, basis .999, rho .5, head label
    keys_an = analysable_keys(A, core, dflt)
    head_keys = [k for k in A if k.startswith(dflt.head_prefix)]

    # ---- T1 symmetry, bitwise, all presets x eta x head -----------------------
    ok, bad = True, []
    for name in ("sum", "average", "orthogonal-full", "single-full"):
        for eta in (0.8, 1.0):
            for head in ("label", "average"):
                cfg = MergeConfig.preset(name, eta=eta, head=head)
                same = bitwise_equal(merge(A, B, core, cfg, ca, cb), merge(B, A, core, cfg, cb, ca))
                ok &= same
                if not same:
                    bad.append(f"{name}@{eta}/{head}")
    record("T1", ok, "merge(A,B) == merge(B,A) bitwise, 4 presets x 2 eta x 2 head"
           + (f"  failed: {bad}" if bad else ""))

    # ---- T2 average == plain averaging, for ANY eta (v2 property) --------------
    worst = 0.0
    for eta in (0.8, 1.0):
        cfg = MergeConfig.preset("average", eta=eta, head="average")
        child = merge(A, B, core, cfg, ca, cb)
        for k in keys_an:
            worst = max(worst, float((child[k].double() - 0.5 * (A[k].double() + B[k].double())).abs().max()))
        exact_non = all(torch.equal(child[k], 0.5 * (A[k] + B[k])) if A[k].is_floating_point() else True
                        for k in A if k not in keys_an)
    record("T2", worst < 1e-5 and exact_non,
           f"max|child - 0.5(A+B)| over eta in {{0.8, 1.0}} = {worst:.2e}; non-analysed exact: {exact_non}")

    # ---- T3 sum == A + B - core, for ANY eta -----------------------------------
    worst = 0.0
    for eta in (0.8, 1.0):
        child = merge(A, B, core, MergeConfig.preset("sum", eta=eta, head="average"), ca, cb)
        for k in keys_an:
            worst = max(worst, float((child[k].double() - (A[k].double() + B[k].double() - core[k].double())).abs().max()))
    record("T3", worst < 1e-5, f"max|child - (A + B - core)| over eta in {{0.8, 1.0}} = {worst:.2e}")

    # ---- T4 tau_B == 0, everything typed A, alpha_A = 1 -> child == A ----------
    B0 = deepcopy(A)
    for k in keys_an:
        B0[k] = core[k].clone()
    cfg = MergeConfig.preset("orthogonal-full", eta=1.0, svd_energy=1.0, basis_energy=1.0, head="average")
    child = merge(A, B0, core, cfg, ca, ca)
    worst = max(float((child[k].double() - A[k].double()).abs().max()) for k in keys_an)
    nbit = sum(torch.equal(child[k], A[k]) for k in keys_an)
    record("T4", worst < 1e-5, f"(svd_energy=1, eta=1) max|child - A| = {worst:.2e}; bitwise on {nbit}/{len(keys_an)}")

    # ---- T5 partition + T9 decomposition identity ------------------------------
    ok, worst_gap, worst_id, n = True, 0.0, 0.0, 0
    for k in keys_an:
        tp = as_matrix(A[k].double() - core[k].double()); tq = as_matrix(B[k].double() - core[k].double())
        Pu, Pv, Cp, Cq, masks, Rp, Rq = decompose(tp, tq, dflt)
        union = (Cp != 0) | (Cq != 0)
        stack = torch.stack([masks[t] for t in TYPES])
        ok &= bool((stack.sum(0) <= 1).all()) and bool(torch.equal(stack.any(0), union))
        S = Cp + Cq; tot = float((S * S).sum())
        parts = sum(float(((S * masks[t]) ** 2).sum()) for t in TYPES)
        worst_gap = max(worst_gap, abs(parts - tot) / tot if tot > 0 else 0.0)
        worst_id = max(worst_id, float(((Pu @ Cp @ Pv.T + Rp) - tp).norm() / tp.norm()),
                       float(((Pu @ Cq @ Pv.T + Rq) - tq).norm() / tq.norm()))
        n += 1
    record("T5", ok and worst_gap < 1e-9, f"{n} layers: masks disjoint & cover the union: {ok}; energy gap {worst_gap:.1e}")
    record("T9", worst_id < 1e-12, f"tau == Pu C Pv^T + R, worst rel. error {worst_id:.1e}  (hybrid_t is lossless)")

    # ---- T6 determinism --------------------------------------------------------
    record("T6", bitwise_equal(merge(A, B, core, dflt, ca, cb), merge(A, B, core, dflt, ca, cb)),
           "same inputs twice -> bitwise identical")

    # ---- T7 average never exceeds the mean parent norm -------------------------
    _, st = merge(A, B, core, MergeConfig.preset("average", eta=0.8), ca, cb, return_stats=True)
    r = max(s["norm_ratio"] for s in st.values())
    record("T7", r <= 1.0 + 1e-9, f"max ||tau_m|| / mean parent norm = {r:.4f} (<= 1)")

    # ---- T8 label-aware head ----------------------------------------------------
    child = merge(A, B, core, MergeConfig.preset("average", head="label"), ca, cb)
    ok = True
    for k in head_keys:
        for c in range(A[k].shape[0]):
            exp = A[k][c] if (c in ca and c not in cb) else B[k][c] if (c in cb and c not in ca) else 0.5 * (A[k][c] + B[k][c])
            ok &= torch.equal(child[k][c], exp)
    only_a, only_b, shared = sorted(ca - cb), sorted(cb - ca), sorted(ca & cb)
    record("T8", ok, f"head rows {only_a} from A, {only_b} from B, {shared} + unknown averaged: {ok}")

    # ---- summary under the default config ---------------------------------------
    _, st = merge(A, B, core, MergeConfig.preset("single-full"), ca, cb, return_stats=True)
    w = {t: 0.0 for t in TYPES}; wsum = 0.0; typed = []
    for s in st.values():
        ww = s["norm_p"] ** 2 + s["norm_q"] ** 2
        for t in TYPES: w[t] += s["energy_frac"][t] * ww
        wsum += ww; typed += [s["typed_frac_p"], s["typed_frac_q"]]
    ratios = sorted(s["norm_ratio"] for s in st.values())
    print(f"\n  default basis (svd .90, basis .999, eta .80) on this pair, {len(st)} layers:")
    print("   typed energy share:  " + "  ".join(f"{t}={w[t]/wsum:.3f}" for t in TYPES))
    print(f"   analysed fraction of each parent's task vector: {min(typed):.2f}-{max(typed):.2f}")
    print(f"   single-full norm ratio: min {ratios[0]:.3f} med {ratios[len(ratios)//2]:.3f} max {ratios[-1]:.3f}")

    failed = [t for t, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f"  FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
