"""Stage -1, the D- question asked fairly.

Natural specialists put only 12-18% of their energy into D- (opposite-sign
direct overlap), so "D- is the most damaging type" (proposal 5.1 (ii)) was
never tested where D- is the bulk. Here D- is produced by REAL training:

  A          digits {0,1,2}, normal images
  B_invert   digits {0,1,2}, pixel-INVERTED images.  Normalised x' = c - x, so
             to compute the same first-layer activations B must learn
             W' ~ -W: fc1's task vector flips sign. Genuine opposite-sign
             updates from a genuine task.
  B_disjoint digits {3,4,5}, normal          -- the usual pair (control)
  B_same     digits {0,1,2}, normal, other seed -- same task twice (D+ control)

Measured, 5 seeds:
  * type shares per scenario
  * directional: A + one type of B  ->  A's accuracy on its normal digits
  * symmetric merge with alpha_D- in {0, 0.5, 1.0, 1.2} (others 0.5, label
    head), scored on A's task, on B's task, pre and post mutation

    python -m globa.sandbox_dminus
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from globa.operator import TYPES  # noqa: E402
from globa.sandbox import A_DIGITS, MLP, acc_on, decompose_layer, energy_shares, mnist, subset, train  # noqa: E402
from globa.sandbox_v2 import BACKBONE, make_specialist, merge_head, train_rotation  # noqa: E402

C_INV = (1.0 - 0.1307) / 0.3081 + 0.1307 / 0.3081      # x' = C_INV - x  <=>  pixel p -> 255 - p


def invert(X):
    return C_INV - X


def sym_merge(core, decs, sd_a, sd_b, alpha, digits_a, digits_b):
    from globa.sandbox import assemble  # noqa: F401  (assemble expects a recipe name; do it inline)
    sd = {}
    for k in BACKBONE:
        d = decs[k]
        S = d["Cp"] + d["Cq"]
        C = sum(alpha[t] * (S * d["masks"][t]) for t in TYPES)
        sd[k] = (core[k] + d["Pu"] @ C @ d["Pv"].T + 0.5 * (d["Rp"] + d["Rq"])).float()
    for k in ("fc1.bias", "fc2.bias"):
        sd[k] = (0.5 * (sd_a[k] + sd_b[k])).float()
    sd.update({k: v.float() for k, v in merge_head(sd_a, sd_b, digits_a, digits_b, "label").items()})
    return sd


def directional(core, decs, sd_a, sd_b, t):
    """A whole + type t of B (hybrid_t: A exact = Pu Cp Pv^T + Rp)."""
    sd = {}
    for k in BACKBONE:
        d = decs[k]
        C = d["Cp"] + d["Cq"] * d["masks"][t]
        sd[k] = (core[k] + d["Pu"] @ C @ d["Pv"].T + d["Rp"]).float()
    for k in ("fc1.bias", "fc2.bias", "fc3.weight", "fc3.bias"):
        sd[k] = sd_a[k].float()
    return sd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.sandbox_dminus")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--mutation-steps", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--out", default="sandbox_stage_minus1_dminus.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    dev = args.device; t0 = time.time()
    Xtr, Ytr, Xte, Yte = mnist(dev)
    XA, YA = subset(Xtr, Ytr, A_DIGITS); XAte = subset(Xte, Yte, A_DIGITS)
    X35, Y35 = subset(Xtr, Ytr, (3, 4, 5))
    A_D, B_DIS = set(A_DIGITS), {3, 4, 5}
    ALPHAS = {f"aDm{a}": {"A": .5, "B": .5, "C": .5, "E": .5, "D_plus": .5, "D_minus": a} for a in (0.0, 0.5, 1.0, 1.2)}

    rows = []
    probe = MLP().to(dev)
    for seed in args.seeds:
        torch.manual_seed(seed)
        rot = MLP().to(dev); rot.fc3 = nn.Linear(256, 4).to(dev)
        rot = train_rotation(rot, Xtr, 3000, 1e-3, 128, seed)
        bb = {k: v.detach().clone() for k, v in rot.state_dict().items() if k.startswith(("fc1.", "fc2."))}
        core = {k: v.double().cpu() for k, v in bb.items()}
        A = train(make_specialist(bb, dev, seed * 100 + 11), XA, YA, args.steps, 1e-3, 128, seed * 100 + 1)
        sd_a = {k: v.detach().double().cpu() for k, v in A.state_dict().items()}

        scenarios = {
            "B_invert":   (train(make_specialist(bb, dev, seed * 100 + 31), invert(XA), YA, args.steps, 1e-3, 128, seed * 100 + 3),
                           lambda m: acc_on(m, invert(Xte), Yte, A_DIGITS), A_D, invert(XA), YA),
            "B_disjoint": (train(make_specialist(bb, dev, seed * 100 + 41), X35, Y35, args.steps, 1e-3, 128, seed * 100 + 4),
                           lambda m: acc_on(m, Xte, Yte, (3, 4, 5)), B_DIS, X35, Y35),
            "B_same":     (train(make_specialist(bb, dev, seed * 100 + 51), XA, YA, args.steps, 1e-3, 128, seed * 100 + 5),
                           lambda m: acc_on(m, Xte, Yte, A_DIGITS), A_D, XA, YA),
        }
        for name, (B, acc_b, digits_b, XB, YB) in scenarios.items():
            sd_b = {k: v.detach().double().cpu() for k, v in B.state_dict().items()}
            decs = {k: decompose_layer(sd_a[k] - core[k], sd_b[k] - core[k], "hybrid_t", args.eta, args.eta) for k in BACKBONE}
            row = {"seed": seed, "scenario": name, "A_alone": acc_on(A, Xte, Yte, A_DIGITS), "B_alone": acc_b(B)}
            sh = {t: 0.0 for t in TYPES}; e_tot = 0.0
            for k in BACKBONE:
                s, _, e = energy_shares(decs[k]); e_tot += e
                for t in TYPES: sh[t] += s[t] * e
            for t in TYPES: row[f"E_{t}"] = sh[t] / e_tot
            # fc1 alone: where the sign flip must show up
            s1, _, _ = energy_shares(decs["fc1.weight"]); row["E_Dminus_fc1"] = s1["D_minus"]; row["E_Dplus_fc1"] = s1["D_plus"]
            # cosine of the two fc1 task vectors
            tp, tq = sd_a["fc1.weight"] - core["fc1.weight"], sd_b["fc1.weight"] - core["fc1.weight"]
            row["cos_fc1"] = float((tp * tq).sum() / (tp.norm() * tq.norm()))
            # directional: A + type t of B, A's accuracy
            for t in TYPES:
                probe.load_state_dict(directional(core, decs, sd_a, sd_b, t)); probe.eval()
                row[f"dA_{t}"] = acc_on(probe, Xte, Yte, A_DIGITS) - row["A_alone"]
            # symmetric, alpha_D- sweep
            XU = torch.cat((XA, XB)); YU = torch.cat((YA, YB))
            for an, alpha in ALPHAS.items():
                probe.load_state_dict(sym_merge(core, decs, sd_a, sd_b, alpha, A_D, digits_b)); probe.eval()
                row[f"{an}_preA"], row[f"{an}_preB"] = acc_on(probe, Xte, Yte, A_DIGITS), acc_b(probe)
                train(probe, XU, YU, args.mutation_steps, 1e-3, 128, seed * 100 + 60)
                row[f"{an}_postA"], row[f"{an}_postB"] = acc_on(probe, Xte, Yte, A_DIGITS), acc_b(probe)
            rows.append(row)
        print(f"seed {seed} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    short = lambda t: t[0] if t[0] != "D" else t.replace("_plus", "+").replace("_minus", "-")
    print(f"\n{len(df)} rows -> {args.out}\n")
    for name in ("B_invert", "B_disjoint", "B_same"):
        d = df[df.scenario == name]
        print(f"=== {name}   A alone {d.A_alone.mean():.3f}   B alone {d.B_alone.mean():.3f}   cos(tau_A,tau_B) fc1 = {d.cos_fc1.mean():+.3f} ===")
        print("  type shares (backbone): " + "  ".join(f"{short(t)}={d[f'E_{t}'].mean():.2f}" for t in TYPES)
              + f"   | fc1 only: D-={d.E_Dminus_fc1.mean():.2f} D+={d.E_Dplus_fc1.mean():.2f}")
        print("  A + one type of B, dA (mean +- sd over seeds):")
        print("    " + "  ".join(f"{short(t)}={d[f'dA_{t}'].mean():+.3f}+-{d[f'dA_{t}'].std():.3f}" for t in TYPES))
        print(f"  symmetric merge, alpha_D- sweep (others 0.5, label head)  [A-task / B-task]")
        for an in ALPHAS:
            print(f"    {an:6s} pre  {d[f'{an}_preA'].mean():.3f}+-{d[f'{an}_preA'].std():.3f} / {d[f'{an}_preB'].mean():.3f}+-{d[f'{an}_preB'].std():.3f}"
                  f"    post {d[f'{an}_postA'].mean():.3f}+-{d[f'{an}_postA'].std():.3f} / {d[f'{an}_postB'].mean():.3f}+-{d[f'{an}_postB'].std():.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
