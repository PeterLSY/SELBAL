"""Stage -1 v2: the sandbox in SESiL's regime, not the paper's.

v1 (globa/sandbox.py) fine-tunes specialists from a base that already
classifies all ten digits at 0.977 -- the paper's setting, where averaging
two specialists trivially recovers both tasks. SESiL's core is a
self-supervised BACKBONE with no head; every agent learns its classes from
scratch on a randomly initialised head. There, plain averaging destroyed
disjoint classes (Phase 2 on CIFAR: -0.379). This file rebuilds the sandbox
with that structure:

  backbone   fc1, fc2 pretrained by ROTATION PREDICTION (4-way, 0/90/180/270),
             rotation head discarded  -- the analogue of the SimSiam core
  specialist backbone copy + FRESH random 10-way head, all parameters trained
             on 3 digits           -- the analogue of a SESiL agent
  merge      backbone through the GLOBA operator (hybrid_t basis), head
             handled separately: `avg` (element-wise) or `label` (a class row
             comes from the parent that knows the class; shared -> average)
  metric     child accuracy on A|B, plus each side

    python -m globa.sandbox_v2 --steps 800 3200

Nothing here touches sesil/.
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from globa.operator import TYPES  # noqa: E402
from globa.sandbox import (A_DIGITS, B_DIGITS, MLP, SYMMETRIC, acc_on, assemble,  # noqa: E402
                           decompose_layer, energy_shares, mnist, subset, train)

BACKBONE = ("fc1.weight", "fc2.weight")
RECIPES = ["avg", "sum", "sym_A1", "sym_single1", "sym_single1_Dp7", "sym_Dp0"]
HEADS = ["avg", "label"]


def train_rotation(model, X, steps, lr, bs, seed):
    """Self-supervised pretext: predict the rotation applied to the image."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (bs,), generator=g, device=X.device)
        rot = torch.randint(0, 4, (bs,), generator=g, device=X.device)
        x = X[idx]
        xr = torch.stack([torch.rot90(x[i], int(rot[i]), dims=(1, 2)) for i in range(bs)])
        loss = F.cross_entropy(model(xr), rot)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def train_dae(model, X, steps, lr, bs, seed, noise=0.5):
    """Self-supervised pretext #2: denoising autoencoder. Encoder = fc1, fc2;
    a throw-away linear decoder reconstructs the clean image from a noisy one.
    Rotation prediction gave an MLP backbone no better than random init
    (probe 0.80 vs 0.82); this one does (see the probe printed by callers)."""
    dec = nn.Linear(256, 784).to(X.device)
    g = torch.Generator(device=X.device).manual_seed(seed)
    params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (bs,), generator=g, device=X.device)
        x = X[idx].flatten(1)
        xn = x + noise * torch.randn(x.shape, generator=g, device=X.device)
        h = F.relu(model.fc2(F.relu(model.fc1(xn))))
        loss = F.mse_loss(dec(h), x)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def pretrain_backbone(kind, X, steps, dev, seed):
    """Returns the fc1/fc2 state dict of a backbone that knows no classes."""
    m = MLP().to(dev)
    if kind == "rotation":
        m.fc3 = nn.Linear(256, 4).to(dev)
        m = train_rotation(m, X, steps, 1e-3, 128, seed)
    elif kind == "dae":
        m = train_dae(m, X, steps, 1e-3, 128, seed)
    elif kind == "random":
        pass
    else:
        raise ValueError(kind)
    return {k: v.detach().clone() for k, v in m.state_dict().items() if k.startswith(("fc1.", "fc2."))}


def linear_probe(backbone_sd, X, Y, Xte, Yte, dev, steps=500, seed=123):
    m = MLP().to(dev)
    m.load_state_dict({**m.state_dict(), **backbone_sd})
    for p in list(m.fc1.parameters()) + list(m.fc2.parameters()):
        p.requires_grad_(False)
    g = torch.Generator(device=dev).manual_seed(seed)
    opt = torch.optim.Adam(m.fc3.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (128,), generator=g, device=dev)
        loss = F.cross_entropy(m(X[idx]), Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return acc_on(m, Xte, Yte, tuple(range(10)))


def make_specialist(backbone_sd, dev, head_seed):
    m = MLP().to(dev)
    torch.manual_seed(head_seed)
    m.fc3.reset_parameters()                     # fresh random head, as in SESiL
    sd = m.state_dict(); sd.update(backbone_sd); m.load_state_dict(sd)
    return m


def merge_head(sd_a, sd_b, digits_a, digits_b, policy):
    """fc3.weight [10,256] and fc3.bias [10], row = class."""
    out = {}
    for k in ("fc3.weight", "fc3.bias"):
        a, b = sd_a[k], sd_b[k]
        if policy == "avg":
            out[k] = 0.5 * (a + b)
        else:
            r = 0.5 * (a + b)                     # shared or unknown-to-both -> average
            for c in range(10):
                in_a, in_b = c in digits_a, c in digits_b
                if in_a and not in_b:
                    r[c] = a[c]
                elif in_b and not in_a:
                    r[c] = b[c]
            out[k] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.sandbox_v2")
    ap.add_argument("--steps", type=int, nargs="+", default=[800, 3200])
    ap.add_argument("--pretext-steps", type=int, default=3000)
    ap.add_argument("--mode", default="hybrid_t")
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--out", default="sandbox_stage_minus1_v2.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    dev = args.device
    t0 = time.time()

    Xtr, Ytr, Xte, Yte = mnist(dev)

    # ---- backbone: rotation prediction, head discarded -------------------------
    rot = MLP().to(dev); rot.fc3 = nn.Linear(256, 4).to(dev)
    rot = train_rotation(rot, Xtr, args.pretext_steps, 1e-3, 128, args.seed)
    backbone_sd = {k: v.detach().clone() for k, v in rot.state_dict().items() if k.startswith(("fc1.", "fc2."))}
    probe = linear_probe(backbone_sd, Xtr, Ytr, Xte, Yte, dev)
    torch.manual_seed(args.seed + 999)
    rnd_sd = {k: v.detach().clone() for k, v in MLP().to(dev).state_dict().items() if k.startswith(("fc1.", "fc2."))}
    probe_rnd = linear_probe(rnd_sd, Xtr, Ytr, Xte, Yte, dev)
    print(f"backbone: rotation pretext {args.pretext_steps} steps; frozen linear probe on 10 digits = {probe:.4f}"
          f"   (random-init backbone, same probe: {probe_rnd:.4f})   ({time.time()-t0:.0f}s)\n")

    core = {k: v.double().cpu() for k, v in backbone_sd.items()}
    rows = []
    child = MLP().to(dev)
    short = lambda t: t[0] if t[0] != "D" else t.replace("_plus", "+").replace("_minus", "-")

    for steps in args.steps:
        XA, YA = subset(Xtr, Ytr, A_DIGITS)
        A = train(make_specialist(backbone_sd, dev, args.seed + 11), XA, YA, steps, 1e-3, 128, args.seed + 1)
        sd_a = {k: v.detach().double().cpu() for k, v in A.state_dict().items()}
        for k_shared, bd in B_DIGITS.items():
            XB, YB = subset(Xtr, Ytr, bd)
            B = train(make_specialist(backbone_sd, dev, args.seed + 21 + k_shared), XB, YB, steps, 1e-3, 128, args.seed + 2 + k_shared)
            sd_b = {k: v.detach().double().cpu() for k, v in B.state_dict().items()}
            union = tuple(sorted(set(A_DIGITS) | set(bd)))
            base = {"steps": steps, "shared": k_shared, "B_digits": "".join(map(str, bd)),
                    "A_on_A": acc_on(A, Xte, Yte, A_DIGITS), "B_on_B": acc_on(B, Xte, Yte, bd),
                    "A_on_union": acc_on(A, Xte, Yte, union), "B_on_union": acc_on(B, Xte, Yte, union)}

            decs = {k: decompose_layer(sd_a[k] - core[k], sd_b[k] - core[k], args.mode, args.eta, args.eta) for k in BACKBONE}
            sh = {t: 0.0 for t in TYPES}; e_tot = 0.0
            for k in BACKBONE:
                s, _, e = energy_shares(decs[k])
                for t in TYPES: sh[t] += s[t] * e
                e_tot += e
            for t in TYPES:
                base[f"E_{t}"] = sh[t] / e_tot if e_tot else 0.0

            print(f"steps={steps:4d} shared={k_shared} B={base['B_digits']}  A on A {base['A_on_A']:.3f}  B on B {base['B_on_B']:.3f}  "
                  f"A on A|B {base['A_on_union']:.3f}  B on A|B {base['B_on_union']:.3f}   E: "
                  + " ".join(f"{short(t)}={base[f'E_{t}']:.2f}" for t in TYPES))
            print(f"    {'backbone recipe':18s} | " + " | ".join(f"head={h:5s}  A|B    A-side  B-side" for h in HEADS))
            for recipe in RECIPES:
                row = dict(base, recipe=recipe)
                line = f"    {recipe:18s} |"
                for hp in HEADS:
                    sd = {}
                    for k in BACKBONE:                                    # backbone weights via GLOBA
                        sd[k] = (core[k] + assemble(decs[k], recipe)).float()
                    for k in ("fc1.bias", "fc2.bias"):
                        sd[k] = (0.5 * (sd_a[k] + sd_b[k])).float()
                    sd.update({k: v.float() for k, v in merge_head(sd_a, sd_b, set(A_DIGITS), set(bd), hp).items()})
                    child.load_state_dict(sd); child.eval()
                    u, a_side, b_side = acc_on(child, Xte, Yte, union), acc_on(child, Xte, Yte, A_DIGITS), acc_on(child, Xte, Yte, bd)
                    row[f"union_head_{hp}"] = u; row[f"Aside_head_{hp}"] = a_side; row[f"Bside_head_{hp}"] = b_side
                    line += f"            {u:.3f}  {a_side:.3f}  {b_side:.3f} |"
                rows.append(row)
                print(line)
            print()
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}   ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
