"""Stage -1, MultiMNIST-style (proposal section 5.1): the sandbox on a
multi-attribute task, the shape CelebA has (section 4).

Each image is two different MNIST digits overlaid (pixel-wise max); the label
is a 10-way multi-hot vector. A specialist learns to detect a SUBSET of
digits-as-attributes with a per-attribute sigmoid head (BCE on its own
attributes only); the shared backbone is a DAE on composite images. Overlap
between A's and B's attribute sets is swept 0..3 as before. Everything else
mirrors sandbox_v3: 5 seeds, SESiL-style mutation on the union, copyA control.

Metric: mean per-attribute accuracy (threshold 0.5) over the attributes named.

    python -m globa.sandbox_multi --seeds 0 1 2 3 4 --mutation-steps 30
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
import torch.nn.functional as F  # noqa: E402

from globa.operator import TYPES  # noqa: E402
from globa.sandbox import A_DIGITS, B_DIGITS, MLP, decompose_layer, energy_shares, mnist  # noqa: E402
from globa.sandbox_v2 import BACKBONE, make_specialist, pretrain_backbone  # noqa: E402
from globa.sandbox_v3 import RECIPES, build_child  # noqa: E402


def compose(X, Y, n, seed):
    """n composite images: two different digits overlaid by max; multi-hot labels."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    i = torch.randint(0, len(X), (n,), generator=g, device=X.device)
    j = torch.randint(0, len(X), (n,), generator=g, device=X.device)
    same = Y[i] == Y[j]
    while same.any():                                   # resample collisions
        j[same] = torch.randint(0, len(X), (int(same.sum()),), generator=g, device=X.device)
        same = Y[i] == Y[j]
    Xc = torch.maximum(X[i], X[j])
    Yc = torch.zeros(n, 10, device=X.device); Yc[torch.arange(n), Y[i]] = 1; Yc[torch.arange(n), Y[j]] = 1
    return Xc, Yc


def train_bce(model, X, Y, attrs, steps, lr, bs, seed):
    g = torch.Generator(device=X.device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    a = torch.tensor(sorted(attrs), device=X.device)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (bs,), generator=g, device=X.device)
        loss = F.binary_cross_entropy_with_logits(model(X[idx])[:, a], Y[idx][:, a])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def attr_acc(model, X, Y, attrs):
    a = torch.tensor(sorted(attrs), device=X.device)
    pred = (model(X)[:, a] > 0).float()
    return (pred == Y[:, a]).float().mean().item()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.sandbox_multi")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--mutation-steps", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--out", default="sandbox_stage_minus1_multi.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    dev = args.device; t0 = time.time()
    Xtr, Ytr, Xte, Yte = mnist(dev)
    Xc, Yc = compose(Xtr, Ytr, 60000, 7)
    Xct, Yct = compose(Xte, Yte, 10000, 8)
    print(f"composites: train {tuple(Xc.shape)} test {tuple(Xct.shape)}; attributes per image = 2")

    rows = []
    child = MLP().to(dev)
    for seed in args.seeds:
        bb = pretrain_backbone("dae", Xc, 3000, dev, seed)
        core = {k: v.double().cpu() for k, v in bb.items()}
        A = train_bce(make_specialist(bb, dev, seed * 100 + 11), Xc, Yc, set(A_DIGITS), args.steps, 1e-3, 128, seed * 100 + 1)
        sd_a = {k: v.detach().double().cpu() for k, v in A.state_dict().items()}
        for k_shared, bd in B_DIGITS.items():
            B = train_bce(make_specialist(bb, dev, seed * 100 + 21 + k_shared), Xc, Yc, set(bd), args.steps, 1e-3, 128, seed * 100 + 2 + k_shared)
            sd_b = {k: v.detach().double().cpu() for k, v in B.state_dict().items()}
            union = set(A_DIGITS) | set(bd)
            decs = {k: decompose_layer(sd_a[k] - core[k], sd_b[k] - core[k], "hybrid_t", args.eta, args.eta) for k in BACKBONE}
            sh = {t: 0.0 for t in TYPES}; e_tot = 0.0
            for k in BACKBONE:
                s, _, e = energy_shares(decs[k]); e_tot += e
                for t in TYPES: sh[t] += s[t] * e
            for recipe in RECIPES:
                child.load_state_dict(build_child(core, sd_a, sd_b, decs, recipe, set(A_DIGITS), set(bd))); child.eval()
                pre = attr_acc(child, Xct, Yct, union)
                train_bce(child, Xc, Yc, union, args.mutation_steps, 1e-3, 128, seed * 100 + 50 + k_shared)
                rows.append({"seed": seed, "shared": k_shared, "recipe": recipe,
                             "A_on_union": attr_acc(A, Xct, Yct, union), "B_on_union": attr_acc(B, Xct, Yct, union),
                             "A_on_A": attr_acc(A, Xct, Yct, set(A_DIGITS)), "B_on_B": attr_acc(B, Xct, Yct, set(bd)),
                             "pre": pre, "post": attr_acc(child, Xct, Yct, union),
                             **{f"E_{t}": sh[t] / e_tot for t in TYPES}})
        print(f"seed {seed} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows -> {args.out}\n")
    short = lambda t: t[0] if t[0] != "D" else t.replace("_plus", "+").replace("_minus", "-")
    for k in sorted(df.shared.unique()):
        d = df[df.shared == k]
        print(f"shared={k}  parents: on own attrs {d.A_on_A.mean():.3f}/{d.B_on_B.mean():.3f}, on union {d.A_on_union.mean():.3f}/{d.B_on_union.mean():.3f}   "
              f"E: " + " ".join(f"{short(t)}={d[f'E_{t}'].mean():.2f}" for t in TYPES))
        agg = d.groupby("recipe")[["pre", "post"]].agg(["mean", "std"])
        for r in RECIPES:
            a = agg.loc[r]
            print(f"    {r:14s} pre {a[('pre','mean')]:.3f}+-{a[('pre','std')]:.3f}   post {a[('post','mean')]:.3f}+-{a[('post','std')]:.3f}")
        print()
    d = df[df.shared < 3]
    piv = d.pivot_table(index=["seed", "shared"], columns="recipe", values="post")
    print("=== post-mutation vs copyA, shared 0-2 (n=15) ===")
    for r in RECIPES:
        if r == "copyA": continue
        dd = piv[r] - piv["copyA"]
        print(f"  {r:14s} {dd.mean():+.4f}+-{dd.std():.4f}  wins {int((dd>0).sum())}/{len(dd)}")
    dd = piv["single+label"] - piv["avg+label"]; print(f"  single - avg   {dd.mean():+.4f}+-{dd.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
