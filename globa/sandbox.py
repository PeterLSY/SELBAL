"""Stage -1 (proposal section 5.1): does the decomposition mean what GLOBA
claims, in GLOBA's native habitat?

MLP 784 -> 512 -> 256 -> 10, linear layers only, MNIST. A shared base trained
on all ten digits; two specialists fine-tuned from it on 3-digit subsets whose
overlap is swept 0..3 shared digits. Protocol as in the proposal and in the
paper's Tables 4-5: keep specialist A's task vector whole, add EXACTLY ONE
overlap type of B's, measure A's accuracy on A's own digits. Two symmetric
Strategy-3 probes are added on top (see RECIPES).

Basis modes, run side by side on identical specialists:
  full    spec v1 -- thin SVD, no truncation, numerical-rank basis (lossless,
          but degenerate whenever out <= in: [U_A U_B] has all singular values
          sqrt(2), so the output-side basis is an arbitrary rotation)
  code    the reference implementation -- truncate 0.90/0.99, basis 0.999,
          prune 0.95/0.80 (structured, lossy)
  hybrid  [Spec] -- truncate 0.90/0.90 to define the ANALYSED subspace; project,
          prune and classify inside it; everything pruned away or outside the
          subspace is a residual carried by plain 0.5 averaging. alpha == 0.5
          reproduces plain averaging exactly for any eta; alpha == 1 reproduces
          tau_A + tau_B exactly; types matter only where alpha != 0.5.
  ordered truncate-for-order 0.99 + complete to full span (kept for the record;
          rejected on 2026-09-08 -- completion re-creates the degenerate rows)

    python -m globa.sandbox --steps 200 800 3200

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
from torchvision import datasets, transforms  # noqa: E402

from globa.operator import TYPES, _orthonormal_basis, classify, prune  # noqa: E402

A_DIGITS = (0, 1, 2)
B_DIGITS = {0: (3, 4, 5), 1: (2, 3, 4), 2: (1, 2, 3), 3: (0, 1, 2)}
LAYERS = ("fc1.weight", "fc2.weight", "fc3.weight")

# directional (paper protocol): A whole + one type of B; then two references
DIRECTIONAL = list(TYPES) + ["sum", "avg"]
# symmetric Strategy-3 probes: alpha per type, residual (hybrid only) at 0.5
SYMMETRIC = {
    "sym_A1": {"A": 1.0, "B": .5, "C": .5, "E": .5, "D_plus": .5, "D_minus": .5},
    "sym_Dp0": {"A": .5, "B": .5, "C": .5, "E": .5, "D_plus": 0.0, "D_minus": .5},
    # single-parent cells whole, colliding cells averaged: "sum where disjoint,
    # average where overlapping". The natural two-parent Strategy 3.
    "sym_single1": {"A": 1.0, "B": 1.0, "C": 1.0, "E": 1.0, "D_plus": .5, "D_minus": .5},
    # same, but D+ leaning towards full (the paper's 0.7 for beneficial types)
    "sym_single1_Dp7": {"A": 1.0, "B": 1.0, "C": 1.0, "E": 1.0, "D_plus": .7, "D_minus": .5},
}


# --------------------------------------------------------------------------
# model / data / training
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.flatten(1))))))


def mnist(device):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    # tensor path, not per-item PIL decoding
    Xtr = ((tr.data.float() / 255 - 0.1307) / 0.3081).unsqueeze(1).to(device); Ytr = tr.targets.to(device)
    Xte = ((te.data.float() / 255 - 0.1307) / 0.3081).unsqueeze(1).to(device); Yte = te.targets.to(device)
    return Xtr, Ytr, Xte, Yte


def train(model, X, Y, steps, lr, bs, seed):
    g = torch.Generator(device=X.device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (bs,), generator=g, device=X.device)
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def acc_on(model, X, Y, digits):
    m = torch.isin(Y, torch.tensor(digits, device=Y.device))
    return (model(X[m]).argmax(1) == Y[m]).float().mean().item()


def subset(X, Y, digits):
    m = torch.isin(Y, torch.tensor(digits, device=Y.device))
    return X[m], Y[m]


# --------------------------------------------------------------------------
# basis variants
# --------------------------------------------------------------------------
def _truncate(U, S, Vh, energy):
    if energy >= 1.0:
        return U, S, Vh
    k = max(1, int((torch.cumsum(S ** 2, 0) / (S ** 2).sum() <= energy).sum().item()))
    return U[:, :k], S[:k], Vh[:k]


def _energy_basis(M, thr=0.999):
    P, D, _ = torch.linalg.svd(M, full_matrices=False)
    k = int(torch.searchsorted(torch.cumsum(D ** 2, 0) / (D ** 2).sum(), thr).item()) + 1
    return P[:, :min(k, P.shape[1])]


def _complete(lead, full):
    resid = full - lead @ (lead.T @ full)
    if resid.norm() <= 1e-10 * max(full.norm().item(), 1e-30):
        return lead
    return torch.cat((lead, _orthonormal_basis(resid)), 1)


def make_basis(tau_p, tau_q, mode, trunc=(0.90, 0.99)):
    Up, Sp, Vhp = torch.linalg.svd(tau_p, full_matrices=False)
    Uq, Sq, Vhq = torch.linalg.svd(tau_q, full_matrices=False)
    if mode == "full":
        return _orthonormal_basis(torch.cat((Up, Uq), 1)), _orthonormal_basis(torch.cat((Vhp.T, Vhq.T), 1))
    if mode in ("code", "hybrid"):
        Up_, _, Vhp_ = _truncate(Up, Sp, Vhp, trunc[0]); Uq_, _, Vhq_ = _truncate(Uq, Sq, Vhq, trunc[1])
        return _energy_basis(torch.cat((Up_, Uq_), 1)), _energy_basis(torch.cat((Vhp_.T, Vhq_.T), 1))
    if mode == "ordered":
        Up_, _, Vhp_ = _truncate(Up, Sp, Vhp, 0.99); Uq_, _, Vhq_ = _truncate(Uq, Sq, Vhq, 0.99)
        lead_u = _orthonormal_basis(torch.cat((Up_, Uq_), 1)); lead_v = _orthonormal_basis(torch.cat((Vhp_.T, Vhq_.T), 1))
        return _complete(lead_u, torch.cat((Up, Uq), 1)), _complete(lead_v, torch.cat((Vhp.T, Vhq.T), 1))
    raise ValueError(mode)


def decompose_layer(tau_p, tau_q, mode, eta_p, eta_q):
    """One layer, once. Returns everything every recipe needs."""
    if mode == "code":
        # the reference projects the TRUNCATED task vectors and drops the rest
        Pu, Pv = make_basis(tau_p, tau_q, "code", (0.90, 0.99))
        Up, Sp, Vhp = torch.linalg.svd(tau_p, full_matrices=False); Up, Sp, Vhp = _truncate(Up, Sp, Vhp, 0.90)
        Uq, Sq, Vhq = torch.linalg.svd(tau_q, full_matrices=False); Uq, Sq, Vhq = _truncate(Uq, Sq, Vhq, 0.99)
        tp_in, tq_in = Up @ torch.diag(Sp) @ Vhp, Uq @ torch.diag(Sq) @ Vhq
        Rp = Rq = None
    elif mode == "hybrid":
        Pu, Pv = make_basis(tau_p, tau_q, "hybrid", (0.90, 0.90))
        tp_in, tq_in = tau_p, tau_q                       # project the FULL vectors
    elif mode == "hybrid_t":
        # structure from `code` (classify the TRUNCATED vectors), losslessness
        # from `hybrid` (residual = whatever the typed cells do not carry)
        Pu, Pv = make_basis(tau_p, tau_q, "hybrid", (0.90, 0.90))
        Up, Sp, Vhp = torch.linalg.svd(tau_p, full_matrices=False); Up, Sp, Vhp = _truncate(Up, Sp, Vhp, 0.90)
        Uq, Sq, Vhq = torch.linalg.svd(tau_q, full_matrices=False); Uq, Sq, Vhq = _truncate(Uq, Sq, Vhq, 0.90)
        tp_in, tq_in = Up @ torch.diag(Sp) @ Vhp, Uq @ torch.diag(Sq) @ Vhq
    else:
        Pu, Pv = make_basis(tau_p, tau_q, mode)
        tp_in, tq_in = tau_p, tau_q
    Cp = prune(Pu.T @ tp_in @ Pv, eta_p)
    Cq = prune(Pu.T @ tq_in @ Pv, eta_q)
    if mode in ("hybrid", "hybrid_t"):
        # residual = everything not sitting in a typed (retained) cell
        Rp = tau_p - Pu @ Cp @ Pv.T
        Rq = tau_q - Pu @ Cq @ Pv.T
    elif mode != "code":
        Rp = Rq = None
    return dict(Pu=Pu, Pv=Pv, Cp=Cp, Cq=Cq, masks=classify(Cp, Cq), Rp=Rp, Rq=Rq,
                tau_p=tau_p, tau_q=tau_q)


def assemble(dec, recipe):
    """Merged task vector for one layer under one recipe."""
    Pu, Pv, Cp, Cq, m = dec["Pu"], dec["Pv"], dec["Cp"], dec["Cq"], dec["masks"]
    hybrid = dec["Rp"] is not None
    if recipe in SYMMETRIC:
        a = SYMMETRIC[recipe]
        S = Cp + Cq
        C = sum(a[t] * (S * m[t]) for t in TYPES)
        out = Pu @ C @ Pv.T
        return out + 0.5 * (dec["Rp"] + dec["Rq"]) if hybrid else out
    if recipe == "sum":
        out = Pu @ (Cp + Cq) @ Pv.T
        return out + dec["Rp"] + dec["Rq"] if hybrid else out
    if recipe == "avg":
        out = Pu @ (0.5 * (Cp + Cq)) @ Pv.T
        return out + 0.5 * (dec["Rp"] + dec["Rq"]) if hybrid else out
    # directional: A whole + one type of B
    out = Pu @ (Cp + Cq * m[recipe]) @ Pv.T
    return out + dec["Rp"] if hybrid else out


def energy_shares(dec):
    Cq, m = dec["Cq"], dec["masks"]
    e = float((Cq * Cq).sum())
    shares = {t: (float(((Cq * m[t]) ** 2).sum()) / e if e else 0.0) for t in TYPES}
    typed_q = e / float((dec["tau_q"] ** 2).sum())        # how much of B is analysed at all
    return shares, typed_q, e


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.sandbox")
    ap.add_argument("--steps", type=int, nargs="+", default=[200, 800, 3200])
    ap.add_argument("--base-steps", type=int, default=3000)
    ap.add_argument("--modes", nargs="+", default=["full", "code", "hybrid", "hybrid_t"])
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--out", default="sandbox_stage_minus1.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    dev = args.device

    t0 = time.time()
    Xtr, Ytr, Xte, Yte = mnist(dev)
    base = train(MLP().to(dev), Xtr, Ytr, args.base_steps, 1e-3, 128, args.seed)
    base_sd = {k: v.detach().double().cpu() for k, v in base.state_dict().items()}
    print(f"base: all-digit {acc_on(base, Xte, Yte, tuple(range(10))):.4f}  "
          f"A's digits {acc_on(base, Xte, Yte, A_DIGITS):.4f}   ({time.time()-t0:.0f}s)\n")

    recipes = DIRECTIONAL + list(SYMMETRIC)
    rows = []
    probe = MLP().to(dev)
    for steps in args.steps:
        XA, YA = subset(Xtr, Ytr, A_DIGITS)
        A = train(deepcopy(base), XA, YA, steps, 1e-3, 128, args.seed + 1)
        sd_a = {k: v.detach().double().cpu() for k, v in A.state_dict().items()}
        base_acc = acc_on(A, Xte, Yte, A_DIGITS)
        for k_shared, bd in B_DIGITS.items():
            XB, YB = subset(Xtr, Ytr, bd)
            B = train(deepcopy(base), XB, YB, steps, 1e-3, 128, args.seed + 2 + k_shared)
            sd_b = {k: v.detach().double().cpu() for k, v in B.state_dict().items()}
            for mode in args.modes:
                eta_p, eta_q = (0.95, 0.80) if mode == "code" else (args.eta, args.eta)
                # decompose each layer ONCE
                decs = {k: decompose_layer(sd_a[k] - base_sd[k], sd_b[k] - base_sd[k], mode, eta_p, eta_q)
                        for k in LAYERS}
                sh = {t: 0.0 for t in TYPES}; e_tot = 0.0; typed_num = 0.0; typed_den = 0.0
                for k in LAYERS:
                    s, tq_frac, e = energy_shares(decs[k])
                    for t in TYPES: sh[t] += s[t] * e
                    e_tot += e
                    typed_num += e; typed_den += float((decs[k]["tau_q"] ** 2).sum())
                union = tuple(sorted(set(A_DIGITS) | set(bd)))
                jA, jB = acc_on(A, Xte, Yte, union), acc_on(B, Xte, Yte, union)
                row = {"steps": steps, "shared": k_shared, "B_digits": "".join(map(str, bd)),
                       "mode": mode, "eta_p": eta_p, "eta_q": eta_q, "A_alone": base_acc,
                       "jA_alone": jA, "jB_alone": jB, "typed_frac_B": typed_num / typed_den}
                for t in TYPES:
                    row[f"E_{t}"] = sh[t] / e_tot if e_tot else 0.0
                for recipe in recipes:
                    sd = {}
                    for k in base_sd:
                        if k in LAYERS:
                            sd[k] = (base_sd[k] + assemble(decs[k], recipe)).float()
                        else:                                   # biases
                            sym = recipe in SYMMETRIC or recipe == "avg"
                            sd[k] = (0.5 * (sd_a[k] + sd_b[k]) if sym else sd_a[k]).float()
                    probe.load_state_dict(sd); probe.eval()
                    row[f"d_{recipe}"] = acc_on(probe, Xte, Yte, A_DIGITS) - base_acc   # A's retention
                    row[f"j_{recipe}"] = acc_on(probe, Xte, Yte, union)               # child on A|B
                rows.append(row)
                short = lambda t: t[0] if t[0] != "D" else t.replace("_plus", "+").replace("_minus", "-")
                print(f"steps={steps:4d} shared={k_shared} {mode:8s} A={base_acc:.3f} | dA: "
                      + " ".join(f"{short(t)}={row[f'd_{t}']:+.3f}" for t in TYPES)
                      + f" sum={row['d_sum']:+.3f} avg={row['d_avg']:+.3f}"
                      + f" | JOINT on A|B: parents {jA:.3f}/{jB:.3f}  avg={row['j_avg']:.3f} sum={row['j_sum']:.3f}"
                      + f" single1={row['j_sym_single1']:.3f} single1_Dp7={row['j_sym_single1_Dp7']:.3f} A1={row['j_sym_A1']:.3f}"
                      + " | E: " + " ".join(f"{short(t)}={row[f'E_{t}']:.2f}" for t in TYPES))
        print()
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}   ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
