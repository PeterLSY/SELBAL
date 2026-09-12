"""Is the sandbox's decomposition the authors' decomposition, on the sandbox's
own matrices? Same verbatim transcription as globa/check_reference.py, applied
to MLP specialists trained exactly as in globa/sandbox_v2.py.

If (ii) "D- is the most damaging" and (iii) "the full sum lies in between" do
not reproduce, this is the first thing to rule out.

    python -m globa.check_reference_mlp
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from globa.check_reference import reference_decompose, _trunc  # noqa: E402
from globa.operator import TYPES, classify, prune  # noqa: E402
from globa.sandbox import A_DIGITS, B_DIGITS, MLP, make_basis, mnist, subset, train  # noqa: E402
from globa.sandbox_v2 import make_specialist, train_rotation  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    Xtr, Ytr, Xte, Yte = mnist(dev)
    rot = MLP().to(dev); rot.fc3 = nn.Linear(256, 4).to(dev)
    rot = train_rotation(rot, Xtr, 3000, 1e-3, 128, 0)
    bb = {k: v.detach().clone() for k, v in rot.state_dict().items() if k.startswith(("fc1.", "fc2."))}
    core = {k: v.double().cpu() for k, v in bb.items()}
    XA, YA = subset(Xtr, Ytr, A_DIGITS)
    A = train(make_specialist(bb, dev, 11), XA, YA, 800, 1e-3, 128, 1)
    sd_a = {k: v.detach().double().cpu() for k, v in A.state_dict().items()}

    print(f"{'pair':10s} {'layer':6s} {'shape':>10s} {'k1':>3s} {'k2':>3s} {'basis':>9s} {'|C1 diff|':>10s} {'|C2 diff|':>10s} {'types==':>8s}")
    worst, ok_all = 0.0, True
    for k_shared, bd in B_DIGITS.items():
        XB, YB = subset(Xtr, Ytr, bd)
        B = train(make_specialist(bb, dev, 21 + k_shared), XB, YB, 800, 1e-3, 128, 2 + k_shared)
        sd_b = {k: v.detach().double().cpu() for k, v in B.state_dict().items()}
        for layer in ("fc1.weight", "fc2.weight"):
            tp, tq = sd_a[layer] - core[layer], sd_b[layer] - core[layer]
            ref = reference_decompose(tp, tq)
            Pu, Pv = make_basis(tp, tq, "code")
            Cp = prune(Pu.T @ _trunc(tp, 0.90) @ Pv, 0.95)
            Cq = prune(Pu.T @ _trunc(tq, 0.99) @ Pv, 0.80)
            ours = classify(Cp, Cq)
            d1 = ((ref["Pu"] @ ref["C1"] @ ref["Pv"].T - Pu @ Cp @ Pv.T).norm() / (ref["Pu"] @ ref["C1"] @ ref["Pv"].T).norm()).item()
            d2 = ((ref["Pu"] @ ref["C2"] @ ref["Pv"].T - Pu @ Cq @ Pv.T).norm() / (ref["Pu"] @ ref["C2"] @ ref["Pv"].T).norm()).item()
            ok = True
            for t in TYPES:
                e_ref = ((ref["C2"] * ref["masks"][t]) ** 2).sum().item()
                e_our = ((Cq * (ours[t] & (Cq != 0))) ** 2).sum().item()
                ok &= abs(e_ref - e_our) <= 1e-9 * max(e_ref, e_our, 1e-30)
            worst = max(worst, d1, d2); ok_all &= ok
            print(f"shared={k_shared}   {layer[:3]:6s} {str(tuple(tp.shape)):>10s} {ref['k1']:3d} {ref['k2']:3d} "
                  f"{str((ref['Pu'].shape[1], ref['Pv'].shape[1])):>9s} {d1:10.1e} {d2:10.1e} {'yes' if ok else 'NO':>8s}")
    print(f"\nworst reconstruction difference {worst:.1e}; per-type energies identical: {ok_all}")
    return 0 if (worst < 1e-8 and ok_all) else 1


if __name__ == "__main__":
    raise SystemExit(main())
