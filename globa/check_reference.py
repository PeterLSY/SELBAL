"""Does our decomposition reproduce the authors' code, number for number?

`reference_decompose` is a verbatim transcription of the mathematics in
GLOBA/model_merging_methods/merging_methods.py, svd_merging, lines 141-269
(SVD -> truncation -> global basis -> projection -> pruning -> six masks),
stripped of the LLaMA model plumbing and the print statements. Nothing else
is changed: same thresholds, same cutoff conventions, same mask formulas.

Ours is run through globa.sandbox.make_basis("code") + globa.operator.prune
+ globa.operator.classify with the same thresholds. Their masks are defined
on C2 relative to C1; ours on supp(C1) | supp(C2) -- restricted to supp(C2)
they must coincide.

    python -m globa.check_reference
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from globa.check import load_sd, members
from globa.operator import TYPES, as_matrix, classify, prune
from globa.sandbox import make_basis


def reference_decompose(delta1, delta2,
                        svd_energy_threshold_model1=0.90, svd_energy_threshold_model2=0.99,
                        c_pruning_energy_c1=0.95, c_pruning_energy_c2=0.80,
                        global_basis_energy_threshold=0.999):
    # ---- lines 141-150 -------------------------------------------------------
    U1, S1, V1t = torch.linalg.svd(delta1, full_matrices=False); V1 = V1t.T
    U2, S2, V2t = torch.linalg.svd(delta2, full_matrices=False); V2 = V2t.T
    total_energy1 = torch.sum(S1 ** 2)
    k1 = max(1, (torch.cumsum(S1 ** 2, dim=0) / total_energy1 <= svd_energy_threshold_model1).sum().item()) if total_energy1 > 0 else 1
    total_energy2 = torch.sum(S2 ** 2)
    k2 = max(1, (torch.cumsum(S2 ** 2, dim=0) / total_energy2 <= svd_energy_threshold_model2).sum().item()) if total_energy2 > 0 else 1
    delta1_updated = U1[:, :k1] @ torch.diag(S1[:k1]) @ V1[:, :k1].T
    delta2_updated = U2[:, :k2] @ torch.diag(S2[:k2]) @ V2[:, :k2].T
    # ---- lines 152-166 -------------------------------------------------------
    U_cat = torch.cat((U1[:, :k1], U2[:, :k2]), dim=1)
    V_cat = torch.cat((V1[:, :k1], V2[:, :k2]), dim=1)
    Pu_double, Du_double, _ = torch.linalg.svd(U_cat.double(), full_matrices=False)
    total_energy_u = torch.sum(Du_double ** 2)
    k_u = torch.searchsorted(torch.cumsum(Du_double ** 2, dim=0) / total_energy_u, global_basis_energy_threshold).item() + 1 if total_energy_u > 0 else 1
    Pu = Pu_double[:, :k_u].to(U_cat.dtype)
    Pv_double, Dv_double, _ = torch.linalg.svd(V_cat.double(), full_matrices=False)
    total_energy_v = torch.sum(Dv_double ** 2)
    k_v = torch.searchsorted(torch.cumsum(Dv_double ** 2, dim=0) / total_energy_v, global_basis_energy_threshold).item() + 1 if total_energy_v > 0 else 1
    Pv = Pv_double[:, :k_v].to(V_cat.dtype)
    # ---- lines 172-173 -------------------------------------------------------
    C1 = torch.matmul(torch.matmul(Pu.T, delta1_updated), Pv)
    C2 = torch.matmul(torch.matmul(Pu.T, delta2_updated), Pv)

    # ---- lines 175-215 (identical block for C1 and C2) ------------------------
    def _prune(C, thr):
        if C.numel() == 0:
            return torch.zeros_like(C)
        energy = C ** 2
        total = torch.sum(energy)
        if total.item() == 0.0 or thr <= 0.0:
            return torch.zeros_like(C)
        if thr >= 1.0:
            return C.clone()
        sorted_energy, sorted_indices = torch.sort(energy.flatten(), descending=True)
        cumulative = torch.cumsum(sorted_energy, dim=0)
        num_keep = torch.searchsorted(cumulative, total * thr, right=False).item() + 1
        retained = min(num_keep, C.numel())
        mask_flat = torch.zeros(C.numel(), dtype=torch.bool, device=C.device)
        if retained > 0:
            mask_flat[sorted_indices[:retained]] = True
        return torch.where(mask_flat.view_as(C), C, torch.zeros_like(C))

    C1_masked = _prune(C1, c_pruning_energy_c1)
    C2_masked = _prune(C2, c_pruning_energy_c2)

    # ---- lines 223-269 (the six masks, verbatim) -------------------------------
    C1_nonzero_mask = C1_masked != 0
    C2_nonzero_mask = C2_masked != 0
    masks = {}
    masks["D_minus"] = C1_nonzero_mask & C2_nonzero_mask & (torch.sign(C1_masked) != torch.sign(C2_masked))
    masks["D_plus"] = C1_nonzero_mask & C2_nonzero_mask & (torch.sign(C1_masked) == torch.sign(C2_masked))
    C1_occupied_rows_mask = C1_nonzero_mask.any(dim=1)
    C1_occupied_cols_mask = C1_nonzero_mask.any(dim=0)
    masks["E"] = (C1_masked == 0) & C1_occupied_rows_mask.unsqueeze(1) & C1_occupied_cols_mask
    masks["B"] = C1_occupied_rows_mask.unsqueeze(1) & ~C1_occupied_cols_mask
    masks["C"] = ~C1_occupied_rows_mask.unsqueeze(1) & C1_occupied_cols_mask
    masks["A"] = ~C1_occupied_rows_mask.unsqueeze(1) & ~C1_occupied_cols_mask
    return dict(k1=k1, k2=k2, Pu=Pu, Pv=Pv, C1=C1_masked, C2=C2_masked,
                masks=masks, C2nz=C2_nonzero_mask)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="runs_exp2d/seed0/population")
    ap.add_argument("--core", default="runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
    ap.add_argument("--pairs", type=int, default=3)
    args = ap.parse_args(argv)

    ids = members(args.population)
    core = load_sd(args.core)
    layers = ["conv1.weight", "layer1.0.conv1.weight", "layer2.1.conv1.weight",
              "layer3.2.conv2.weight", "layer2.0.shortcut.0.weight"]
    print(f"{'pair':18s} {'layer':22s} {'k1':>4s} {'k2':>4s} {'basis':>9s} {'|C1 diff|':>10s} {'|C2 diff|':>10s} {'masks==':>8s} {'nnz(C2)':>7s}")
    worst_c, all_masks_ok, n = 0.0, True, 0
    for i in range(args.pairs):
        a, b = ids[i], ids[(i + 1) % len(ids)]
        A = load_sd(f"{args.population}/{a}/resnet20x4_v0.pth.tar")
        B = load_sd(f"{args.population}/{b}/resnet20x4_v0.pth.tar")
        for k in layers:
            tp = as_matrix(A[k].double() - core[k].double())
            tq = as_matrix(B[k].double() - core[k].double())

            ref = reference_decompose(tp, tq)

            Pu, Pv = make_basis(tp, tq, "code")
            Cp = prune(Pu.T @ _trunc(tp, 0.90) @ Pv, 0.95)
            Cq = prune(Pu.T @ _trunc(tq, 0.99) @ Pv, 0.80)
            ours = classify(Cp, Cq)

            same_shape = (Pu.shape == ref["Pu"].shape) and (Pv.shape == ref["Pv"].shape)
            # The basis itself is unique only up to sign/rotation within equal
            # singular values, so compare the *projected, pruned* matrices in
            # a basis-free way: reconstruct both to weight space.
            rec_ref1 = ref["Pu"] @ ref["C1"] @ ref["Pv"].T
            rec_ref2 = ref["Pu"] @ ref["C2"] @ ref["Pv"].T
            rec_our1 = Pu @ Cp @ Pv.T
            rec_our2 = Pu @ Cq @ Pv.T
            d1 = ((rec_ref1 - rec_our1).norm() / rec_ref1.norm()).item()
            d2 = ((rec_ref2 - rec_our2).norm() / rec_ref2.norm()).item()
            worst_c = max(worst_c, d1, d2)
            # masks: theirs on C2's support vs ours restricted to C2's support.
            # Compare through the energy each type carries (basis-free too).
            ok = True
            for t in TYPES:
                e_ref = ((ref["C2"] * ref["masks"][t]) ** 2).sum().item()
                e_our = ((Cq * (ours[t] & (Cq != 0))) ** 2).sum().item()
                if abs(e_ref - e_our) > 1e-9 * max(e_ref, e_our, 1e-30):
                    ok = False
            all_masks_ok &= ok
            n += 1
            print(f"{a[:8]}x{b[:8]}  {k[:-7]:22s} {ref['k1']:4d} {ref['k2']:4d} "
                  f"{str(tuple(ref['Pu'].shape[1:]) + tuple(ref['Pv'].shape[1:])):>9s} "
                  f"{d1:10.2e} {d2:10.2e} {'yes' if ok else 'NO':>8s} {int(ref['C2nz'].sum()):7d}"
                  + ("" if same_shape else "   SHAPE MISMATCH"))
    print(f"\n{n} layer-pairs: worst reconstruction difference {worst_c:.2e}; "
          f"per-type energies identical: {all_masks_ok}")
    return 0 if (worst_c < 1e-8 and all_masks_ok) else 1


def _trunc(tau, energy):
    U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
    k = max(1, int((torch.cumsum(S ** 2, 0) / (S ** 2).sum() <= energy).sum().item()))
    return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]


if __name__ == "__main__":
    sys.exit(main())
