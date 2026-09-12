"""GLOBA merge operator: two parents + shared core -> one child.

Executable form of docs/globa_operator_spec.md (v2, 2026-09-08). Section
numbers refer to that document; [P]/[App]/[Code]/[Spec] tags as there.

    merge(sd_a, sd_b, sd_core, cfg, classes_a, classes_b) -> sd_child

What changed from v1 after Stage -1 (globa/sandbox.py, globa/sandbox_v2.py):

* Basis.  v1 used an untruncated basis, which is lossless but DEGENERATE for
  every layer with out <= in (the output-side basis is an arbitrary
  rotation). v2 uses the `hybrid_t` construction: a symmetric energy
  truncation (`svd_energy`) defines the ANALYSED subspace; types and
  coefficients act there; everything the typed cells do not carry -- the
  subspace complement and the pruned cells -- is a RESIDUAL carried at
  coefficient `rho`. Nothing is discarded, so no ratchet forms, and the
  identities hold for any eta: alpha == rho == 0.5 is exactly plain
  averaging, alpha == rho == 1 is exactly tau_A + tau_B.

* Head.  v1 averaged the classifier. In SESiL's regime (a core that knows no
  classes, a random head per agent) that is what destroys disjoint classes:
  in the sandbox the same backbone merge scores 0.29 on A|B with an averaged
  head and 0.91 with a LABEL-AWARE head. A class row now comes from the
  parent that knows the class; shared or unknown classes are averaged. This
  needs the parents' class sets, hence the two extra arguments. [Spec] -- the
  paper has no headless core and no counterpart.

All linear algebra in float64; outputs cast back to the parents' dtype.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch

TYPES: Tuple[str, ...] = ("A", "B", "C", "E", "D_plus", "D_minus")

# Spec section 4. Each preset fixes the six type coefficients and the residual
# coefficient rho. "single-full" is the sandbox's sym_single1: every cell held
# by one parent kept whole, colliding cells averaged.
PRESETS: Dict[str, Dict] = {
    "sum":             dict(alpha={t: 1.0 for t in TYPES}, rho=1.0),   # [P] Eq. 18, alpha=1
    "average":         dict(alpha={t: 0.5 for t in TYPES}, rho=0.5),   # [P] Eq. 18, alpha=0.5
    "orthogonal-full": dict(alpha={"A": 1.0, "B": .5, "C": .5, "E": .5, "D_plus": .5, "D_minus": .5}, rho=0.5),
    "single-full":     dict(alpha={"A": 1.0, "B": 1.0, "C": 1.0, "E": 1.0, "D_plus": .5, "D_minus": .5}, rho=0.5),
}


@dataclass(frozen=True)
class MergeConfig:
    """Everything the operator can be told.

    eta          pruning threshold inside the analysed subspace, [P] Eq. 17,
                 same for both parents (symmetry). 0.80 = modal value in [App]
                 Table 1. Pruned cells go to the residual, so eta no longer
                 loses anything.
    svd_energy   symmetric energy truncation that DEFINES the analysed
                 subspace ([Code] uses 0.90/0.99 and drops the rest; here 0.90
                 for both and nothing is dropped). 1.0 = analyse everything =
                 v1's degenerate basis.
    basis_energy energy kept when re-orthogonalising the concatenated
                 singular vectors, [Code] 0.999. 1.0 = numerical rank.
    alpha        per-type coefficient for [P] Eq. 20.
    rho          coefficient on the residual (complement + pruned cells).
    head         'label' (class row from the parent that knows the class,
                 shared/unknown averaged; needs classes) or 'average'.
    head_prefix  how the classifier is recognised in the state dict.
    """
    eta: float = 0.80
    svd_energy: float = 0.90
    basis_energy: float = 0.999
    alpha: Mapping[str, float] = field(default_factory=lambda: dict(PRESETS["average"]["alpha"]))
    rho: float = 0.5
    head: str = "label"
    head_prefix: str = "linear."

    @classmethod
    def preset(cls, name: str, **kw) -> "MergeConfig":
        p = PRESETS[name]
        return cls(alpha=dict(p["alpha"]), rho=p["rho"], **kw)

    def __post_init__(self):
        missing = [t for t in TYPES if t not in self.alpha]
        if missing:
            raise ValueError(f"alpha lacks coefficients for {missing}")
        if self.head not in ("label", "average"):
            raise ValueError(f"head must be 'label' or 'average', got {self.head!r}")


# --------------------------------------------------------------------------
# which tensors are analysed (spec section 2)
# --------------------------------------------------------------------------
def is_head(name: str, cfg: MergeConfig) -> bool:
    return name.startswith(cfg.head_prefix)


def is_analysable(name: str, t: torch.Tensor, sd_core: Mapping[str, torch.Tensor],
                  cfg: MergeConfig) -> bool:
    if name not in sd_core or is_head(name, cfg):
        return False
    return t.ndim in (2, 4)


def as_matrix(t: torch.Tensor) -> torch.Tensor:
    """[out, in, k, k] -> [out, in*k*k]; 2-D unchanged. Proposal section 5.2."""
    return t.reshape(t.shape[0], -1)


# --------------------------------------------------------------------------
# the pieces (spec section 3)
# --------------------------------------------------------------------------
def _truncate(U, S, Vh, energy: float):
    """[Code] convention: keep the singular triples whose cumulative squared
    energy stays <= energy (at least one)."""
    if energy >= 1.0:
        return U, S, Vh
    k = max(1, int((torch.cumsum(S ** 2, 0) / (S ** 2).sum() <= energy).sum().item()))
    return U[:, :k], S[:k], Vh[:k]


def _lead_basis(M: torch.Tensor, basis_energy: float) -> torch.Tensor:
    """Left singular vectors of the concatenated singular vectors.
    basis_energy < 1: [Code]'s searchsorted+1 energy cut; else numerical rank."""
    P, D, _ = torch.linalg.svd(M, full_matrices=False)
    if D.numel() == 0 or D[0] <= 0:
        return P[:, :1]
    if basis_energy >= 1.0:
        tol = D[0] * max(M.shape) * torch.finfo(M.dtype).eps
        k = int((D > tol).sum().item())
    else:
        k = int(torch.searchsorted(torch.cumsum(D ** 2, 0) / (D ** 2).sum(), basis_energy).item()) + 1
    return P[:, :max(1, min(k, P.shape[1]))]


def _orthonormal_basis(M: torch.Tensor) -> torch.Tensor:
    """Numerical-rank basis (v1's construction; kept for globa/sandbox.py)."""
    return _lead_basis(M, 1.0)


def prune(C: torch.Tensor, eta: float) -> torch.Tensor:
    """[P] Eq. 17, [Code]'s searchsorted+1 convention."""
    if eta >= 1.0:
        return C.clone()
    energy = (C * C).flatten()
    total = energy.sum()
    if total <= 0 or eta <= 0.0:
        return torch.zeros_like(C)
    vals, order = torch.sort(energy, descending=True, stable=True)
    cum = torch.cumsum(vals, 0)
    k = min(int(torch.searchsorted(cum, total * eta, right=False).item()) + 1, energy.numel())
    mask = torch.zeros_like(energy, dtype=torch.bool)
    mask[order[:k]] = True
    return torch.where(mask.view_as(C), C, torch.zeros_like(C))


def classify(Cp: torch.Tensor, Cq: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Spec step 6: six boolean masks partitioning supp(Cp) | supp(Cq).
    D+/D- are [App] Eq. 29-30. A cell held by one parent is typed by the
    OTHER parent's occupancy: A neither row nor col, B row only, C col only,
    E both ([App] Eq. 25/31; B/C per [P] sec 4.2 and [Code]). Symmetric."""
    nz_p, nz_q = Cp != 0, Cq != 0
    both = nz_p & nz_q
    same = torch.sign(Cp) == torch.sign(Cq)
    only_p, only_q = nz_p & ~nz_q, nz_q & ~nz_p
    row_p, col_p = nz_p.any(1, keepdim=True), nz_p.any(0, keepdim=True)
    row_q, col_q = nz_q.any(1, keepdim=True), nz_q.any(0, keepdim=True)

    def single(only, row_o, col_o):
        return {"A": only & ~row_o & ~col_o, "B": only & row_o & ~col_o,
                "C": only & ~row_o & col_o, "E": only & row_o & col_o}

    sp, sq = single(only_p, row_q, col_q), single(only_q, row_p, col_p)
    masks = {t: sp[t] | sq[t] for t in ("A", "B", "C", "E")}
    masks["D_plus"], masks["D_minus"] = both & same, both & ~same
    return masks


def decompose(tau_p: torch.Tensor, tau_q: torch.Tensor, cfg: MergeConfig):
    """Steps 2-6 for one layer (hybrid_t).

    Returns Pu, Pv, Cp, Cq, masks, Rp, Rq with the exact identity
        tau = Pu @ C @ Pv.T + R   for each parent.
    """
    Up, Sp, Vhp = torch.linalg.svd(tau_p, full_matrices=False)
    Uq, Sq, Vhq = torch.linalg.svd(tau_q, full_matrices=False)
    Up_, Sp_, Vhp_ = _truncate(Up, Sp, Vhp, cfg.svd_energy)
    Uq_, Sq_, Vhq_ = _truncate(Uq, Sq, Vhq, cfg.svd_energy)
    Pu = _lead_basis(torch.cat((Up_, Uq_), 1), cfg.basis_energy)
    Pv = _lead_basis(torch.cat((Vhp_.T, Vhq_.T), 1), cfg.basis_energy)
    # classify the truncated vectors (structure), as [Code] does ...
    tp_in = Up_ @ torch.diag(Sp_) @ Vhp_ if cfg.svd_energy < 1.0 else tau_p
    tq_in = Uq_ @ torch.diag(Sq_) @ Vhq_ if cfg.svd_energy < 1.0 else tau_q
    Cp = prune(Pu.T @ tp_in @ Pv, cfg.eta)
    Cq = prune(Pu.T @ tq_in @ Pv, cfg.eta)
    # ... but keep what the typed cells do not carry (losslessness)
    Rp = tau_p - Pu @ Cp @ Pv.T
    Rq = tau_q - Pu @ Cq @ Pv.T
    return Pu, Pv, Cp, Cq, classify(Cp, Cq), Rp, Rq


def merge_matrix(tau_p: torch.Tensor, tau_q: torch.Tensor, cfg: MergeConfig):
    """Steps 2-8 for one layer. Returns (tau_merged, stats)."""
    Pu, Pv, Cp, Cq, masks, Rp, Rq = decompose(tau_p, tau_q, cfg)
    S = Cp + Cq                                            # [P] Eq. 20 / 23
    Cm = torch.zeros_like(S)
    for t in TYPES:                                        # fixed order
        a = cfg.alpha[t]
        if a != 0.0:
            Cm = Cm + a * (S * masks[t])
    tau_m = Pu @ Cm @ Pv.T + cfg.rho * (Rp + Rq)          # [P] Eq. 21 + residual

    tot = float((S * S).sum())
    np_, nq_ = float(tau_p.norm()), float(tau_q.norm())
    stats = {
        "basis_u": Pu.shape[1], "basis_v": Pv.shape[1],
        "nnz_p": int((Cp != 0).sum()), "nnz_q": int((Cq != 0).sum()),
        "energy_frac": {t: (float(((S * masks[t]) ** 2).sum()) / tot if tot > 0 else 0.0) for t in TYPES},
        "typed_frac_p": 1.0 - float((Rp * Rp).sum()) / (np_ ** 2) if np_ > 0 else 0.0,
        "typed_frac_q": 1.0 - float((Rq * Rq).sum()) / (nq_ ** 2) if nq_ > 0 else 0.0,
        "norm_p": np_, "norm_q": nq_, "norm_merged": float(tau_m.norm()),
    }
    denom = 0.5 * (np_ + nq_)
    stats["norm_ratio"] = stats["norm_merged"] / denom if denom > 0 else 0.0
    return tau_m, stats


# --------------------------------------------------------------------------
# the head (spec section 2, v2)
# --------------------------------------------------------------------------
def merge_head(name: str, tp: torch.Tensor, tq: torch.Tensor,
               classes_p: Optional[Iterable[int]], classes_q: Optional[Iterable[int]],
               cfg: MergeConfig) -> torch.Tensor:
    """Row = class for linear.weight [C, F] and linear.bias [C]."""
    if cfg.head == "average" or not tp.is_floating_point():
        return 0.5 * (tp + tq) if tp.is_floating_point() else tp.clone()
    if classes_p is None or classes_q is None:
        raise ValueError("head='label' needs classes_a and classes_b")
    cp, cq = set(int(c) for c in classes_p), set(int(c) for c in classes_q)
    out = 0.5 * (tp + tq)                                  # shared or unknown-to-both
    for c in range(tp.shape[0]):
        if c in cp and c not in cq:
            out[c] = tp[c]
        elif c in cq and c not in cp:
            out[c] = tq[c]
    return out


# --------------------------------------------------------------------------
# the operator (spec section 1)
# --------------------------------------------------------------------------
def _fingerprint(sd: Mapping[str, torch.Tensor]) -> bytes:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.digest()


def merge(sd_a: Mapping[str, torch.Tensor],
          sd_b: Mapping[str, torch.Tensor],
          sd_core: Mapping[str, torch.Tensor],
          cfg: Optional[MergeConfig] = None,
          classes_a: Optional[Iterable[int]] = None,
          classes_b: Optional[Iterable[int]] = None,
          return_stats: bool = False):
    """Two parents, their shared core, and their class sets -> one child.

    Non-analysable non-head tensors: floating -> element-wise average;
    integer buffers -> copied from the canonical-first parent.
    """
    cfg = cfg or MergeConfig()
    if set(sd_a) != set(sd_b):
        raise ValueError("parents have different key sets")

    # Canonical order: symmetry by construction. Classes travel with their parent.
    if _fingerprint(sd_b) < _fingerprint(sd_a):
        sd_a, sd_b = sd_b, sd_a
        classes_a, classes_b = classes_b, classes_a
    p, q = sd_a, sd_b

    child: Dict[str, torch.Tensor] = {}
    stats: Dict[str, dict] = {}
    for name in p:
        tp, tq = p[name], q[name]
        if is_head(name, cfg):
            child[name] = merge_head(name, tp, tq, classes_a, classes_b, cfg)
            continue
        if not is_analysable(name, tp, sd_core, cfg):
            child[name] = 0.5 * (tp + tq) if tp.is_floating_point() else tp.clone()
            continue
        core = sd_core[name]
        dtype, shape = tp.dtype, tp.shape
        tau_p = as_matrix(tp.double() - core.double())
        tau_q = as_matrix(tq.double() - core.double())
        tau_m, st = merge_matrix(tau_p, tau_q, cfg)
        child[name] = (core.double() + tau_m.reshape(shape)).to(dtype)
        stats[name] = st

    return (child, stats) if return_stats else child
