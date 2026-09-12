# GLOBA merge operator for SESiL — specification

Status: **v2, 2026-09-08.** Implemented in `globa/operator.py`; invariants in `globa/check.py` (T1–T9, 9/9 on the exp2d pair `09111d7d × 1887bcd2`); reference diff in `globa/check_reference.py` (identical to the authors' `svd_merging` to 0.00e+00 on 15 layer-pairs under their thresholds). v1 (2026-09-07) is superseded; what changed and why is in §7.

Sources. **[P]** = Liu, Li & Zhou, *GLOBA: Rethinking Parameter Conflicts in Model Merging*, AAAI-26, main text. **[App]** = its appendix. **[Code]** = `svd_merging` in the authors' repo (`C:\Users\32063\Desktop\GLOBA`). **[Spec]** = a decision made here. Every design element carries one of these tags.

## 1. Contract

```
merge(θ_A, θ_B, θ_core, cfg, classes_A, classes_B) -> θ_child
```

Three state dicts of the same architecture and the two parents' class sets. `θ_core` is the shared core every agent was initialised from (an SSL **backbone**, no classifier). The operator is **symmetric**: `merge(θ_A, θ_B, ·, C_A, C_B) == merge(θ_B, θ_A, ·, C_B, C_A)` bitwise. Pure function — no data, no evaluation, no side effects. Mate choice, screening, evaluation and mutation are outside its scope.

## 2. Which tensors are analysed

| Tensor | Treatment | Tag |
|---|---|---|
| 4-D conv weight `[out,in,k,k]` | reshape to `[out, in·k·k]`, analysed (§3) | proposal §5.2 |
| 2-D weight not in the head | analysed as-is | [P] (linear layers are its native case) |
| BatchNorm weight/bias/running stats | element-wise average | [Code] averages skipped layers 0.5/0.5; SESiL resets BN after the merge anyway |
| any other 1-D float tensor | element-wise average | [Code] |
| integer buffers (`num_batches_tracked`) | copy from the **canonical-first** parent (§3, determinism) | [Spec] — keeps T1 exact; SESiL's BN reset overwrites it |
| **classifier head** `linear.weight [C,F]`, `linear.bias [C]` | **label-aware** (default): row `c` comes from the parent whose class set contains `c` and the other's does not; rows for shared or unknown-to-both classes are averaged. `head="average"` restores element-wise averaging. | [Spec] — the core has no head, so `θ_head − θ_core,head` is undefined and GLOBA's geometry cannot apply. Measured in Stage −1 v2 (§7.6): same backbone merge, head averaged 0.29 vs label-aware 0.91 on A∪B. |

*Coverage note.* [Code] iterates `named_parameters()` only, so buffers are never touched (LLaMA has none). The BN rows are a path the reference does not exercise. [Code] skips layers by *name* and analyses every remaining 2-D tensor including `lm_head`; here the rule is by *shape* and the head is handled by the row above.

## 3. Per analysable layer (the `hybrid_t` construction)

Subscripts A, B are the two parents; the operator never designates a base.

**Step 1 — task vectors** [P, Eq. 1]. `τ_A = θ_A − θ_core`, `τ_B = θ_B − θ_core`, reshaped to `m × n`.

**Step 2 — per-task SVD and the analysed subspace** [P, Eq. 10] + [Spec]. `τ_i = U_i Σ_i V_iᵀ`. Keep the leading singular triples holding a fraction `svd_energy` of `Σσ²` ([Code]'s `≤` count, at least one), **the same fraction for both parents** — default **0.90**. This truncation does *not* discard anything (step 7); it defines the subspace inside which types are read. [Code] truncates asymmetrically (0.90 / 0.99) and drops the remainder. `svd_energy = 1` analyses everything and reproduces v1's basis.

**Step 3 — global basis** [P, Eq. 11–14]. `U_cat = [U_A' U_B']`, `V_cat = [V_A' V_B']` from the truncated singular vectors; SVD each and keep the left singular vectors holding `basis_energy` of the energy ([Code]'s `searchsorted+1`, default **0.999** as in [Code]; `1.0` = numerical rank) → `U_g`, `V_g`.
*Precision:* the whole layer runs in float64 and is cast back at the end ([Code] does only these two SVDs in float64). [Spec]
*Determinism:* the parents are concatenated in a canonical order fixed by a content hash of the whole state dict, not by argument position, so `merge(A,B)` and `merge(B,A)` execute the same floating-point program. Class sets travel with their parent. [Spec] — this is what makes T1 bitwise.

**Step 4 — task interaction matrices** [P, Eq. 15]. `Ĉ_i = U_gᵀ τ_i' V_g` where `τ_i'` is the **truncated** vector — as [Code] does. (Projecting the full `τ_i` instead re-smears the residual into the analysed rows and destroys the classification; tested in the sandbox, §7.5.)

**Step 5 — energy-based pruning** [P, Eq. 17]. For each parent keep the smallest set of positions holding `η` of `Σ|Ĉ|²`, zero the rest → `C̃_A`, `C̃_B`. `η_A = η_B`, default **0.80** ([App] Table 1 modal value; symmetry requires equality). Pruned cells are not lost — they fall into the residual (step 7).

**Step 6 — type assignment** [App, Eq. 25–31] extended to `supp(C̃_A) ∪ supp(C̃_B)`. [Spec]

Let `row_Q(r) := ∃j. C̃_Q[r,j] ≠ 0`, `col_Q(c) := ∃i. C̃_Q[i,c] ≠ 0`.

| Position (r,c) | Type | Source |
|---|---|---|
| both non-zero, same sign | **D⁺** | [App, Eq. 29] |
| both non-zero, opposite sign | **D⁻** | [App, Eq. 30] |
| exactly one parent P non-zero, Q the other; `¬row_Q ∧ ¬col_Q` | **A** | [App, Eq. 25] with (1,2)→(Q,P) |
| one non-zero; `row_Q ∧ ¬col_Q` | **B** | [P] §4.2, [Code] `rows & ~cols` |
| one non-zero; `¬row_Q ∧ col_Q` | **C** | [P] §4.2, [Code] |
| one non-zero; `row_Q ∧ col_Q` | **E** | [App, Eq. 31] with (1,2)→(Q,P) |

*B/C.* [P] §4.2 and [Code] put the zero on the column (input side) for B; [App] Eq. 26–27 and the README put it on the row. This spec follows the main text and the code. B and C share a coefficient in every preset, so the choice does not change any merged result.

*Properties (T5):* pairwise disjoint; union = `supp(C̃_A) ∪ supp(C̃_B)`; invariant under A ↔ B.

**Step 7 — residual** [Spec]. `R_i = τ_i − U_g C̃_i V_gᵀ` — everything the typed cells do not carry: the complement of the analysed subspace and the pruned cells. By construction (T9)

```
τ_i = U_g C̃_i V_gᵀ + R_i        exactly, for each parent
```

**Step 8 — merge** [P, Eq. 20] + residual.

```
C̃_merged[r,c] = α_type(r,c) · ( C̃_A[r,c] + C̃_B[r,c] )
τ_merged       = U_g C̃_merged V_gᵀ  +  ρ · ( R_A + R_B )
θ_child        = θ_core + τ_merged
```

On a single-parent cell the sum is that parent's value, so `α = 1` keeps it whole and `α = 0.5` halves it. `ρ` is the residual coefficient. With `α ≡ ρ ≡ 0.5` the operator is **exactly** element-wise averaging, and with `α ≡ ρ ≡ 1` exactly `τ_A + τ_B`, for **any** `svd_energy`, `basis_energy`, `η` (T2, T3). Type-dependent behaviour exists only where `α ≠ ρ`.

## 4. Coefficient regimes

| Preset | α | ρ | What it is | Source |
|---|---|---|---|---|
| **sum** | all 1 | 1 | `τ_A + τ_B`, task arithmetic | [P] Eq. 18, α=1 |
| **average** | all 0.5 | 0.5 | `½(θ_A + θ_B)` on analysed layers | [P] Eq. 18, α=0.5 |
| **orthogonal-full** | A=1, others 0.5 | 0.5 | the theorem's free lunch only | [Spec] v1 default — inert: type A carries ~1–2% of energy |
| **single-full** | A,B,C,E = 1; D⁺,D⁻ = 0.5 | 0.5 | every cell held by one parent kept whole, colliding cells averaged — "sum where disjoint, average where overlapping" | [Spec], sandbox `sym_single1` |

Anything else is a Strategy-3 point. [App] Table 3 tuned 7B Math+Instr to beneficial types 0.7 and D⁻ **1.2**; [App] Table 6 found uniform 0.6 best for 13B Math+Instr; no universal setting, nothing on iteration. Sandbox evidence (§7.5–7.6): zeroing D⁺ is always harmful (α multiplies the *sum*, so it deletes A's own knowledge too); D⁻ was harmless in every MNIST setting.

*Norm.* `single-full` sits between sum and average: near sum while the parents are dissimilar, near average once they converge (most cells D). Measured on the conv pair: `‖τ_merged‖ / ½(‖τ_A‖+‖τ_B‖)` median 0.99, max 1.12. Log it per merge in any evolution run.

## 5. Invariants (`globa/check.py`)

| # | Test | Expected |
|---|---|---|
| T1 | `merge(A,B,·,C_A,C_B)` vs `merge(B,A,·,C_B,C_A)`, 4 presets × η∈{0.8,1} × head∈{label,average} | bitwise identical |
| T2 | **average**, η ∈ {0.8, 1.0} | analysed layers = `½(θ_A+θ_B)` to float32 rounding; non-analysed exactly |
| T3 | **sum**, η ∈ {0.8, 1.0} | analysed layers = `θ_A + θ_B − θ_core` to float32 rounding |
| T4 | `svd_energy = basis_energy = η = 1`, `τ_B ≡ 0`, **orthogonal-full** | child = `θ_A` bitwise (every cell type A, residual zero) |
| T5 | types on a real pair | disjoint; cover the union; per-type energies sum to total |
| T6 | same inputs twice | bitwise identical |
| T7 | **average**, η < 1 | `‖τ_merged‖ ≤ ½(‖τ_A‖+‖τ_B‖)` (triangle inequality — exact averaging) |
| T8 | label head, real class sets | rows only in A from A, only in B from B, shared and unknown averaged; bitwise |
| T9 | decomposition identity | `τ_i = U_g C̃_i V_gᵀ + R_i`, rel. error < 1e−12 |

T2/T3 are [P] Eq. 22–23 and prove basis/projection/reconstruction; T9 proves nothing is discarded; T1 is why the operator is admissible as a SESiL crossover.

## 6. Out of scope

Mate choice and screening; which pairs merge; evaluation and BN recalibration; mutation. Where the operator lives relative to `sesil/` (answer so far: outside it, `sesil/` untouched) does not change anything above.

## 7. Decisions taken and their evidence

1. **Head** — resolved: label-aware default (§2). Evidence §7.6.
2. **η = 0.80** — kept; with the residual it is no longer a loss knob, only a "what is typed" knob.
3. **`svd_energy`** — v1 removed it as the cause of the earlier rank collapse. v2 exposes it (default 0.90) because it is needed to make the basis well-posed (§7.5) and, with the residual kept, it can no longer discard anything.
4. **Coefficients** — `average` is the control, `single-full` the Strategy-3 candidate; tune only after the norm log of an evolution run.
5. **Basis (found 2026-09-08).** With no truncation, every layer with `out ≤ in·k·k` (20/21 conv layers) has `τ` of full row rank, `[U_A U_B][U_A U_B]ᵀ = 2I`, and `U_g` is an arbitrary rotation: every row is occupied by both parents and the row half of the type test is blind. Measured: type A ≈ 0.1%, D⁺ ≈ 55%, E ≈ 31% on exp2d pairs whose raw task vectors have `cos(τ_A,τ_B) ≈ 0.03`, output directions nearly orthogonal (0.06), input directions nearly identical (0.96) — true geometry type C. Also device-unstable: GPU vs CPU decomposition of identical specialists moved the damaging type from D⁺ to C in the sandbox. Pruning at η<1 in that basis removes energy along arbitrary directions, so v1's `average@0.80` numbers were "pruning in a random basis". `average@1.00` was and is exact.
   - *Rejected fix — truncate-for-order + complete to full basis* (`ordered`): orthonormal and lossless (verified 1e−14) but the completed rows are again occupied by both residuals; A stays 0.00, and pruning concentrates its loss on the residual rows (avg down to −0.146).
   - *Adopted — `hybrid_t`* (§3): classify inside the truncated subspace as [Code] does, carry the rest as a residual. Identities exact for any η. Structure: A 1–3%, B 5–10%, C 8–17%, E 25–50%, D⁺ 20–45%, D⁻ 4–25% on both MNIST and conv pairs. **The C-dominance of [Code] (C 76% on conv) is largely an artefact of its asymmetric truncation** — A truncated to rank ~2 occupies almost no rows/cols, so most of B falls in "A-free" cells. Under symmetric truncation, co-equal specialists from one core overlap heavily; that is the population's geometry, not a defect. Projecting the *full* τ into the subspace (`hybrid`) re-smears occupancy and gives the same shares as `full`; projecting the truncated vector is required.
6. **Sandbox regime and the head (Stage −1 v1 → v2).** v1 fine-tuned specialists from a base supervised on all ten digits (0.977): plain averaging then recovers both tasks trivially (0.96–0.99 on A∪B) and every GLOBA deviation hurts — the paper's regime, not SESiL's. v2 (`globa/sandbox_v2.py`) uses a backbone that knows no classes (rotation pretext, linear probe 0.80 ≈ random init 0.82) and a fresh random head per specialist: averaging then **fails** (0.29 on A∪B at one shared digit, below either parent) and changing only the head to label-aware lifts it to **0.91**; across the sweep 0.29–0.75 → 0.83–0.98. With the head fixed, `single-full` on the backbone is worth +0.015/+0.022 at one shared digit, +0.009 at two (3200 steps), −0.014/−0.007 at zero overlap. Type shares in v2 match the real conv pairs. Findings on type damage (directional protocol, both versions): D⁺ is the damaging type, D⁻ harmless everywhere, C harmful under heavy specialisation, full sum worst — the proposal's §9 expectation "D⁻ clearly most damaging" does not hold.
7. **Iteration and the core — measured (`globa/sandbox_iter.py`, 2026-09-08).** With `θ_core` fixed at generation 0, the sandbox population's decomposition degenerates by generation 3: D⁺ 0.96–0.99, single-parent cells ≤ 0.04 (gen 1: 0.47 / 0.52). The collective drift from the stale core dominates every pairwise difference, so from generation 3 every type-dependent α is inert and the operator is averaging. A core refreshed to the current population mean keeps the classification informative throughout (single cells 0.4–0.9) but redefines "single-parent" as "deviation from the population mean one parent has", and `single-full` then gains nothing over average. `average` is core-independent by the T2 identity. Decision: keep the fixed gen-0 core (its semantics are the ones the operator was designed for) and read `globa_stats.jsonl` in any evolution run to see in which generation D⁺ saturates — on MNIST it is generation 3; on CIFAR, where the population converges far more slowly, it may be much later. `single-full` also drifts from the core ~50% faster than `average` (15.7 vs 10.4 after 6 sandbox generations, roughly linear).
8. **Occupancy is presence-based** (one surviving cell occupies a row) — inherited from [App]; a magnitude-aware rule might change the A/C shares. Not built.
