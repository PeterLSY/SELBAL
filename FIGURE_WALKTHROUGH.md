Pretraining (the "weak" part):
  - 10 agents, each assigned a distinct 3-class subset of CIFAR-10 (via mapping.json).
  - Each agent trains a resnet20x4 from scratch (random init, no shared backbone) on only its 3 classes, using labels.
  - Early-stop at target-acc 0.72 (cap 60 epochs). Took ~505s, all 10/10 hit the target.
  - Because it's early-stop-on-accuracy, epochs-per-expert vary a lot (2–18 epochs), summing to 64 subset-epochs = 19.2 full-set-equivalent units.
# This Week's Results — Figure by Figure

All numbers are **single-seed** and verified against CSV (`tools/audit_numbers.py`).

---

## Figure 1: `paper_main.png` — Main figure (SSL budget ledger + baseline)

- **Axes**: x = training budget (full-set-equivalent epochs), y = CIFAR-10 accuracy.
  Five colored evolution curves + a faint grey dashed line at top (joint upper-bound ref).
- **Experiment**: unlabeled common pretraining (SimSiam) + each agent finetuned on its
  assigned 3 classes, sweeping the split between SSL amount (E) and finetune amount (S)
  at the same 9.0-unit starting budget.
- **Result**: warmer color (more SSL) climbs higher — E3→0.843, E6→0.854, E7.5→0.858,
  E9→0.872. All five sit above the supervised reduced-budget method (0.804). The grey
  top line is joint full-budget (0.957); no evolution run reaches it.
- **One line**: more common pretraining → higher end-point; task-side finetuning is
  nearly free (E9 uses only 3 batches of finetuning and still reaches 0.872).

## Figure 2: `paper_honesty.png` — Honesty comparison

- **On it**: same curves, y-axis full 0–1, plus a red dashed line (joint 520-epoch actual
  training curve, 0.25 → 0.957) and a grey diamond (joint 19-epoch = 0.889).
- **Result**: at matched compute, joint training beats evolutionary merging — joint hits
  0.889 in 20 units, evolution needs ~490 units to reach 0.85.
- **Why it exists**: shows the comparison unfavorable to us. The merging paradigm's value
  is data-cannot-be-centralized / incremental / distributed, not compute efficiency. This
  figure prevents over-claiming.

## Figure 3: `startpoint_vs_final.png` — Start-point does not predict end-point

- **On it**: scatter, x = expert start-point accuracy, y = gen25 end-point. Red = SSL,
  blue = supervised, hollow square = strong population (estimated).
- **Result** (most counter-intuitive): exp2d starts lowest (0.557) but ends second-highest
  (0.872); the strong population starts highest (0.93) yet ends at only 0.840. Start ranking
  != end ranking.
- **One line**: the end-point is set by the mergeability of the shared representation, not by
  start-point classification accuracy.

## Figure 4: `collapse_recovery_zoom.png` — Collapse & recovery (gen 1-8 zoom)

- **On it**: seven curves, first 8 generations. All drop to a trough at gen2 (0.11–0.24),
  then diverge from gen3.
- **Result**: the first merge collapses every initialization; SSL runs recover in one
  generation (span=1), weak supervised starts take 3–4; the strong population barely
  collapses (retains 98%).
- **One line**: merge damage hits every initialization, but a shared SSL representation
  recovers fastest.

## Figures 5-7: earlier comparisons

`budget_axis_compare` / `generation_axis_compare` / `exp2b_*` are intermediate versions:
the reduced-budget story (exp1 0.805, exp1b 0.804 catching up to strong 0.840) and the
exp2b trough zoom. Figures 1-4 subsume them; keep as process records.

---

## Three result-threads across all figures

1. **Budget robustness** (Fig 5-7): halve the budget, end-point barely changes.
2. **SSL common-pretraining ledger** (Fig 1): SSL amount monotonically sets the end-point,
   finetuning is dispensable.
3. **Start-point doesn't predict end-point + recovery speed** (Fig 3, 4): the mechanism.

## ⚠ Say this with every figure

**These are all single-seed.** Every curve/point is one run. Multi-seed reproduction is in
progress (seed2 at gen28/40). Until it finishes: gaps between curves are trends, not
statistical significance; there are no error bars on the figures. Once seed2 completes,
Figure 1 can get error bars and "SSL > supervised" upgrades from "looks higher" to
"statistically holds."
