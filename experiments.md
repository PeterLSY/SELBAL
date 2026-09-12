# SESiL experiment ledger

Every launched experiment gets one row in the table below and one detailed entry
further down. Times are local (UTC-0400).

| experiment | launched | script | method | gens | population | seed | status | output |
|---|---|---|---|---|---|---|---|---|
| `exp1_weak_pretrain` | 2026-07-22 12:58 | `run_sesil.py` | permute | 25 | weak pretraining (target-acc 0.72, measured mean 0.7529) | 0 | **done 25/25** (gen25=0.8048) | `runs_weak/seed0/` |
| `hist_permute_25gen` | 2026-07-20 | `training_scripts/evolutionary_permute_training.py` | permute | 25 | `initial/` strong population (~0.93) | 0 (hard-coded) | done (25/25) | `csvs/2026-07-2{0,1}/permute/`, `checkpoints/cifar10_evolution/permute/` |
| `hist_wavg_11gen` | 2026-07-21 | `training_scripts/evolutionary_wavg_training.py` | wavg | 11 (stopped manually) | `initial/` strong population (~0.93) | 0 (hard-coded) | stopped manually (plateau confirmed) | `csvs/2026-07-2{1,2}/wavg/`, `checkpoints/cifar10_evolution/wavg/` |
| `baseline_joint_e19` | 2026-07-23 02:09 | `train_joint_baseline.py` | -- (joint training) | 19 epochs | all 10 classes, from scratch | 0 | **done** (best 0.8886) | `runs_baseline/joint_e19_seed0/` |
| `baseline_joint_e520` | 2026-07-23 02:19 | `train_joint_baseline.py` | -- (joint training) | 520 epochs | all 10 classes, from scratch | 0 | **done** (best 0.9572 @ep422) | `runs_baseline/joint_e520_seed0/` |
| `exp1b_half_budget` | 2026-07-23 06:57 | `run_sesil.py` | permute | 25 | fixed 3 epochs per expert, no early stop | 0 | **done 25/25** (gen25=0.8035) | `runs_half/seed0/` |
| `ssl_simsiam_e50` | 2026-07-23 | `train_ssl.py` | -- (SimSiam SSL) | 50 epochs | all 10 classes, unlabelled | 0 | **done** (loss -0.86, 4.5 h) | `runs_ssl/simsiam_e50_seed0/` |
| `exp2b_ssl_E6S1` | 2026-07-24 04:15 | `run_sesil.py --init-backbone-only` | permute | 25 | SSL E=6 + finetune S=1 | 0 | **done 25/25** (gen25=0.8538) | `runs_exp2b/seed0/` |
| `exp2a_ssl_E3S2` | 2026-07-25 03:16 | `run_sesil.py --init-backbone-only` | permute | 25 | SSL E=3 + finetune S=2 | 0 | **done 25/25** (gen25=0.8426) | `runs_exp2a/seed0/` |
| `exp2c_ssl_E7.5S0.5` | 2026-07-25 15:53 | `run_sesil.py --init-backbone-only` | permute | 25 | SSL E=7.5 + finetune S=0.5 | 0 | **done 25/25** (gen25=0.8581) | `runs_exp2c/seed0/` |
| `exp2d_ssl_E9S0.1` | 2026-07-26 22:10 | `run_sesil.py --init-backbone-only` | permute | 25 | SSL E=9 + finetune S=0.1 (budget end point, start 0.5571) | 0 | **done 25/25** (gen25=0.8722) | `runs_exp2d/seed0/` |
| `exp2e_ssl_E9_zerocal` | -- | `run_sesil.py --init-backbone-only` | permute | 25 | SSL E=9 + zero head finetuning / closed-form calibration | 0 | **planned** (after multi-seed, not essential) | `runs_exp2e/seed0/` |
| `exp2ref_ssl_E50S1` | 2026-07-26 | `run_sesil.py --init-backbone-only` | permute | 25 | SSL E=50 + finetune S=1 (over-budget reference) | 0 | **done 25/25** (gen25=0.8768) | `runs_exp2ref/seed0/` |

---

## `exp1_weak_pretrain`

**Purpose**: rerun permute evolution on a weakly pretrained population and compare
with `hist_permute_25gen` (strong population, also permute, 25 generations) to see
how the strength of the initial population affects the evolution trajectory. The
class split is identical to the strong population's, so the two are directly
comparable.

- **Launched**: 2026-07-22 12:58:36
- **Command**:
  ```bash
  python run_sesil.py --seed 0 --method permute --generations 25 \
         --target-acc 0.72 --out-root ./runs_weak
  ```
- **Seed**: 0 (controls training randomness only; the class split is fixed and
  reuses the 10 subsets of `initial/`)
- **Merging function**: `match_tensors_permute`
- **Key hyper-parameters** (all defaults, identical to the old scripts):
  - `--mutation-epochs 2`, `--mutation-lr 0.001`
  - `--stop-node 21`, `--merge-a 0.0001`, `--merge-b 0.075`
  - `--mate-threshold 0.5`
- **Pretraining**: cap 60 epochs, early stop at `--target-acc 0.72`; 505.2 s, 10/10 reached the target

  | model | hash | epochs | 3-class accuracy |
  |---|---|---|---|
  | `8_7_0` | 09111d7d | 5 | 0.7213 |
  | `2_9_7` | 1887bcd2 | 7 | 0.7683 |
  | `3_9_5` | 366f9057 | 6 | 0.7473 |
  | `9_8_6` | 5ffed8da | 4 | 0.7450 |
  | `6_7_3` | 88ed44d1 | 4 | 0.7670 |
  | `6_3_5` | 905afff7 | 18 | 0.7633 |
  | `3_7_6` | be2e50e9 | 6 | 0.7333 |
  | `1_7_6` | cb3a86af | 2 | 0.8237 |
  | `7_4_5` | d8bd0bc0 | 6 | 0.7337 |
  | `2_1_4` | e1762771 | 6 | 0.7257 |
  | **mean** | | | **0.7529** |

- **Output**:
  - population `runs_weak/seed0/population/<hash>/resnet20x4_v0.pth.tar`
  - generations `runs_weak/seed0/permute/gen_N/<hash>/resnet20x4_v0.pth.tar`
  - CSV `runs_weak/seed0/permute/csv/gen_N/configurations.csv`
  - per-generation statistics `runs_weak/seed0/permute/run.json`
  - log `runs_weak/seed0/permute.log`
- **Status**: running (**restarted once at 13:52**, see below)

### Incident record: 2026-07-22 13:40, crash in generation 2 and restart

Generation 1 completed normally (10 offspring saved); generation 2 crashed while
loading the population:

```
RuntimeError: Error(s) in loading state_dict for ResNet:
  Missing key(s):    "conv1.weight", ...
  Unexpected key(s): "head_models.0.conv1.weight", ..., "merged_model.conv1.weight", ...
```

**Root cause**: `sesil/pipeline.py` saved offspring with `utils.save_model`
(`utils.py:1011`), which does a plain `torch.save(model.state_dict())` and does not
unwrap `ModelMerge`. A merged offspring is a `ModelMerge` whose state_dict carries
`head_models.0/1.*` and `merged_model.*` prefixes (384 keys = 128 x 3), which resnet20
cannot load.

The old scripts each define their own `save_model` with the same name
(`evolutionary_wavg_training.py:835-841`, `evolutionary_permute_training.py:859-866`)
that takes `head_models[head_index].state_dict()` when `head_models` exists. These
definitions **shadow** the version pulled in by `from utils import *`; the refactor
looked only at the `utils` version and missed the in-script override.

**Why the smoke test did not catch it**: the smoke test runs one generation. A bad
checkpoint does not fail in the generation that writes it, only when the **next**
generation loads it. A single-generation test is blind to this class of bug.

**Fix**: `sesil/pipeline.py` gained `save_offspring()`, which replicates the old
scripts' unwrapping; `utils.py` is unchanged. A regression test
`tools/check_offspring_roundtrip.py` (build a real merged offspring -> save -> load
back with resnet20) was added and passes 3/3.

**Impact and handling**: all 10 generation-1 checkpoints were unusable and were
quarantined as a whole in `runs_weak/seed0/permute_BROKEN_gen1/` (not deleted). The
pretrained population is unaffected (plain resnet20, 10/10 verified loadable), so
the run was restarted from the population with `--skip-pretrain`, counting 25
generations afresh:

```bash
python run_sesil.py --seed 0 --method permute --generations 25 \
       --target-acc 0.72 --skip-pretrain --out-root ./runs_weak
```

**Fix verification**:
- Offline: `tools/check_offspring_roundtrip.py` 3/3 (`save_offspring` writes 128 loadable
  keys; `utils.save_model` writes 384 keys and is rejected, reproducing the crash;
  the loner path is fine)
- Live: at 14:21 generation 2 loaded `gen_1/` and evaluated several members in a
  row without a traceback. This is the necessary condition for declaring the fix
  effective; getting through generation 1 alone does not count.

**Reproducibility check**: the restarted generation 1 is bitwise identical to the
pre-crash one (quarantined in `permute_BROKEN_gen1/`):

```
restart  gen1: n=10  Joint mean 0.2238  min 0.2154  max 0.2475
BROKEN   gen1: n=10  Joint mean 0.2238  min 0.2154  max 0.2475
```

This shows seeding works, the restart introduced no drift, and the crash **affected
only checkpoint serialisation, not the computation**. The evaluation numbers from
the pre-crash attempt are numerically trustworthy.

**Incident status: closed** (2026-07-22).

**Known deviation**: `--target-acc` is checked at epoch boundaries, so the actual
population strength spreads over 0.7213-0.8237 (a range of 0.10) rather than a
uniform 0.72; the deviation is anti-correlated with class difficulty. If a
conclusion depends on uniform initial strength, use fixed `--epochs` or a
finer-grained early-stop check.

### Early observation (through gen 2; warning: 2 generations only, not a conclusion)

Matched-generation comparison with `hist_permute_25gen` (strong population):

| | population accuracy | gen 1 Joint | gen 2 Joint |
|---|---|---|---|
| strong (`hist_permute`) | 0.93 | 0.2766 | 0.2724 |
| weak (`exp1`) | 0.7529 | 0.2238 | 0.1253 |
| **weak/strong ratio** | **0.81** | **0.81** | **0.46** |

At gen 1 the ratio 0.81 matches the population-accuracy ratio 0.81 exactly: the
population still consists of the original 3-class experts, and Joint ~ accuracy x 3/10
holds linearly. At gen 2 the population is the product of merge + mutation and the
ratio collapses to 0.46, no longer linear.

Full gen-2 numbers:

```
weak   : n=10  Joint mean 0.1253  min 0.0998  max 0.1822   PerTaskAvg 0.2272
strong : n=10  Joint mean 0.2724  min 0.2612  max 0.2899   PerTaskAvg 0.4899
delta  : -0.1471
```

**Variance**: the weak population's gen-2 min-max span is **0.082**, the strong one's
only **0.029** (about 3x), i.e. merge outcomes are clearly less stable from a weak
start. One example: offspring `4_5_7_8_0` nominally covers 5 classes but only class 0
survived (0.998); the other four went to zero, leaving a near single-class classifier.

> Two generations are too few to judge a trend. If the ratio keeps diverging, the
> exp1 curve is not a shifted copy of the strong curve but a different shape -- which
> needs the full 25 generations to decide. Snapshots were taken at gen 5 and gen 11.

### Final result (25 generations complete, 2026-07-23)

Run time 44537 s ~ 12.4 h, no crash. **The gen-2 trough is not a trend but a one-off
drop when merging first happens; from then on the weak population keeps catching up
and by gen 25 almost fully matches the strong one.**

| gen | 1 | 2 | 5 | 11 | 18 | 25 |
|---|---|---|---|---|---|---|
| strong Joint | 0.2766 | 0.2724 | 0.6095 | 0.7624 | 0.8171 | 0.8404 |
| weak Joint | 0.2238 | 0.1253 | 0.2696 | 0.5734 | 0.7373 | 0.8048 |
| ratio | 0.81 | **0.46** | 0.44 | 0.75 | 0.90 | **0.96** |

Ratio trajectory 0.81 -> 0.46 (trough) -> 0.96 (end). Catch-up mechanism: the strong
population approaches its plateau early (0.76 by gen 11) and slows down, while the
weak one keeps climbing steeply in the unsaturated region (gen 5 -> 11 slope twice the
strong population's), and the gap is systematically eaten. **Final gap only 0.036
(0.8404 vs 0.8048): weakening the initial population from 0.93 to 0.75 costs almost
nothing after 25 generations.** Positive evidence for SESiL's robustness to initial
population quality.

Variance: the weak population's spread narrows from 0.116 at gen 5 to 0.020 at gen 25,
level with the strong population's 0.020 at the end -- final population consistency is
not affected by the weaker start.

**To do**:
- After the full 25 generations, check the hard combinations (such as `6_3_5`, which
  took 18 pretraining epochs) across generations on the complete data. Watching
  individuals per generation costs more than it is worth, so it is not done during the run.
- After this run, split out `--mutation-num-workers` (default 0). The mutation loader
  currently reuses `--num-workers` (a pretraining parameter); the semantics are muddled.
  Speed only, results unaffected.

---

## `hist_permute_25gen`

- **Launched**: 2026-07-20 (gens 1-22 on 07-20, gens 23-25 on 07-21, across midnight)
- **Script**: `training_scripts/evolutionary_permute_training.py` (parameters hard-coded in `__main__`)
- **Population**: `checkpoints/cifar10_evolution/initial/`, 10 three-class experts, accuracy ~0.93
- **Configuration**: `num_generation=25`, `merging_fn='match_tensors_permute'`, `mutation epochs=2`, `stop_node=21`, `a=0.0001`, `b=0.075`
- **Output**: `csvs/2026-07-20/permute/1..22/`, `csvs/2026-07-21/permute/23..25/`; checkpoints `checkpoints/cifar10_evolution/permute/gen_1..25/`
- **Status**: done (25/25)

`csvs/2025-09-24`, `2026-07-08`, `2026-07-16`, `2026-07-19` each hold an isolated
`permute/1/`: earlier single-generation trial runs, not part of this continuous 25-generation run.

---

## `hist_wavg_11gen`

- **Launched**: 2026-07-21 (gens 1-7 on 07-21, gens 8-11 on 07-22, across midnight)
- **Script**: `training_scripts/evolutionary_wavg_training.py`
- **Population**: as above, the `initial/` strong population
- **Configuration**: `num_generation=15`, `merging_fn='match_tensors_identity'`, otherwise as permute
- **Output**: `csvs/2026-07-21/wavg/1..7/`, `csvs/2026-07-22/wavg/8..11/`; checkpoints `checkpoints/cifar10_evolution/wavg/gen_1..11/`
- **Status**: **stopped manually** (Ctrl+C after generation 11). The script's
  `num_generation` was 15, but the curve had clearly reached its plateau by generation
  11 and time before the meeting was short, so it was terminated deliberately -- not a
  crash or abnormal interruption.

---

## `baseline_joint_e19` / `baseline_joint_e520` (planned)

Joint-training baseline: a single resnet20x4 trained on all 10 CIFAR-10 classes,
checkpointed every epoch with the test accuracy recorded.

```bash
python train_joint_baseline.py --epochs 19                      # (a) pretraining budget matched
python train_joint_baseline.py --epochs 520 --save-every 10     # (c) total budget matched
```

**Each budget is its own run with a complete cosine schedule** (`T_max = epochs`), so
each is genuinely optimal for its budget.

> Warning: do **not** take epoch 19 of the 520-epoch run as the 19-epoch baseline --
> at that point the cosine LR is still near its peak and the accuracy is markedly low,
> which would be unfair to the baseline. Mid-run checkpoints of the long run are used
> **only** as exp2 starting points (use b).

**Budget conversion**:
- Pretraining: exp1's 10 experts total 64 epochs x 15000 samples (3/10-class subset) = 960k samples = **19.2** full-set epochs
- Mutation: 25 generations x 10 offspring x 2 epochs x 50000 samples = **500** full-set epochs
- Total ~ **519.2** -> rounded to 520

**Output**: `runs_baseline/joint_e<N>_seed0/` with `epoch_<NNN>.pth.tar`, `best.pth.tar`,
`final.pth.tar`, `history.json` (test_acc / train_loss / lr / per_class_acc / elapsed per epoch).
Disk: about 0.3 GiB for 19 ep; about 0.9 GiB for 520 ep with `--save-every 10`.

Checkpoints are raw state_dicts with a 10-class head, byte-compatible with the
population format, and can be fed to `--init-from` directly.

**Measured reference**: about 74 s/epoch while sharing the GPU with exp1; clearly faster alone.

---

## `exp1b_half_budget` (planned)

**Purpose**: move the evolution starting point earlier -- the supervisor's request to
move it "from 200 units to 100 units". The only difference from exp1 is the halved
pretraining budget; everything else is identical, so the two are directly comparable.

```bash
python run_sesil.py --seed 0 --method permute --generations 25 \
       --epochs 3 --out-root ./runs_half
```

- **Fixed 3 epochs per expert, no early stop**: `--target-acc` is not passed, and early
  stopping is skipped by the `target_acc is not None` guard (`sesil/population.py:170`).
  The dry-run configuration line reads `pretrain epochs : 3` with no `(early stop at ...)` suffix.
- This is a different kind of "weak" from exp1's `--target-acc 0.72`: exp1 aligns on
  **accuracy** (every expert trained to the same accuracy, with very different training
  amounts, 2-18 epochs), exp1b aligns on **training amount** (every expert trained
  equally, with accuracy spreading by class difficulty). The latter's budget
  accounting is far cleaner.
- Queue position: exp1 -> 19 ep baseline -> 520 ep baseline -> **exp1b**

**Budget accounting**:

| | subset epochs total | samples | full-set-equivalent epochs |
|---|---|---|---|
| exp1 | 64 (10 experts, 2-18 each) | 64 x 15000 = 960k | **19.2** |
| exp1b | 30 (10 experts x 3) | 30 x 15000 = 450k | **9.0** |

> Warning: 9.0 / 19.2 = **0.469**, not exactly one half. An exact half needs 9.6
> full-set-equivalent = 32 subset epochs = 3.2 epochs per expert (non-integer). 3 is the
> nearest integer, about 6% below the exact half. If the supervisor's 200 -> 100 units
> requires a strict 1/2 there is a 6% shortfall; report it as "3 epochs per expert",
> not "exactly half the budget".

### Final result (25 generations complete, 2026-07-23)

**The half-budget run (9.0 units) and the exp1 weak population (19.2 units) end almost
at the same point.** Weak vs half, per generation:

| gen | 1 | 5 | 11 | 18 | 25 |
|---|---|---|---|---|---|
| weak (19.2 u) | 0.2238 | 0.2696 | 0.5734 | 0.7373 | 0.8048 |
| half (9.0 u) | 0.1894 | 0.1777 | 0.5569 | 0.7487 | **0.8035** |
| absolute difference | 0.034 | 0.092 | 0.017 | 0.011 | **0.0013** |

Early on (gen 5) the half budget lags by 0.09, but from gen 11 the two curves converge
and at gen 25 differ by only **0.0013**. Halving the pretraining budget (19.2 -> 9.0
units) leaves the 25-generation end point essentially unchanged. Together with exp1's
"weak population catches up with strong", the two pieces of evidence point the same
way: **under this permute + 25-generation setting the final state is mainly determined
by the evolution process itself and is highly insensitive to the investment in the
initial population (quality or budget).**

This answers the supervisor's "start from 100 instead of 200 units" question
positively: moving the start earlier does not hurt the end point.

---

## `ssl_simsiam_e50` (planned -> running 2026-07-23)

Unsupervised shared pretraining: SimSiam trains a resnet20x4 backbone on all 50000
CIFAR-10 training images **without labels**, checkpointing the backbone every epoch.
The four exp2 points take backbones from **the same SSL trajectory** at different
epochs, avoiding repeated training and giving the cleanest comparison.

```bash
python train_ssl.py --epochs 50 --half-checkpoint 7.5 --seed 0
```

- **SimSiam architecture**: backbone (resnet20x4 without linear, 256-d features) +
  3-layer projection MLP (256 -> 512 -> 512 -> 512, BN) + 2-layer prediction MLP
  (512 -> 128 -> 512); loss = symmetric negative cosine, stop-gradient on the target
  branch. The projection hidden size is 512 instead of the paper's 2048 because the
  backbone is only 256-d.
- **Augmentation**: RandomResizedCrop(32, scale 0.2-1.0), HFlip, ColorJitter(0.4 x 3, 0.1)
  p=0.8, Grayscale p=0.2; no blur in the CIFAR version. Two views per image. SGD lr=0.06,
  cosine over 50 epochs.
- **Checkpoint format**: backbone state_dict only (resnet20 keys **without** linear.*),
  126 keys. `--init-backbone-only` loads with `strict=False`; the expert classifier
  head stays random.
- **Mid-epoch checkpoint**: `--half-checkpoint 7.5` saves `backbone_e7.5.pth.tar` when
  epoch 8 reaches 50% of its batches, exactly 7.5 cumulative full-set-equivalent
  epochs (for exp2c).
- File names `backbone_e<E>.pth.tar`: e1...e50 per epoch plus e7.5 mid-epoch.
- **Measured**: about 250 s/epoch, 50 epochs ~ 3.5 h (GPU alone).

## exp2 series (planned): SSL shared pretraining + finetuning, budget E + 3S = 9

Four points at **the same start budget, 9.0 full-set-equivalent** (aligned with exp1b),
splitting between SSL shared pretraining (E full-set epochs) and individual finetuning
(S subset epochs per expert). Conversion: 1 SSL epoch = 1 full-set epoch; finetuning
S subset epochs x 10 experts = 10 x S x 15000 samples = 3S full-set-equivalent.

| run | E | backbone | S | finetune 3S | start budget |
|---|---|---|---|---|---|
| `exp2a_ssl_E3S2` | 3 | `backbone_e3` | 2 | 6 | **9.0** |
| **`exp2b_ssl_E6S1`** <- first | 6 | `backbone_e6` | 1 | 3 | **9.0** |
| `exp2c_ssl_E7.5S0.5` | 7.5 | `backbone_e7.5` (mid-epoch, exact) | 0.5 | 1.5 | **9.0** |
| `exp2ref_ssl_E50S1` | 50 | `backbone_e50` | 1 | 3 | 53 (over-budget reference, not in the budget figure) |

**exp2b command** (queue `tools/queue_exp2b.sh`, launched automatically after SSL finishes):
```bash
python run_sesil.py --seed 0 --method permute --generations 25 \
       --init-from runs_ssl/simsiam_e50_seed0/backbone_e6.pth.tar \
       --init-backbone-only --sub-epochs 1 --init-lr 0.03 --out-root ./runs_exp2b
```

Implementation notes:
- **`--init-backbone-only`**: auto-detected (checkpoint without `linear.weight` ->
  backbone-only); backbone loaded, classifier head random; can also be forced.
  `load_init_weights` rejects checkpoints with mismatched keys or missing backbone keys.
- **`--sub-epochs S` (fractional)**: finetuning runs `round(S x batches_per_epoch)` batches
  with per-batch cosine. Only on the `--init-from` path; the integer `--epochs` path
  (exp1/exp1b) is untouched and equivalence is still 15/15. This is what makes S=0.5 possible.
- **`--init-lr` defaults to 0.03**: the SSL backbone representation is good but the
  classifier head is random, so a medium LR is needed to learn the 3-class head without
  wrecking the backbone. Warning: this is a **tunable hyper-parameter**, not an optimum.
  Ideally head and backbone would get different LRs (currently one LR). The smoke test
  (SSL 1.5 ep + finetune 0.5 ep) at 0.03 gave a 10-expert mean of 0.59, proving it works;
  the real SSL representation is stronger.

**E=7.5 rounding**: handled with the mid-epoch checkpoint, exactly 7.5, no rounding bias
(the alternative of taking epoch 7 or 8 with a note was dropped).

**Backbone-quality linear probe** (`tools/linear_probe.py`, queue `tools/queue_probe.sh`,
in parallel with exp2b): after SSL finishes, for each of `backbone_e{1,3,6,7.5,10,20,30,50}`
run "frozen backbone + 1 epoch of linear-head training (full labelled CIFAR-10)" and
record the probe accuracy in `runs_ssl/.../probe.json`. Purpose: **direct evidence** of the
feature quality at the four exp2 checkpoints (E=3/6/7.5/50); if some E evolves poorly,
the probe tells whether the backbone or the evolution is weak, which is needed to
interpret exp2. Backbone frozen, linear head only, very light (8 checkpoints in about
3 minutes), hence run alongside exp2b.
- Uses `load_init_weights(backbone_only=True)` (the same loading path as exp2), so if
  the probe can load a checkpoint the exp2 start can too.
- 1 epoch is a **relative** probe: it underestimates the asymptotic linear-probe accuracy
  (which needs more epochs), but the **ordering and spacing** of e1...e50 is what matters
  here and 1 epoch is light enough. Smoke test (2-epoch weak backbone) gave ~0.20, a
  little above the 0.10 chance level, a plausible magnitude.

**Pre-launch checklist record (2026-07-23)**: equivalence 15/15, roundtrip 3/3,
init-backbone unit test 4/4, SSL + finetune end-to-end real run (real backbone loaded +
`--sub-epochs 0.5` finetuning of 10 experts succeeded). exp1/exp1b are guaranteed
unaffected by the equivalence test.

### Final result (25 generations complete, 2026-07-24; single seed)

gen25 = **0.8538**, above every other run (exp1b 0.8035, exp1 0.8048, strong 0.8404),
with only a 9.0 u start budget. Full 25-generation mean/min/max in
`runs_exp2b/seed0/permute/run.json`.

**The core argument uses "recovery slope", not "trough depth"** -- wording corrected
(2026-07-25). The original idea was "shared SSL initialisation -> shallower merge damage
(shallower trough)", but this **holds for the mean and not for the min**:

| run | gen2 mean | gen2 **min** | gen1 -> gen5 slope |
|---|---|---|---|
| exp2b SSL | 0.1631 | **0.0856** | **+0.0712/gen** |
| exp1b half | 0.1121 | 0.0953 | -0.0029/gen |
| exp1 weak | 0.1253 | 0.0998 | -- |

exp2b's gen-2 **min (0.0856) is lower than exp1b's (0.0953)**, so "shallower trough" is
wrong. The robust graphical evidence is **faster recovery**: exp2b's gen1 -> 5 slope is
+0.071/gen vs exp1b's -0.003/gen; at gen 5 exp2b is already at 0.50 while exp1b is at
0.18 (a gap of 0.33). See `figures/exp2b_trough_zoom.png`. Reports use the recovery-slope
wording throughout.

**Warning: the tail may not have plateaued**: gen24 -> 25 still jumps +0.010
(0.8435 -> 0.8538) while the other runs are crawling by then, so 0.8538 may be a
25-generation truncation rather than a true plateau. For this reason exp2b's
**seed1/seed2 were changed to 40 generations** (error bars and true plateau in one go);
seed0's 25 generations stay as they are, as the matched-generation reference for the
other runs.

**Warning: single seed, only 0.013 above the strong population**: until seed1/2 are back,
reports should say "under a single seed the SSL start reaches or slightly exceeds the
strong population's end point", not a settled conclusion. The multi-seed queue
`tools/queue_multiseed.sh` supplies the error bars.

> **The supervised exp2 variant (joint-baseline init) is on hold**: the original plan
> was a shared start from a mid-run checkpoint of the joint baseline (labelled). At the
> supervisor's direction the unsupervised (SimSiam) version takes priority. Whether the
> supervised version is still run will be decided after the SSL results. `--init-from`
> remains compatible with a joint checkpoint carrying a 10-way head (auto-detected,
> strict loading) and can be run directly if needed.

---

## exp2 series: SSL budget-split summary (updated 2026-07-26, single seed)

Four points at the same start budget 9.0 u (exp2ref at 53 u is the over-budget
reference), split between SSL shared pretraining E and individual finetuning S.
**gen25 of the three points rises monotonically as the budget tilts towards E (E up, S down)**:

| run | E, S | expert start mean | gen2 mean | gen2 min | gen1 -> 5 slope | **gen25** |
|---|---|---|---|---|---|---|
| exp2a | 3, 2 | 0.7663 | 0.1601 | 0.1032 | +0.0544 | 0.8426 |
| exp2b | 6, 1 | 0.7312 | 0.1631 | 0.0856 | +0.0712 | 0.8538 |
| exp2c | 7.5, 0.5 | 0.6273 | 0.1570 | 0.1275 | +0.0834 | **0.8581** |
| exp2d | 9, 0.1 | (planned) | -- | -- | -- | (planned, end point) |
| -- controls -- | | | | | | |
| exp1b (supervised half) | | 0.6435 | 0.1121 | 0.0953 | -0.0029 | 0.8035 |
| exp1 (supervised weak) | | 0.7529 | 0.1253 | 0.0998 | -- | 0.8048 |
| strong (supervised population) | | ~0.93 | 0.2724 | 0.2612 | +0.0832 | 0.8404 |

**Conclusions (single seed)**:
1. **All three exp2 points beat the supervised controls (exp1b 0.80, exp1 0.80, strong
   0.84)** -- the shared SSL start is a consistently effective direction, not a single lucky point.
2. **The split is monotone towards E**: 0.8426 -> 0.8538 -> 0.8581. More SSL pretraining and
   less finetuning is better. The recovery slope is monotone too (+0.054 -> +0.071 -> +0.083):
   the more SSL training the backbone had, the faster the recovery after merging. **This
   points to exp2d (E=9, S~0) as the end point** -- already first in the multi-seed queue.
3. **The start does not predict the end**: exp2a has the highest start (0.7663) and the
   lowest end (0.8426); exp2c the lowest start (0.6273) and the highest end (0.8581).
   See `figures/startpoint_vs_final.png`.

**Variance observation (a feature of the SSL family, not an anomaly)**: the SSL runs have
a **clearly larger mid-run spread than the supervised ones** -- after merging some
offspring recover fast and some slowly; individuals diverge strongly. Example: exp2a
gen5 max-min reaches 0.198 (0.582 vs 0.384); all three exp2 runs **narrow only after
gen22** (gen22-25 spread ~0.01-0.02), with final consistency level with the supervised
family. An intrinsic feature of the SSL start, not a fault.

**exp2ref (E=50, S=1, 53 u over budget) final result**: gen25 = **0.8768** (no crash). Going
from E=7.5 to 50 (SSL budget x 6.7) adds only +0.019 at the end (0.8581 -> 0.8768):
**the evolution end point still has SSL budget to exploit but with sharply diminishing
returns; 9.0 u is close to the knee of this paradigm's return curve.** Final band
(max-min) 0.0100, nearly half exp2c's 0.0192; gen25 min 0.872 vs exp2c's 0.848 -- the
gap is mainly the **lower end being lifted**, confirming that "ample SSL benefits the
hard combinations first" carries through to the end (the hard combination `6_3_5`
already rose from 0.59 in exp2b to 0.66 in exp2ref at the pretraining stage). **This
table only, not the budget figure** (`figures/paper_*.png`) -- its x-position (53 u
start) is not comparable with the 9.0 u group; its role is to answer "how much further
can the evolution ceiling rise once SSL is trained enough".

**Figure deliverables**:
- `figures/paper_main.png` (figure A, for the supervisor): five evolution curves + the
  joint-full light-grey ceiling, y 0.05-0.9, title "Initialization Strategies under
  Matched Budget". exp2ref excluded.
- `figures/paper_honesty.png` (figure B, the honest companion): every element + the
  joint 19ep diamond, y 0-1; the caption notes that joint training wins at matched
  compute and that the evolutionary paradigm's value is in settings where data cannot
  be centralised.
- `figures/startpoint_vs_final.png`: start vs end scatter (descriptive, not a regression).
- `figures/exp2b_trough_zoom.png`: gens 1-8 recovery-slope zoom.

---

## Pre-launch checklist

Established by the gen-2 crash of 2026-07-22. **A single-generation smoke test is
structurally blind to save/load bugs**: a bad checkpoint raises nothing in the
generation that writes it and only fails when the next generation loads it. Therefore:

**Any change touching the save / load path** (`save_offspring`, `utils.save_model`,
`prepare_resnets`, `load_init_weights`, checkpoint format, directory layout) must, before launch:

1. `python tools/check_offspring_roundtrip.py` -- save -> load back with resnet20, must be 3/3
2. `python tools/check_equivalence.py` -- function-by-function comparison with the old scripts, must be 15/15
3. **Confirm the round trip in a real run**: only when the run actually reaches
   generation 2 and evaluates its first population member successfully does it count
   as verified. Getting through generation 1 alone does not.

**Additionally before launching exp2**: even though `--init-from` has unit tests, run the
same check once more before launch (load the target checkpoint -> verify the 10-class
head -> confirm the initial accuracy evaluation is normal), because it introduces a new
"external checkpoint enters the pipeline" path.

**General lesson**: when refactoring, if a function is **redefined** in an old script
(shadowing the version pulled in by `from utils import *`), the in-script definition is
authoritative. Checked: `save_model` was the only one taken from the wrong source;
`evaluate_logits` / `train_logits` are also redefined in the old scripts, but were
copied verbatim from there.

---

## Budget x-axis conversion (for accuracy-vs-compute figures)

Common unit: **full-set-equivalent epoch** = total samples processed / 50000. A 3-class
subset has 15000 training samples, so 1 subset epoch = 0.3 full-set-equivalent epochs.

**Evolution starting point of each run** (x-position of generation 1):

| run | pretraining composition | start (full-set-equivalent epochs) |
|---|---|---|
| `exp1_weak_pretrain` | 64 subset epochs (target-acc 0.72 early stop, 2-18 each) | **19.2** |
| `exp1b_half_budget` | 30 subset epochs (fixed 3 per expert) | **9.0** |
| `exp2a_ssl_E3S2` | SSL E=3 full-set + finetune S=2 subset/expert | **9.0** (3 + 3 x 2) |
| `exp2b_ssl_E6S1` | SSL E=6 full-set + finetune S=1 subset/expert | **9.0** (6 + 3 x 1) |
| `exp2c_ssl_E7.5S0.5` | SSL E=7.5 full-set + finetune S=0.5 subset/expert | **9.0** (7.5 + 3 x 0.5) |
| `exp2ref_ssl_E50S1` | SSL E=50 full-set + finetune S=1 | 53 (over-budget reference, not plotted) |
| `hist_permute_25gen` | `initial/` population, training amount unrecorded | **unknown** |

**Increment per generation**: the mutation stage is 10 offspring x 2 epochs x full set =
**20 full-set-equivalent epochs per generation**. So generation N sits at start + 20(N-1).
Merging and evaluation involve no back-propagation and count as 0.

> exp2's start cannot be fixed before the specific `--init-from` checkpoint is chosen.
> Note that its accounting differs from the other runs: the joint baseline's epochs are
> **full-set** (50000 samples) while exp1/exp1b's pretraining epochs are **subset**
> (15000). Mixing them over- or under-estimates exp2; always convert to sample counts first.
>
> `hist_permute_25gen` uses the pre-existing `initial/` population, whose training cost
> was not recorded (accuracy about 0.93, but the number of epochs is unknown). It
> therefore **cannot be placed on the budget axis** and can only be used in
> matched-generation comparisons. To put it on the budget figure a fresh population
> with recorded cost would have to be trained.

---

## Notes

**Handling unequal generation counts**: when runs have different numbers of generations
(e.g. `hist_permute_25gen` at 25 vs `hist_wavg_11gen` at 11), comparisons are always
**matched-generation** -- only generations present in both are compared (gen 11 vs
gen 11); permute's generation 25 is never compared with wavg's generation 11. This
avoids reading "ran longer" as "better method".

**CSV directories split by date**: the old scripts' CSV path contains
`time.strftime("%Y-%m-%d")` (`evolutionary_wavg_training.py:878`), so a run crossing
midnight is split across two date directories -- both historical runs above did this.
`run_sesil.py` no longer splits by date and writes `runs*/seed<N>/<method>/csv/gen_<N>/` instead.
