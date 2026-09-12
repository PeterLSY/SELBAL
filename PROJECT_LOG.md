# SESiL project optimisation log

This file records the refactoring and experimental work on the SESiL
(evolutionary model merging) code base. Status is always taken from the file
system / CSVs; lines marked with a warning sign have no data behind them yet.

---

## 1. Initial small tasks

- `plot_evolution.py`: min-max shaded bands (green PTA, orange merged;
  `np.nanmin/nanmax` for NaN)
- Read through `train_cifar_models.py` (epochs / lr / batch / checkpoint / class
  assignment) and flagged 3 latent problems: CIFAR-100 loaded but evaluated as 10
  classes, a `cifar8_classes` parameter-name bug, a hard-coded `per_class_acc[8:]`
- Documented the per-generation workflow of `evolutionary_wavg_training.py`

## 2. Core engineering: a unified pipeline

Survey: the wavg and permute scripts (900 lines) differ in only 5 substantive
places; the 5 evolution scripts are >90% duplicated.

- **`sesil/` package**: config / paths / seeding / mapping_guard / data / population /
  evaluation / mating / merging / mutation / pipeline / ssl_pretrain / joint_baseline
- **`run_sesil.py`** single entry point; every hard-coded constant lifted to a CLI
  flag; outputs isolated per seed
- The old `training_scripts/*` are untouched (kept as the equivalence reference)

## 3. Two incidents and their fixes

| incident | root cause | fix |
|---|---|---|
| gen-2 crash | `utils.save_model` does not unwrap `ModelMerge` (the old scripts shadow it with a same-named version; missed in the refactor) | `save_offspring()` + `tools/check_offspring_roundtrip.py` |
| multi-seed queue crashed entirely | a detached bash does not inherit the conda environment; bare `python` has no torch | pin the conda python's absolute path `$PY` in the queue |

## 4. New capabilities

- Joint-training baseline: `joint_baseline.py` + `train_joint_baseline.py`
- SSL unsupervised pretraining: `ssl_pretrain.py` (SimSiam adapted to resnet20x4) +
  `train_ssl.py`, exact mid-epoch checkpoints
- `--init-from` with three paths: joint (strict) / SSL backbone
  (`--init-backbone-only`) / fractional finetuning (`--sub-epochs`)
- Closed-form zero-finetune calibration: `--calibrate closed_form` (ridge closed-form
  head; measured 0.60-0.76, above 3-batch SGD)

## 5. Experiment matrix (actual state)

### Completed (single seed)

| run | configuration | gen25 |
|---|---|---|
| exp1 | weak pretraining, target-acc 0.72 | 0.8048 |
| exp1b | fixed 3 ep, half budget | 0.8035 |
| exp2a | SSL E3+S2 | 0.8426 |
| exp2b | SSL E6+S1 | 0.8538 |
| exp2c | SSL E7.5+S0.5 | 0.8581 |
| exp2d | SSL E9+S0.1 | 0.8722 |
| exp2ref | SSL E50+S1 (over budget) | 0.8768 |
| joint-19ep / 520ep | joint-training baselines | 0.8886 / 0.9572 |
| SSL SimSiam | 50 epochs | loss -0.86 |
| hist_permute (historical strong population) | permute, 25 generations | 0.8404 |

### Running / queued / never started

- **Running**: exp2b_seed1 (gen24/40)
- **Queue** (`tools/queue_multiseed.sh`, PID at run time):
  exp2b_s2 -> exp2c_s1 -> exp2c_s2 -> exp1_s1 -> exp1_s2 -> exp2e
- Warning, **never started**: exp2c two seeds, exp1 two seeds, exp2e

## 6. Key conclusions (**all single seed, not reproduced across seeds**)

1. SSL budget split is monotone in E: the end point rises with the SSL pretraining
   amount (0.843 -> 0.854 -> 0.858 -> 0.872 -> 0.877); finetuning S is nearly dispensable
2. The start does not predict the end: exp2d has the lowest start (0.557) but the
   second-highest end; strong starts at 0.93 and ends at only 0.840
3. Collapse recovery: SSL family span = 1, supervised weak start span = 3-4; retain
   has a 0.2 gradient (sensitive to the starting-point strategy)
4. The weak population catches up with the strong one in 25 generations (gap only 0.036)

## 7. Deliverables

- **Figures** (`figures/`): paper_main, paper_honesty, startpoint_vs_final,
  collapse_recovery_zoom+table, exp2b_*, budget/generation_axis_compare
- **Documents**: `experiments.md` (full ledger + incidents + budget accounting),
  `weekly_cheatsheet.md`, this file
- **Regression tests**: check_equivalence (15/15), check_offspring_roundtrip (3/3),
  check_init_backbone (4/4)

## 8. Honesty boundary (data does not exist, do not cite)

**As of this log no multi-seed reproduction has completed** (the queue log is the
authority). Therefore all of the following are pending:

- Any "seed spread / error bar" (such as ±0.004, ±0.019, ±0.0015)
- Any "40-generation plateau value" (such as 0.8742): 0 forty-generation trajectories finished
- The "catalyst outlier" observation of the exp1 reproduction
- Fragile-pairing reproduction across seeds
- Whether the late two-stage climb reproduces (plus the methodological obstacle:
  from gen12 on every pairing covers all 10 classes, so the class-composition
  dimension carries zero information)

Meeting material may only endorse single-seed results (the completed parts of
sections 5 and 6).

## 9. Pre-launch checklist (established by the gen-2 incident)

Any change that touches the save / load path must, before launch:
1. `tools/check_offspring_roundtrip.py` -> 3/3
2. `tools/check_equivalence.py` -> 15/15
3. Run for real until generation 2 loads its first member successfully (getting
   through generation 1 alone does not count)

From now on every "generated" claim must be accompanied by actual `ls`/`dir` output.
