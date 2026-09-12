"""Single entry point for a full SESiL run: pretrain + evolution, one seed.

    python run_sesil.py --seed 0 --method wavg --generations 15
    python run_sesil.py --seed 0 --method permute      # reuses seed 0 population
    python run_sesil.py --seed 1 --pretrain-epochs 80 --target-acc 0.93

Everything lands under --out-root/seed<N>/. The legacy scripts under
training_scripts/ are untouched and still runnable; existing results in
./csvs/ and ./checkpoints/cifar10_evolution/{initial,wavg,permute}/ are never
written to.
"""

import argparse
import os
import sys

# All of utils.py's defaults (mapping.json, ./data, ./checkpoints) are relative
# to the CWD, so pin it to the repo root before importing anything from utils.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from sesil.config import METHOD_MERGING_FN, VERIFIED_METHODS, RunConfig  # noqa: E402
from sesil.data import DATASETS  # noqa: E402
from sesil.paths import RunPaths  # noqa: E402
from sesil.pipeline import print_history, run_evolution  # noqa: E402
from sesil.population import pretrain_population, print_summary  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_sesil.py",
        description="Pretrain a SESiL population and evolve it, in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("run")
    g.add_argument("--seed", type=int, default=0,
                   help="controls training randomness only; class assignments are fixed")
    g.add_argument("--method", default="wavg", choices=sorted(METHOD_MERGING_FN),
                   help="merging method (wavg/permute are verified; zipit is untested)")
    g.add_argument("--generations", type=int, default=15)
    g.add_argument("--start-from", type=int, default=0,
                   help="generation number offset, for resuming")
    g.add_argument("--out-root", default="./runs")
    g.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS),
                   help="dataset to run on; sets the data, the head width and "
                        "the full-set-equivalent epoch")
    g.add_argument("--num-agents", dest="n_agents", type=int, default=10,
                   help="population size (only used when no reference "
                        "population exists for this dataset)")
    g.add_argument("--classes-per-agent", type=int, default=3,
                   help="classes each agent is trained on (same caveat)")
    # NOTE: no --mapping-file flag on purpose. utils.decode_labels is called
    # without one from the carried-over evaluation/merging code, so a custom
    # map would be written but never read. One shared mapping.json is safe
    # across datasets anyway: keys are md5 of the label string ("49_97_53"),
    # which cannot collide between label spaces.
    g.add_argument("--config-name", default="cifar_evolution_resnet20",
                   help="evolution config in configs/; --dataset overrides its "
                        "dataset name, so the default works for any dataset")
    g.add_argument("--updates-per-epoch", type=int, default=20, metavar="N",
                   help="optimizer steps per full-set-equivalent epoch, applied "
                        "to pretrain and mutation alike (a 3-class subset gets "
                        "0.3*N). Same unit as train_joint_baseline.py. "
                        "0 = one full pass (legacy: 100 full-set / 30 subset)")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = p.add_argument_group("pretrain")
    g.add_argument("--epochs", "--pretrain-epochs", dest="pretrain_epochs",
                   type=int, default=60)
    g.add_argument("--target-acc", type=float, default=None,
                   help="stop a model early once its 3-class test acc reaches this")
    g.add_argument("--pretrain-lr", type=float, default=0.1)
    g.add_argument("--batch-size", dest="pretrain_batch_size", type=int, default=500)
    g.add_argument("--num-workers", dest="pretrain_num_workers", type=int, default=0)
    g.add_argument("--force-pretrain", action="store_true",
                   help="retrain population members even if checkpoints exist")
    g.add_argument("--reference-population", default=None,
                   help="read-only source of the class assignments "
                        "(default: ./checkpoints/<dataset>_evolution/initial; "
                        "if absent, assignments are drawn at --seed)")
    g.add_argument("--init-from", default=None, metavar="PATH",
                   help="initialise every expert from this checkpoint instead of "
                        "from scratch, then finetune to --target-acc (exp2 entry "
                        "point; needs a 10-wide head, e.g. runs_baseline/*/epoch_005.pth.tar)")
    g.add_argument("--init-lr", type=float, default=0.01,
                   help="LR for the --init-from finetune (pretrain-lr would wreck "
                        "already-converged weights)")
    g.add_argument("--init-backbone-only", dest="init_backbone_only",
                   action="store_const", const=True, default=None,
                   help="treat --init-from as a bare backbone (no classifier); "
                        "load the backbone and leave each expert's head random "
                        "(default: auto-detect from the checkpoint)")
    g.add_argument("--sub-epochs", dest="finetune_sub_epochs", type=float, default=None,
                   metavar="S",
                   help="fractional finetune length in SUBSET epochs (exp2); "
                        "runs round(S*batches) batches. Only on the --init-from path")
    g.add_argument("--calibrate", choices=["closed_form"], default=None,
                   help="exp2e: closed-form ridge head calibration, ZERO SGD "
                        "(the strict S->0 budget-ledger endpoint). --init-from backbone only")

    g = p.add_argument_group("evolution")
    g.add_argument("--mutation-epochs", type=int, default=2)
    g.add_argument("--mutation-lr", type=float, default=0.001)
    g.add_argument("--stop-node", type=int, default=21)
    g.add_argument("--merge-a", type=float, default=0.0001)
    g.add_argument("--merge-b", type=float, default=0.075)
    g.add_argument("--mate-threshold", type=float, default=0.5)

    g = p.add_argument_group("stages")
    g.add_argument("--skip-pretrain", action="store_true")
    g.add_argument("--skip-evolution", action="store_true")
    g.add_argument("--dry-run", action="store_true",
                   help="print the resolved plan and exit")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # The reference population is per dataset: CIFAR-10's fixed 10x3 assignment
    # must not be reused for a dataset with a different label space.
    reference_population = (
        args.reference_population
        or f"./checkpoints/{args.dataset}_evolution/initial"
    )

    cfg = RunConfig(
        seed=args.seed,
        method=args.method,
        generations=args.generations,
        start_from=args.start_from,
        out_root=args.out_root,
        device=args.device,
        dataset=args.dataset,
        n_agents=args.n_agents,
        classes_per_agent=args.classes_per_agent,
        config_name=args.config_name,
        updates_per_epoch=args.updates_per_epoch,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_num_workers=args.pretrain_num_workers,
        target_acc=args.target_acc,
        force_pretrain=args.force_pretrain,
        reference_population=reference_population,
        init_from=args.init_from,
        init_lr=args.init_lr,
        init_backbone_only=args.init_backbone_only,
        finetune_sub_epochs=args.finetune_sub_epochs,
        calibrate=args.calibrate,
        mutation_epochs=args.mutation_epochs,
        mutation_lr=args.mutation_lr,
        stop_node=args.stop_node,
        merge_a=args.merge_a,
        merge_b=args.merge_b,
        mate_threshold=args.mate_threshold,
        skip_pretrain=args.skip_pretrain,
        skip_evolution=args.skip_evolution,
    )

    paths = RunPaths(cfg.out_root, cfg.seed, cfg.method, cfg.model_width)
    paths.makedirs()

    print("=" * 68)
    print("SESiL run")
    print("=" * 68)
    print(cfg.describe())
    print(f"  output           : {paths.seed_dir}")
    if cfg.method not in VERIFIED_METHODS:
        print(f"  !! method {cfg.method!r} is wired up but NOT equivalence-checked")
    print("=" * 68)

    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    # -- stage 0: pretrain ----------------------------------------------
    if not cfg.skip_pretrain:
        print("\n--- Stage 0: pretrain population ---")
        summary = pretrain_population(cfg, paths)
        print_summary(summary)
    else:
        print("\n--- Stage 0: skipped (--skip-pretrain) ---")

    # -- stages 1-4: evolution ------------------------------------------
    if not cfg.skip_evolution:
        print("\n--- Stages 1-4: evolution ---")
        history = run_evolution(cfg, paths)
        print_history(history)
    else:
        print("\n--- Evolution skipped (--skip-evolution) ---")

    print(f"\nDone. Results under {paths.seed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
