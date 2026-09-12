"""One command to run the whole SESiL experiment on any dataset.

    python run_experiment.py --dataset svhn                     # default budget
    python run_experiment.py --dataset svhn --core 6 --finetune 1.5
    python run_experiment.py --dataset cifar100 --classes-per-agent 10 --baseline
    python run_experiment.py --dataset mnist --dry-run          # show the plan

It chains the three stages that otherwise have to be run by hand:

    fetch data (if missing)
      -> train_ssl.py         : shared SimSiam backbone, checkpointed per epoch
      -> run_sesil.py         : population pretrain + evolution, --init-from
                                the backbone epoch named by --core
      -> train_joint_baseline : optional reference curve (--baseline)

and does the budget arithmetic. Budgets are quoted the way the figures quote
them, "core + finetune" in full-set-equivalent epochs:

    --core X      X epochs of unsupervised pretraining (shared, trained once)
    --finetune Y  TOTAL supervised finetune across all agents

Y is converted to the per-agent --sub-epochs S the CLI actually takes:

    Y = n_agents * (classes_per_agent / num_classes) * S

so on CIFAR-10's 10 agents x 3 classes that is the familiar Y = 3S, and the
same notation carries to a dataset with different agents/classes.

Every stage is skipped if its output already exists, so re-running after an
interruption resumes rather than repeats.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sesil.data import DATASETS, get_spec  # noqa: E402

PY = sys.executable


def sub_epochs_for(finetune_total, n_agents, classes_per_agent, num_classes):
    """Per-agent S from a TOTAL finetune budget in full-set epochs."""
    share = n_agents * (classes_per_agent / num_classes)
    if share <= 0:
        raise ValueError("n_agents and classes_per_agent must be positive")
    return finetune_total / share


def backbone_path(ssl_dir, core):
    return os.path.join(ssl_dir, f"backbone_e{core:g}.pth.tar")


def run(cmd, dry):
    print("\n$ " + " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.call([str(c) for c in cmd])


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="Run SSL pretraining + SESiL evolution (+ baseline) on one dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    p.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("budget (full-set-equivalent epochs)")
    g.add_argument("--core", type=float, default=9.0,
                   help="unsupervised pretraining budget X; reads backbone_e<X>")
    g.add_argument("--finetune", type=float, default=0.3,
                   help="TOTAL supervised finetune budget Y across all agents")
    g.add_argument("--ssl-epochs", type=int, default=50,
                   help="length of the SSL run (must be >= --core)")
    g.add_argument("--generations", type=int, default=25)
    g.add_argument("--updates-per-epoch", type=int, default=20)

    g = p.add_argument_group("population")
    g.add_argument("--num-agents", type=int, default=10)
    g.add_argument("--classes-per-agent", type=int, default=3)
    g.add_argument("--method", default="permute",
                   choices=["permute", "wavg", "zipit"])
    g.add_argument("--init-lr", type=float, default=0.03)

    g = p.add_argument_group("stages")
    g.add_argument("--baseline", action="store_true",
                   help="also train the joint baseline (matched total budget)")
    g.add_argument("--baseline-epochs", type=int, default=None,
                   help="default: core + finetune + 20*generations, rounded up")
    g.add_argument("--skip-ssl", action="store_true")
    g.add_argument("--skip-sesil", action="store_true")
    g.add_argument("--out-root", default=None,
                   help="default: ./runs_<dataset>_<core>+<finetune>")
    g.add_argument("--dry-run", action="store_true",
                   help="print the plan and the resolved budget, run nothing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = get_spec(args.dataset)
    dry = args.dry_run

    if args.core > args.ssl_epochs:
        print(f"error: --core {args.core} exceeds --ssl-epochs {args.ssl_epochs}; "
              f"the backbone checkpoint would never be written.")
        return 2

    S = sub_epochs_for(args.finetune, args.num_agents,
                       args.classes_per_agent, spec.num_classes)
    ssl_tag = f"simsiam_{args.dataset}_e{args.ssl_epochs}_seed{args.seed}"
    ssl_dir = os.path.join("./runs_ssl", ssl_tag)
    backbone = backbone_path(ssl_dir, args.core)
    out_root = args.out_root or f"./runs_{args.dataset}_{args.core:g}+{args.finetune:g}"
    evo_epochs = 20.0 * args.generations
    total = args.core + args.finetune + evo_epochs
    baseline_epochs = args.baseline_epochs or int(round(total))

    print("=" * 68)
    print(f"SESiL experiment: {args.dataset}")
    print("=" * 68)
    print(f"  dataset          : {args.dataset} "
          f"({spec.num_classes} classes, {spec.train_size} train images)")
    print(f"  population       : {args.num_agents} agents x "
          f"{args.classes_per_agent} classes")
    print(f"  budget           : {args.core:g} core + {args.finetune:g} finetune "
          f"+ {evo_epochs:g} evolution = {total:g} epochs")
    print(f"  -> --sub-epochs  : {S:.5f} per agent "
          f"(Y = {args.num_agents} x {args.classes_per_agent}/{spec.num_classes} x S)")
    print(f"  updates/epoch    : {args.updates_per_epoch}")
    print(f"  ssl backbone     : {backbone}")
    print(f"  output           : {out_root}")
    if args.baseline:
        print(f"  baseline         : {baseline_epochs} epochs (budget-matched)")
    print("=" * 68)

    # -- stage A: data -------------------------------------------------
    try:
        from sesil.data import get_full_loaders
        get_full_loaders(args.dataset, batch_size=spec.batch_size, seed=args.seed)
        print(f"\n[data] {args.dataset} present")
    except RuntimeError:
        print(f"\n[data] {args.dataset} missing -- fetching")
        if run([PY, "tools/fetch_dataset.py", args.dataset], dry) != 0:
            return 1

    # -- stage B: SSL backbone -----------------------------------------
    if args.skip_ssl:
        print("\n[ssl] skipped (--skip-ssl)")
    elif os.path.exists(backbone):
        print(f"\n[ssl] {backbone} exists -- skipping SSL")
    else:
        cmd = [PY, "train_ssl.py", "--dataset", args.dataset,
               "--epochs", args.ssl_epochs, "--seed", args.seed,
               "--updates-per-epoch", args.updates_per_epoch,
               "--tag", ssl_tag]
        if args.core != int(args.core):        # fractional core needs a snapshot
            cmd += ["--half-checkpoint", args.core]
        if run(cmd, dry) != 0:
            return 1

    # -- stage C: SESiL -------------------------------------------------
    if args.skip_sesil:
        print("\n[sesil] skipped (--skip-sesil)")
    else:
        cmd = [PY, "run_sesil.py", "--dataset", args.dataset,
               "--seed", args.seed, "--method", args.method,
               "--generations", args.generations,
               "--num-agents", args.num_agents,
               "--classes-per-agent", args.classes_per_agent,
               "--init-from", backbone, "--init-backbone-only",
               "--sub-epochs", f"{S:.5f}", "--init-lr", args.init_lr,
               "--updates-per-epoch", args.updates_per_epoch,
               "--out-root", out_root]
        if run(cmd, dry) != 0:
            return 1

    # -- stage D: baseline ----------------------------------------------
    if args.baseline:
        cmd = [PY, "train_joint_baseline.py", "--dataset", args.dataset,
               "--epochs", baseline_epochs, "--seed", args.seed,
               "--updates-per-epoch", args.updates_per_epoch,
               "--save-every", 10,
               "--tag", f"joint_{args.dataset}_e{baseline_epochs}_seed{args.seed}"]
        if run(cmd, dry) != 0:
            return 1

    print(f"\nDone. Results under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
