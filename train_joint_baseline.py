"""Joint-training baseline for SESiL comparisons.

    # (a) budget-matched to exp1's PRETRAIN cost (19.2 full-set epochs)
    python train_joint_baseline.py --epochs 19

    # (c) budget-matched to exp1's TOTAL cost (19.2 pretrain + 500 mutation)
    python train_joint_baseline.py --epochs 520 --save-every 10

Each budget gets its own run with a complete cosine schedule, so each is the
true optimum for that budget. Do NOT read epoch 19 out of the 520-epoch run and
call it the 19-epoch baseline -- the LR is still near peak there. Mid-run
checkpoints exist only to seed exp2 (`run_sesil.py --init-from ...`).

Checkpoints are raw state_dicts with a 10-wide head, byte-compatible with the
population format, so --init-from consumes them directly.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from sesil.data import DATASETS  # noqa: E402
from sesil.joint_baseline import JointConfig, print_summary, train_joint  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        prog="train_joint_baseline.py",
        description="Train resnet20x4 on all 10 CIFAR-10 classes, "
                    "checkpointing every epoch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=19,
                   help="19 matches exp1 pretrain budget; 520 matches total budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS),
                   help="dataset to train on; also sets the head width")
    p.add_argument("--out-root", default="./runs_baseline")
    p.add_argument("--tag", default=None,
                   help="run directory name (default joint_e<epochs>_seed<seed>)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--updates-per-epoch", type=int, default=20, metavar="N",
                   help="optimizer steps per epoch -- the x-axis unit shared "
                        "with run_sesil.py. 0 = one full pass (legacy: 100)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=1,
                   help="save a checkpoint every N epochs (raise it for long runs)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved plan (incl. disk estimate) and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    cfg = JointConfig(
        epochs=args.epochs,
        seed=args.seed,
        dataset=args.dataset,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        updates_per_epoch=args.updates_per_epoch,
        num_workers=args.num_workers,
        device=args.device,
        out_root=args.out_root,
        tag=args.tag,
        save_every=args.save_every,
    )

    print("=" * 68)
    print("Joint-training baseline")
    print("=" * 68)
    print(cfg.describe())
    print("=" * 68)

    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    history = train_joint(cfg)
    print_summary(history)
    print(f"\nHistory: {cfg.history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
