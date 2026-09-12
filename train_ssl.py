"""SimSiam unsupervised pretraining of a resnet20x4 backbone on CIFAR-10.

    # one 50-epoch run; every exp2 point reads a different epoch's backbone
    python train_ssl.py --epochs 50 --half-checkpoint 7.5

Produces runs_ssl/<tag>/backbone_e<E>.pth.tar for E = 1..50 plus any
--half-checkpoint, and history.json with the SSL loss curve. Feed a backbone to
the evolution via:

    python run_sesil.py --init-from runs_ssl/<tag>/backbone_e6.pth.tar \\
           --init-backbone-only --sub-epochs 1 ...

The backbone is classifier-free; --init-backbone-only loads it and leaves each
expert's linear layer randomly initialised.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402


@dataclass
class SSLConfig:
    epochs: int = 50
    seed: int = 0
    dataset: str = "cifar10"   # key into sesil.data.DATASETS
    model_width: int = 4
    lr: float = 0.06                 # 0.03 * batch/256, batch=512
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 512
    # Steps per epoch, same unit as run_sesil.py / train_joint_baseline.py, so
    # a backbone_e6 checkpoint costs the same 6 epochs the figures charge it.
    # 0 = one full pass (97 updates at batch 512 with drop_last).
    updates_per_epoch: int = 20
    num_workers: int = 0
    device: str = "cuda"
    out_root: str = "./runs_ssl"
    tag: Optional[str] = None
    half_checkpoints: List[float] = field(default_factory=list)

    @property
    def run_dir(self):
        default = f"simsiam_e{self.epochs}_seed{self.seed}"
        if self.dataset != "cifar10":
            default = f"simsiam_{self.dataset}_e{self.epochs}_seed{self.seed}"
        return os.path.join(self.out_root, self.tag or default)

    @property
    def history_path(self):
        return os.path.join(self.run_dir, "history.json")

    def describe(self):
        return "\n".join("  " + ln for ln in [
            f"epochs          : {self.epochs}",
            f"seed            : {self.seed}",
            f"dataset         : {self.dataset}",
            f"lr              : {self.lr} (SGD momentum={self.momentum}, wd={self.weight_decay}, cosine)",
            f"batch size      : {self.batch_size}",
            f"updates/epoch   : {self.updates_per_epoch or 'full pass'}",
            f"half checkpoints: {self.half_checkpoints or 'none'}",
            f"out dir         : {self.run_dir}",
        ])


def build_parser():
    p = argparse.ArgumentParser(
        prog="train_ssl.py",
        description="SimSiam backbone pretraining on unlabelled CIFAR-10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="cifar10",
                   help="dataset to pretrain the backbone on (labels unused)")
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--updates-per-epoch", type=int, default=20, metavar="N",
                   help="optimizer steps per epoch -- the x-axis unit shared "
                        "with run_sesil.py. 0 = one full pass (97 at batch 512)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-root", default="./runs_ssl")
    p.add_argument("--tag", default=None)
    p.add_argument("--half-checkpoint", type=float, action="append", default=None,
                   dest="half_checkpoints", metavar="E",
                   help="also save a backbone at fractional epoch E (e.g. 7.5); repeatable")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from sesil.ssl_pretrain import train_ssl, print_summary

    cfg = SSLConfig(
        epochs=args.epochs, seed=args.seed, dataset=args.dataset, lr=args.lr,
        batch_size=args.batch_size, updates_per_epoch=args.updates_per_epoch,
        num_workers=args.num_workers,
        out_root=args.out_root, tag=args.tag, device=args.device,
        half_checkpoints=args.half_checkpoints or [],
    )

    print("=" * 68)
    print("SimSiam SSL pretraining")
    print("=" * 68)
    print(cfg.describe())
    print("=" * 68)

    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    history = train_ssl(cfg)
    print_summary(history)
    print(f"\nHistory: {cfg.history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
