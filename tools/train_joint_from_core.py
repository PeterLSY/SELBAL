"""Joint-training baseline initialised from the shared SSL core (proposal
section 4: "both the baseline and every member of the generation-0 population
are initialized from this core-skill network").

The existing joint baselines (runs_baseline/joint_e19, joint_e520) start from
scratch. This wrapper reuses sesil.joint_baseline.train_joint unchanged and
only swaps the model factory so the backbone is loaded from the core exactly
as run_sesil.py --init-from --init-backbone-only does for the experts (the
10-wide head stays randomly initialised). Nothing in sesil/ is modified.

    python tools/train_joint_from_core.py --epochs 19 --lr 0.03 --updates-per-epoch 0
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

import sesil.joint_baseline as jb  # noqa: E402
from sesil.population import load_init_weights  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", default="runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
    p.add_argument("--epochs", type=int, default=19)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--lr", type=float, default=0.03, help="exp2d's experts used --init-lr 0.03 from this core")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--updates-per-epoch", type=int, default=0,
                   help="0 = full pass (the definition every existing exp2* result and joint_e19 used)")
    p.add_argument("--out-root", default="./runs_baseline")
    p.add_argument("--tag", default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    init_from = args.init_from
    _orig = jb.resnet20

    def from_core(w, num_classes):
        model = _orig(w=w, num_classes=num_classes)
        return load_init_weights(model, init_from, "cpu", backbone_only=True, num_classes=num_classes)

    jb.resnet20 = from_core  # train_joint builds the model through this name

    core_tag = os.path.splitext(os.path.splitext(os.path.basename(init_from))[0])[0]
    tag = args.tag or f"joint_core-{core_tag}_e{args.epochs}_lr{args.lr:g}_seed{args.seed}"
    cfg = jb.JointConfig(epochs=args.epochs, seed=args.seed, dataset=args.dataset, lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay, batch_size=args.batch_size,
                         updates_per_epoch=args.updates_per_epoch, device=args.device,
                         out_root=args.out_root, tag=tag, save_every=args.save_every)
    print(f"init from core: {init_from}\n{cfg.describe()}\n")
    hist = jb.train_joint(cfg)
    jb.print_summary(hist)
    print(f"\nbest: epoch {hist['best']['epoch']}, test acc {hist['best']['test_acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
