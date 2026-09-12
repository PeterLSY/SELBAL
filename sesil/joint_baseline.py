"""Joint-training baseline: one resnet20x4 on all 10 CIFAR-10 classes.

Three uses, per the experiment plan:
  (a) budget-matched baseline -- `--epochs 19` matches exp1's PRETRAIN budget
      (10 experts x 64 epochs total over 15000-sample subsets = 960k samples
      = 19.2 full-set epochs).
  (b) starting point for exp2's shared pretraining, via `run_sesil.py
      --init-from <one of the epoch_*.pth.tar>`.
  (c) full-budget baseline -- `--epochs 520` matches exp1's TOTAL budget
      (19.2 pretrain + 25 generations x 10 offspring x 2 mutation epochs = 500).

Each budget is trained as its OWN run with a complete cosine schedule, so each
is genuinely optimal for that budget. Mid-run checkpoints of the long run are
NOT valid baselines for a shorter budget -- at epoch 19 of a 520-epoch cosine
the LR is still near its peak. Use them only for (b).

Checkpoints match the population format exactly: raw state_dict, 10-wide head,
so `--init-from` can load them without any adaptation.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from models.resnets import resnet20
from utils import save_model

from .data import cycle_batches, get_full_loaders, get_spec, updates_for
from .seeding import set_all_seeds

NUM_CLASSES = 10  # CIFAR-10 default; a run uses its dataset spec's count
CKPT_MIB = 17.8  # measured size of one resnet20x4 state_dict


@dataclass
class JointConfig:
    epochs: int = 19
    seed: int = 0
    dataset: str = "cifar10"   # key into sesil.data.DATASETS
    model_width: int = 4
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 500
    # Optimizer steps per epoch. The epoch is the x-axis unit of every figure,
    # so it is defined by update count, not by a pass over the data: with 20,
    # one epoch is 20 updates here AND 20 in SESiL's mutation stage, whatever
    # the batch size. 0 = legacy, one full pass (100 updates at batch 500).
    updates_per_epoch: int = 20
    num_workers: int = 0
    device: str = "cuda"
    out_root: str = "./runs_baseline"
    tag: Optional[str] = None
    save_every: int = 1
    resume: bool = False

    @property
    def spec(self):
        return get_spec(self.dataset)

    @property
    def num_classes(self):
        return self.spec.num_classes

    @property
    def run_dir(self):
        name = self.tag or f"joint_e{self.epochs}_seed{self.seed}"
        return os.path.join(self.out_root, name)

    @property
    def history_path(self):
        return os.path.join(self.run_dir, "history.json")

    def describe(self):
        n_saved = (self.epochs + self.save_every - 1) // self.save_every
        disk = n_saved * CKPT_MIB / 1024
        return "\n".join(
            "  " + ln
            for ln in [
                f"epochs      : {self.epochs}",
                f"seed        : {self.seed}",
                f"dataset     : {self.dataset} ({self.num_classes} classes)",
                f"lr          : {self.lr} (SGD momentum={self.momentum}, "
                f"wd={self.weight_decay}, cosine T_max={self.epochs})",
                f"batch size  : {self.batch_size}",
                f"updates/ep  : {self.updates_per_epoch or 'full pass'}",
                f"device      : {self.device}",
                f"out dir     : {self.run_dir}",
                f"save every  : {self.save_every} epoch(s) "
                f"-> {n_saved} checkpoints, ~{disk:.1f} GiB",
            ]
        )


@torch.no_grad()
def evaluate(model, loader, device, num_classes=NUM_CLASSES):
    """Top-1 over all classes, plus per-class accuracy."""
    model.eval()
    correct = total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, list):
            out = out[0]
        pred = out.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for c in range(num_classes):
            m = y == c
            if m.any():
                per_class_total[c] += m.sum().item()
                per_class_correct[c] += (pred[m] == y[m]).sum().item()
    per_class = [
        (c / t if t else 0.0) for c, t in zip(per_class_correct, per_class_total)
    ]
    return correct / max(total, 1), per_class


def ckpt_path(cfg, epoch):
    return os.path.join(cfg.run_dir, f"epoch_{epoch:03d}.pth.tar")


def train_joint(cfg: JointConfig):
    os.makedirs(cfg.run_dir, exist_ok=True)
    set_all_seeds(cfg.seed)

    train_loader, test_loader = get_full_loaders(
        cfg.dataset, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )

    model = resnet20(w=cfg.model_width, num_classes=cfg.num_classes).to(cfg.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )
    loss_fn = nn.CrossEntropyLoss()

    history = {"config": asdict(cfg), "epochs": []}
    best_acc, best_epoch = 0.0, 0
    t0 = time.time()

    nb = updates_for(train_loader, cfg.updates_per_epoch)
    batches = cycle_batches(train_loader)

    pbar = tqdm(range(1, cfg.epochs + 1), desc="joint baseline")
    for epoch in pbar:
        model.train()
        running = 0.0
        for _ in range(nb):
            x, y = next(batches)
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if isinstance(logits, list):
                logits = logits[0]
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        acc, per_class = evaluate(model, test_loader, cfg.device, cfg.num_classes)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            save_model(model, os.path.join(cfg.run_dir, "best.pth.tar"))

        saved = None
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            saved = ckpt_path(cfg, epoch)
            save_model(model, saved)

        history["epochs"].append(
            {
                "epoch": epoch,
                "test_acc": acc,
                "train_loss": running / max(nb, 1),
                "lr": lr_now,
                "per_class_acc": per_class,
                "checkpoint": os.path.basename(saved) if saved else None,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        history["best"] = {"epoch": best_epoch, "test_acc": best_acc}
        with open(cfg.history_path, "w") as f:
            json.dump(history, f, indent=2)

        pbar.set_postfix(acc=f"{acc:.4f}", best=f"{best_acc:.4f}", lr=f"{lr_now:.4f}")

    save_model(model, os.path.join(cfg.run_dir, "final.pth.tar"))
    history["total_seconds"] = round(time.time() - t0, 1)
    with open(cfg.history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def print_summary(history, every=None):
    eps = history["epochs"]
    if not eps:
        print("(no epochs recorded)")
        return
    step = every or max(1, len(eps) // 20)
    print()
    print(f"{'Epoch':>6} | {'Test acc':>9} | {'Train loss':>10} | {'LR':>8} | {'Elapsed':>8}")
    print("-" * 56)
    for e in eps:
        if e["epoch"] % step == 0 or e["epoch"] in (1, len(eps)):
            print(
                f"{e['epoch']:>6} | {e['test_acc']:>9.4f} | {e['train_loss']:>10.4f} | "
                f"{e['lr']:>8.5f} | {e['elapsed_s']:>7.1f}s"
            )
    print("-" * 56)
    b = history["best"]
    print(f"best: epoch {b['epoch']}, test acc {b['test_acc']:.4f}")
    if "total_seconds" in history:
        print(f"total: {history['total_seconds']:.1f}s")
