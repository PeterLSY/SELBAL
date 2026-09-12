"""Stage -1 (exp2): unsupervised shared pretraining with SimSiam.

Trains a resnet20x4 BACKBONE on all 50000 CIFAR-10 training images with NO
labels, then hands the backbone to run_sesil.py --init-from --init-backbone-only
as a shared starting point for every expert.

SimSiam (Chen & He, 2021): two augmented views of an image go through a shared
backbone + projection MLP; a prediction MLP maps one view's projection to the
other's, and the target branch is stop-gradient'd. Loss is the negative cosine
similarity, symmetrised. No negatives, no momentum encoder, no large batch.

Backbone adaptation for resnet20x4:
  * feature dim is w*64 = 256 (avgpool output), NOT the 2048 of a full ResNet,
    so the projection hidden width is scaled down to 512 rather than 2048.
  * the backbone IS a resnet20 with self.linear replaced by nn.Identity(), so
    its forward already returns the 256-d feature and its state_dict has the
    exact resnet20 keys MINUS linear.* -- which is precisely what
    --init-backbone-only expects (load_state_dict(strict=False) fills the
    backbone and leaves the classifier randomly initialised).

Checkpointing: one backbone per epoch, plus optional mid-epoch snapshots so a
fractional budget like E=7.5 lands on an exact checkpoint. Files are named
backbone_e<E>.pth.tar with E the cumulative full-set-equivalent epochs
(backbone_e6.pth.tar, backbone_e7.5.pth.tar, ...).
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm.auto import tqdm

from models.resnets import resnet20
from utils import save_model as _raw_save  # torch.save(state_dict) -- backbone only

from .data import _build, cycle_batches, get_spec, updates_for
from .seeding import loader_generator, set_all_seeds, worker_init_fn

BACKBONE_DIM = 256  # w*64 for resnet20x4


# --------------------------------------------------------------------------
# augmentation: two independent views of one image
# --------------------------------------------------------------------------
class TwoView:
    def __init__(self, spec):
        normalize = T.Normalize(np.array(spec.mean) / 255, np.array(spec.std) / 255)
        steps = []
        # A 1-channel source must become 3-channel FIRST: ColorJitter's
        # saturation/hue terms are undefined on an 'L' image, and the 3-channel
        # normalisation below would fail to broadcast onto [1, 32, 32].
        if spec.to_rgb:
            steps.append(T.Grayscale(num_output_channels=3))
        # RandomResizedCrop already outputs 32x32 from any input size, so
        # spec.resize is not needed here (STL-10 crops straight from 96x96).
        steps += [
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5) if spec.hflip else T.Lambda(lambda im: im),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # no Gaussian blur: pointless at 32x32
            T.ToTensor(),
            normalize,
        ]
        self.tf = T.Compose(steps)

    def __call__(self, x):
        return self.tf(x), self.tf(x)


def ssl_loader(batch_size, num_workers, seed, dataset="cifar10", data_dir=None):
    """Unlabelled two-view loader over the dataset's whole train split."""
    spec = get_spec(dataset)
    dset = _build(spec, data_dir or spec.dir, True, TwoView(spec))
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,  # SimSiam BN wants full batches
        generator=loader_generator(seed),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    loader._full_train_size = spec.train_size
    return loader


# --------------------------------------------------------------------------
# SimSiam model
# --------------------------------------------------------------------------
def _make_backbone(width):
    """resnet20 with the classifier removed -> 256-d feature extractor.

    Replacing linear with Identity means the backbone's state_dict carries the
    resnet20 keys WITHOUT linear.*, so it loads straight into a classifier-bearing
    resnet20 via strict=False.
    """
    net = resnet20(w=width, num_classes=10)
    net.linear = nn.Identity()
    return net


class SimSiam(nn.Module):
    def __init__(self, width=4, proj_dim=512, proj_hidden=512, pred_hidden=128):
        super().__init__()
        self.backbone = _make_backbone(width)

        # 3-layer projection MLP; final layer has BN and no ReLU (SimSiam).
        self.projection = nn.Sequential(
            nn.Linear(BACKBONE_DIM, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=True),
        )
        # 2-layer bottleneck prediction MLP; no BN/ReLU on the output.
        self.prediction = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

    def forward(self, x1, x2):
        z1 = self.projection(self.backbone(x1))
        z2 = self.projection(self.backbone(x2))
        p1 = self.prediction(z1)
        p2 = self.prediction(z2)
        return z1, z2, p1, p2


def _neg_cos(p, z):
    """Negative cosine similarity with stop-gradient on the target z."""
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(z1, z2, p1, p2):
    return 0.5 * _neg_cos(p1, z2) + 0.5 * _neg_cos(p2, z1)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def save_backbone(model, path):
    """Save only the backbone state_dict (resnet20 keys minus linear.*)."""
    _raw_save(model.backbone, path)


def _ckpt_name(frac_epoch):
    # backbone_e6.pth.tar / backbone_e7.5.pth.tar  (%g drops trailing zeros)
    return f"backbone_e{frac_epoch:g}.pth.tar"


def train_ssl(cfg):
    os.makedirs(cfg.run_dir, exist_ok=True)
    set_all_seeds(cfg.seed)
    device = cfg.device

    loader = ssl_loader(
        cfg.batch_size, cfg.num_workers, cfg.seed,
        dataset=getattr(cfg, "dataset", "cifar10"),
    )
    # Same epoch unit as the supervised stages: an epoch is `updates_per_epoch`
    # steps, not a pass over the data. This also removes the batch-size wrinkle
    # -- at batch 512 with drop_last a full pass is 97 updates, not 100.
    nb = updates_for(loader, getattr(cfg, "updates_per_epoch", 0))
    batches = cycle_batches(loader)

    model = SimSiam(width=cfg.model_width).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1)
    )

    # Which fractional budgets need a mid-epoch snapshot. For each target t in
    # (e-1, e) we save once the epoch has processed (t-(e-1)) of its batches.
    half_targets = sorted(set(cfg.half_checkpoints or []))
    for t in half_targets:
        if float(t).is_integer():
            raise ValueError(f"--half-checkpoint {t} is an integer; "
                             f"integer epochs are saved anyway")

    history = {"config": _cfg_dict(cfg), "epochs": [], "checkpoints": []}
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        # mid-epoch targets that fall inside this epoch: (epoch-1, epoch)
        mids = [t for t in half_targets if epoch - 1 < t < epoch]
        mid_batch = {round((t - (epoch - 1)) * nb): t for t in mids}

        pbar = tqdm(range(1, nb + 1), desc=f"ssl e{epoch}/{cfg.epochs}", leave=False)
        for bi in pbar:
            (x1, x2), _ = next(batches)
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad(set_to_none=True)
            z1, z2, p1, p2 = model(x1, x2)
            loss = simsiam_loss(z1, z2, p1, p2)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if bi in mid_batch:
                t = mid_batch[bi]
                path = os.path.join(cfg.run_dir, _ckpt_name(t))
                save_backbone(model, path)
                history["checkpoints"].append(
                    {"frac_epoch": t, "file": os.path.basename(path)}
                )
                print(f"    mid-epoch checkpoint at {t} -> {os.path.basename(path)}")

        scheduler.step()
        avg = running / max(nb, 1)

        path = os.path.join(cfg.run_dir, _ckpt_name(float(epoch)))
        save_backbone(model, path)
        history["checkpoints"].append(
            {"frac_epoch": float(epoch), "file": os.path.basename(path)}
        )
        history["epochs"].append(
            {
                "epoch": epoch,
                "ssl_loss": avg,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        with open(cfg.history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  epoch {epoch}/{cfg.epochs}  ssl_loss {avg:.4f}  -> {os.path.basename(path)}")

    history["total_seconds"] = round(time.time() - t0, 1)
    with open(cfg.history_path, "w") as f:
        json.dump(history, f, indent=2)
    return history


def _cfg_dict(cfg):
    from dataclasses import asdict
    return asdict(cfg)


def print_summary(history, every=None):
    eps = history["epochs"]
    if not eps:
        print("(no epochs)")
        return
    step = every or max(1, len(eps) // 20)
    print()
    print(f"{'epoch':>5} | {'ssl_loss':>9} | {'lr':>8} | {'elapsed':>8}")
    print("-" * 42)
    for e in eps:
        if e["epoch"] % step == 0 or e["epoch"] in (1, len(eps)):
            print(f"{e['epoch']:>5} | {e['ssl_loss']:>9.4f} | {e['lr']:>8.5f} | {e['elapsed_s']:>7.1f}s")
    print("-" * 42)
    print(f"checkpoints saved: {len(history['checkpoints'])}")
    if "total_seconds" in history:
        print(f"total: {history['total_seconds']:.1f}s")
