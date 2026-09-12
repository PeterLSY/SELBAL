"""Stage 0 -- pretrain the initial population.

Trains one resnet20x4 expert per class assignment, reproducing the format of
`./checkpoints/cifar10_evolution/initial/` exactly:

  * 10-wide classifier (`linear.weight` is (10, 256)), NOT 3-wide
  * original CIFAR-10 label values, NOT remapped to 0..2
  * raw state_dict saved as `<hash>/resnet20x4_v0.pth.tar`, no wrapper dict
  * folder named by `utils.encode_labels` (md5 of "8_7_0", first 8 hex chars)

Verified against the reference checkpoints before writing this module: a
reference expert on classes [8,7,0] scores 0.9303 against original labels and
0.3207 against 0..2-remapped labels, and its prediction histogram over the 10
logits is [1114,0,0,0,0,0,0,892,994,0] -- only the three trained classes are
ever predicted.

The LR schedule deliberately differs from train_cifar_models.py:85-87, where
`LinearLR(total_iters=len(train_loader))` is stepped once per batch, decaying to
1e-7 within the first epoch and staying there. Here: SGD + cosine over the full
run, stepped per epoch.
"""

import os
import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from models.resnets import resnet20
from utils import encode_labels, save_model

from .data import cycle_batches, get_subset_loaders, updates_for
from .mapping_guard import assignments_from_dir, mapping_guard
from .seeding import derive, set_all_seeds

NUM_CLASSES = 10  # CIFAR-10 default; per-run width comes from cfg.num_classes


def fallback_assignments(seed, n_models=10, k=3, mapping_file="mapping.json",
                         num_classes=NUM_CLASSES):
    """Generate `n_models` random k-class subsets when the reference is absent."""
    import random

    if k > num_classes:
        raise ValueError(
            f"classes_per_agent={k} exceeds the dataset's {num_classes} classes"
        )
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n_models:
        subset = tuple(rng.sample(range(num_classes), k))
        if frozenset(subset) in seen:
            continue
        seen.add(frozenset(subset))
        hash_id = encode_labels(list(subset), mapping_file=mapping_file)
        out.append((hash_id, list(subset)))
    return out


def resolve_assignments(cfg):
    """The class assignments to train, reusing the reference population if any.

    The reference population only exists for CIFAR-10 (10 agents x 3 classes,
    fixed so every run is comparable). Any other dataset draws its own
    assignments from cfg.n_agents / cfg.classes_per_agent at cfg.seed.
    """
    ref = cfg.reference_population
    if os.path.isdir(ref):
        pairs = assignments_from_dir(ref, mapping_file=cfg.mapping_file)
        print(f"[pretrain] reusing {len(pairs)} assignments from {ref}")
        return pairs
    print(
        f"[pretrain] {ref} not found -- drawing {cfg.n_agents} random "
        f"{cfg.classes_per_agent}-class subsets from {cfg.num_classes} classes "
        f"(seed={cfg.seed}). These will NOT be comparable to existing results."
    )
    return fallback_assignments(
        cfg.seed,
        n_models=cfg.n_agents,
        k=cfg.classes_per_agent,
        mapping_file=cfg.mapping_file,
        num_classes=cfg.num_classes,
    )


@torch.no_grad()
def subset_accuracy(model, test_loader, device):
    """Top-1 over the 10 logits, scored against original labels."""
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        if isinstance(pred, list):
            pred = pred[0]
        pred = pred.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def load_init_weights(model, init_from, device, backbone_only=None,
                      num_classes=NUM_CLASSES):
    """Initialise an expert from a shared checkpoint (exp2 entry point).

    Two kinds of checkpoint are accepted:

    * Full-width model (joint baseline): loaded strictly. The head must span the
      dataset's whole label space so the expert stays in original label
      coordinates -- the invariant the whole pipeline depends on.
    * Backbone only (SimSiam SSL): has no linear.* keys. Loaded with
      strict=False so the backbone fills in and the classifier stays randomly
      initialised, ready to learn the 3-class subset from scratch.

    backbone_only: None auto-detects (backbone iff no linear.weight); True/False
    force the choice. Forcing True on a full checkpoint ignores its head.
    """
    sd = torch.load(init_from, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    for k in list(sd.keys()):
        if k.startswith("module."):
            sd[k.replace("module.", "")] = sd.pop(k)

    has_head = "linear.weight" in sd
    if backbone_only is None:
        backbone_only = not has_head

    if backbone_only:
        # drop any classifier weights and load the rest non-strictly
        sd = {k: v for k, v in sd.items() if not k.startswith("linear.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        stray = [k for k in unexpected]
        if stray:
            raise ValueError(
                f"--init-from backbone checkpoint {init_from} has unexpected "
                f"keys not in resnet20: {stray[:5]}"
            )
        non_linear_missing = [k for k in missing if not k.startswith("linear.")]
        if non_linear_missing:
            raise ValueError(
                f"--init-from backbone checkpoint {init_from} is missing backbone "
                f"keys: {non_linear_missing[:5]}"
            )
        return model

    if not has_head:
        raise ValueError(
            f"--init-from checkpoint {init_from} has no linear.* head but "
            f"backbone_only was forced False; it cannot be loaded strictly."
        )
    out_features = sd["linear.weight"].shape[0]
    if out_features != num_classes:
        raise ValueError(
            f"--init-from checkpoint {init_from} has a {out_features}-wide head; "
            f"this dataset requires {num_classes} (original label space)."
        )
    model.load_state_dict(sd)
    return model


@torch.no_grad()
def calibrate_closed_form(model, train_loader, classes, device, ridge=1e-2,
                          num_classes=NUM_CLASSES):
    """Fill classifier rows for `classes` by ridge least-squares on frozen
    backbone features. ZERO SGD (exp2e).

    W* = argmin ||X_aug W - T||^2 + ridge||W||^2, solved in closed form, where
    X_aug is [feature, 1] (bias column) and T is one-hot over all `num_classes`.
    The backbone is untouched.
    """
    model.eval()
    lin = model.linear
    model.linear = nn.Identity()  # expose the 256-d backbone feature
    feats, labels = [], []
    for x, y in train_loader:
        f = model(x.to(device))
        if isinstance(f, list):
            f = f[0]
        feats.append(f.float().cpu())
        labels.append(y)
    model.linear = lin

    X = torch.cat(feats)                       # [N, D]
    Y = torch.cat(labels)                      # [N]
    N, D = X.shape
    Xa = torch.cat([X, torch.ones(N, 1)], dim=1)   # [N, D+1] bias column
    T = torch.zeros(N, num_classes)
    T[torch.arange(N), Y] = 1.0

    A = Xa.T @ Xa + ridge * torch.eye(D + 1)
    W = torch.linalg.solve(A, Xa.T @ T)        # [D+1, num_classes]

    # Fill ALL rows, not just the subset's. Non-subset columns have all-zero
    # targets, so ridge drives their rows to ~0 -- essential, because leaving
    # them at random init lets those large random rows win the argmax over the
    # modestly-scaled closed-form subset rows (observed: subset acc 0.003 when
    # only 3 rows were filled). With all rows set, non-subset logits ~0 and the
    # subset rows decide the prediction.
    w_dev = W.to(device)
    lin.weight.copy_(w_dev[:D].T)   # [num_classes, D]
    lin.bias.copy_(w_dev[D])        # [num_classes]
    return model


def train_one(classes, cfg, member_seed, desc=""):
    """Train a single k-class expert. Returns (model, epochs_used, final_acc)."""
    device = cfg.device
    num_classes = cfg.num_classes
    set_all_seeds(member_seed)

    train_loader, test_loader = get_subset_loaders(
        classes,
        dataset=cfg.dataset,
        batch_size=cfg.pretrain_batch_size,
        num_workers=cfg.pretrain_num_workers,
        seed=member_seed,
    )

    model = resnet20(w=cfg.model_width, num_classes=num_classes).to(device)
    lr = cfg.pretrain_lr
    if cfg.init_from:
        model = load_init_weights(
            model, cfg.init_from, device, backbone_only=cfg.init_backbone_only,
            num_classes=num_classes,
        )
        lr = cfg.init_lr

    # Closed-form calibration (exp2e): fill the head by ridge least-squares,
    # zero SGD. Precedence over sub-epochs; the strict S->0 ledger endpoint.
    if cfg.calibrate == "closed_form":
        model = calibrate_closed_form(
            model, train_loader, classes, device, ridge=cfg.calibrate_ridge,
            num_classes=num_classes,
        )
        acc = subset_accuracy(model, test_loader, device)
        print(f"    closed-form calibration (0 SGD steps), acc {acc:.4f}")
        return model, 0, acc

    # Fractional-epoch finetune (exp2, S<integer) runs a fixed batch count and
    # bypasses the integer-epoch loop entirely, so exp1/exp1b are untouched.
    if cfg.finetune_sub_epochs is not None:
        return _finetune_sub_epochs(model, train_loader, test_loader, cfg, lr)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg.pretrain_momentum,
        weight_decay=cfg.pretrain_weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.pretrain_epochs, 1)
    )
    loss_fn = nn.CrossEntropyLoss()

    best_acc, epochs_used = 0.0, 0

    # A shared init may already clear the bar on this subset, in which case the
    # expert needs no training at all. Checking up front avoids spending an
    # epoch just to discover that.
    if cfg.init_from and cfg.target_acc is not None:
        acc0 = subset_accuracy(model, test_loader, device)
        print(f"    init acc {acc0:.4f} (target {cfg.target_acc})")
        if acc0 >= cfg.target_acc:
            print("    already at target -- 0 epochs")
            return model, 0, acc0

    nb = updates_for(train_loader, cfg.updates_per_epoch)
    batches = cycle_batches(train_loader)

    pbar = tqdm(range(cfg.pretrain_epochs), desc=desc, leave=False)
    for epoch in pbar:
        model.train()
        for _ in range(nb):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if isinstance(logits, list):
                logits = logits[0]
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()          # per epoch, over the full training length
        epochs_used = epoch + 1

        acc = subset_accuracy(model, test_loader, device)
        best_acc = max(best_acc, acc)
        pbar.set_postfix(acc=f"{acc:.4f}", lr=f"{scheduler.get_last_lr()[0]:.4f}")

        if cfg.target_acc is not None and acc >= cfg.target_acc:
            print(
                f"    early stop at epoch {epochs_used}: acc {acc:.4f} "
                f">= target {cfg.target_acc}"
            )
            break

    final_acc = subset_accuracy(model, test_loader, device)
    return model, epochs_used, final_acc


def _finetune_sub_epochs(model, train_loader, test_loader, cfg, lr):
    """Finetune for round(S * batches_per_epoch) batches (S = sub-epochs).

    Used only on the --init-from path with a fractional S (exp2). LR follows a
    per-batch cosine over the truncated length; with S=1.0 this is one subset
    epoch. Returns (model, sub_epochs, final_acc) -- epochs_used is reported as
    the fractional S so the caller's accounting stays in sub-epoch units.

    batches_per_epoch is the subset's share of --updates-per-epoch (0.3x, since
    a 3-class subset is 15000/50000 of the full set), so the y = 3S ledger holds
    under any epoch definition. With updates_per_epoch=0 it is len(train_loader)
    as before.
    """
    S = cfg.finetune_sub_epochs
    nb = updates_for(train_loader, cfg.updates_per_epoch)
    total = max(1, round(S * nb))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=cfg.pretrain_momentum,
        weight_decay=cfg.pretrain_weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    batches = cycle_batches(train_loader)
    for _ in range(total):
        x, y = next(batches)
        x, y = x.to(cfg.device), y.to(cfg.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if isinstance(logits, list):
            logits = logits[0]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    final_acc = subset_accuracy(model, test_loader, cfg.device)
    print(f"    sub-epoch finetune: {total} batches (S={S}), acc {final_acc:.4f}")
    return model, S, final_acc


def pretrain_population(cfg, paths):
    """Train (or reuse) the full initial population. Returns a summary list."""
    os.makedirs(paths.population_dir, exist_ok=True)

    with mapping_guard(cfg.mapping_file) as _:
        assignments = resolve_assignments(cfg)

    summary = []
    t_start = time.time()
    for i, (hash_id, classes) in enumerate(assignments, 1):
        name = "_".join(str(c) for c in classes)
        ckpt = paths.member_ckpt(hash_id)

        if os.path.exists(ckpt) and not cfg.force_pretrain:
            print(f"[{i}/{len(assignments)}] {name} ({hash_id}) -- exists, skipping")
            summary.append(
                {
                    "model": name,
                    "hash": hash_id,
                    "epochs": 0,
                    "acc": float("nan"),
                    "status": "reused",
                }
            )
            continue

        print(f"[{i}/{len(assignments)}] training {name} ({hash_id}) ...")
        member_seed = derive(cfg.seed, hash_id)
        model, epochs_used, acc = train_one(
            classes, cfg, member_seed, desc=f"{name}"
        )

        # Confirm the hash we are about to write under really is this label list.
        with mapping_guard(cfg.mapping_file):
            recomputed = encode_labels(classes, mapping_file=cfg.mapping_file)
        if recomputed != hash_id:
            raise RuntimeError(
                f"hash mismatch for {classes}: folder {hash_id} vs "
                f"encode_labels {recomputed}"
            )

        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        save_model(model, ckpt)     # utils.save_model -> raw state_dict
        print(f"    saved {ckpt}  (epochs={epochs_used}, acc={acc:.4f})")

        summary.append(
            {
                "model": name,
                "hash": hash_id,
                "epochs": epochs_used,
                "acc": acc,
                "status": "trained",
            }
        )
        del model
        torch.cuda.empty_cache()

    print(f"\n[pretrain] finished in {time.time() - t_start:.1f}s")
    return summary


def print_summary(summary):
    print()
    print(f"{'Model':>12} | {'Hash':>10} | {'Epochs':>6} | {'3-class test acc':>16} | Status")
    print("-" * 68)
    for row in summary:
        acc = "-" if row["acc"] != row["acc"] else f"{row['acc']:.4f}"
        print(
            f"{row['model']:>12} | {row['hash']:>10} | {row['epochs']:>6} | "
            f"{acc:>16} | {row['status']}"
        )
    trained = [r["acc"] for r in summary if r["status"] == "trained"]
    if trained:
        print("-" * 68)
        print(f"{'mean':>12} | {'':>10} | {'':>6} | {sum(trained)/len(trained):>16.4f} |")
