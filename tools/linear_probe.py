"""Linear-probe evaluation of SSL backbones (exp2 feature-quality evidence).

For each backbone_e<E>.pth.tar in an SSL run, freeze the backbone and train a
single linear head for 1 epoch on labelled CIFAR-10, then record test accuracy.
This gives a direct, cheap measure of how good each checkpoint's features are,
which is required to interpret the exp2a/b/c/ref results (E=3/6/7.5/50) -- a
weak evolution result at some E could be a weak-backbone problem or an
evolution problem, and the probe tells them apart.

Design notes:
  * backbone is frozen and kept in eval() so its BatchNorm running stats do not
    move; only linear.* gets gradients.
  * 1 epoch is a RELATIVE probe -- it under-reads the asymptotic linear-probe
    accuracy (which needs many epochs), but across e1..e50 the ordering and
    gaps are what matter here, and 1 epoch keeps it light enough to run
    alongside exp2b.
  * uses load_init_weights(..., backbone_only=True), the same loader exp2 uses,
    so a probe that loads is evidence the exp2 init path loads too.

    python tools/linear_probe.py --run runs_ssl/simsiam_e50_seed0
    python tools/linear_probe.py --run <dir> --epochs-list 1,3,6,7.5,10,20,30,50
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from models.resnets import resnet20  # noqa: E402
from sesil.data import get_cifar10_loaders  # noqa: E402
from sesil.population import load_init_weights  # noqa: E402

DEFAULT_EPOCHS = [1, 3, 6, 7.5, 10, 20, 30, 50]


def backbone_path(run_dir, E):
    return os.path.join(run_dir, f"backbone_e{E:g}.pth.tar")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, list):
            out = out[0]
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def probe_one(ckpt, train_loader, test_loader, cfg):
    device = cfg["device"]
    model = resnet20(w=4, num_classes=10).to(device)
    load_init_weights(model, ckpt, device, backbone_only=True)

    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("linear.")

    # eval() the whole model so backbone BN stays frozen; linear has no BN so
    # this does not affect head training.
    model.eval()
    opt = torch.optim.SGD(model.linear.parameters(), lr=cfg["lr"], momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for x, y in tqdm(train_loader, desc="probe", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if isinstance(logits, list):
            logits = logits[0]
        loss_fn(logits, y).backward()
        opt.step()

    return evaluate(model, test_loader, device)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SSL run dir with backbone_e*.pth.tar")
    ap.add_argument("--epochs-list", default=None,
                    help="comma list of E to probe (default 1,3,6,7.5,10,20,30,50)")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="output json (default <run>/probe.json)")
    args = ap.parse_args(argv)

    epochs = (
        [float(x) for x in args.epochs_list.split(",")]
        if args.epochs_list else list(DEFAULT_EPOCHS)
    )
    cfg = {"device": args.device, "lr": args.lr}
    out_path = args.out or os.path.join(args.run, "probe.json")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
    )

    results = {"run": args.run, "lr": args.lr, "probe_epochs": 1, "probes": []}
    print(f"{'E':>6} | {'probe_acc':>9} | {'file':>24} | {'time':>7}")
    print("-" * 56)
    t0 = time.time()
    for E in epochs:
        ckpt = backbone_path(args.run, E)
        if not os.path.exists(ckpt):
            print(f"{E:>6g} | {'MISSING':>9} | {os.path.basename(ckpt):>24} |")
            results["probes"].append({"E": E, "probe_acc": None, "status": "missing"})
            continue
        ts = time.time()
        acc = probe_one(ckpt, train_loader, test_loader, cfg)
        dt = time.time() - ts
        print(f"{E:>6g} | {acc:>9.4f} | {os.path.basename(ckpt):>24} | {dt:>6.1f}s")
        results["probes"].append({"E": E, "probe_acc": acc, "status": "ok"})
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    results["total_seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("-" * 56)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
