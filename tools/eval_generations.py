"""Score the offspring that a run actually SAVED, generation by generation.

WHY THIS EXISTS
---------------
Neither number a run already records is usable for comparing merge operators.

1. `run.json`'s `mutated_acc_mean` is what `finetune_merged_model` returned.
   On the default path the object it scored is a `ModelMerge` -- merged trunk
   plus two private branches -- and `evaluate_logits` reads `outputs[0]`. But
   `save_offspring` writes `head_models[0]`, a standalone resnet20 whose early
   layers never received the trunk's updates. The network scored is therefore
   NOT the network saved: this number compares a composite against a real
   network.

2. The per-generation CSV writes `Per Class` as an unquoted Python list, so its
   commas shift every later column (`Stop Node` holds per-class accuracies).
   And its `Joint` comes from `evaluate_fitness`, which sums accuracy over the
   agent's own classes and divides by the FULL class count -- a different
   quantity from the plain top-1 that `mutated_acc_mean` reports.

This tool loads each saved checkpoint into a bare resnet20 and evaluates it
directly, so every run is measured on the same object with the same metric.

    python tools/eval_generations.py runs_exp2d/seed0/permute --gens 1-25
    python tools/eval_generations.py runs_half/seed0/permute --out half.csv
"""

import argparse
import csv
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from models.resnets import resnet20  # noqa: E402
from sesil.data import get_full_loaders  # noqa: E402
from utils import decode_labels, reset_bn_stats  # noqa: E402


def parse_gens(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Plain top-1 over the whole test set, plus per-class accuracy."""
    model.eval()
    correct = total = 0
    per_c = torch.zeros(num_classes)
    tot_c = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for c in range(num_classes):
            m = y == c
            if m.any():
                tot_c[c] += m.sum().item()
                per_c[c] += (pred[m] == c).sum().item()
    per_c = (per_c / tot_c.clamp(min=1)).tolist()
    return correct / total, per_c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method_dir",
                   help="e.g. runs_exp2d/seed0/permute (the dir holding gen_*/)")
    p.add_argument("--gens", default="1-25")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reset-bn", action="store_true",
                   help="recompute BN statistics on the train set before scoring "
                        "(the saved checkpoint carries whatever stats it was left "
                        "with; off by default so the number reflects the file)")
    p.add_argument("--out", default=None, help="write a CSV of the per-model rows")
    args = p.parse_args(argv)

    train_loader, test_loader = get_full_loaders(
        args.dataset, batch_size=args.batch_size, num_workers=0, seed=0
    )
    num_classes = len(test_loader.dataset.classes) if hasattr(
        test_loader.dataset, "classes") else 10

    rows = []
    print(f"{'gen':>4} {'n':>3} {'top1 mean':>10} {'top1 min':>9} "
          f"{'top1 max':>9} {'known-cls':>10}")
    for gen in parse_gens(args.gens):
        gdir = os.path.join(args.method_dir, f"gen_{gen}")
        if not os.path.isdir(gdir):
            continue
        accs, knowns = [], []
        for hid in sorted(os.listdir(gdir)):
            ckpts = [f for f in os.listdir(os.path.join(gdir, hid))
                     if f.endswith(".pth.tar")]
            if not ckpts:
                continue
            # highest _v<N> is the newest
            ckpt = max(ckpts, key=lambda f: int(re.search(r"_v(\d+)", f).group(1)))
            sd = torch.load(os.path.join(gdir, hid, ckpt), map_location="cpu")
            sd = sd.get("state_dict", sd)
            model = resnet20(w=args.width, num_classes=num_classes)
            model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
            model = model.to(args.device)
            if args.reset_bn:
                reset_bn_stats(model, train_loader)
            acc, per_c = evaluate(model, test_loader, args.device, num_classes)

            # the classes this agent is nominally responsible for
            try:
                own = sorted({int(x) for x in decode_labels(hid).split("_")})
            except Exception:                                    # noqa: BLE001
                own = list(range(num_classes))
            known = sum(per_c[c] for c in own) / len(own)

            accs.append(acc)
            knowns.append(known)
            rows.append({"gen": gen, "hash": hid, "ckpt": ckpt, "top1": acc,
                         "known_class_acc": known, "n_own": len(own),
                         "per_class": per_c})
            del model
            torch.cuda.empty_cache()
        if accs:
            print(f"{gen:>4} {len(accs):>3} {sum(accs)/len(accs):>10.4f} "
                  f"{min(accs):>9.4f} {max(accs):>9.4f} "
                  f"{sum(knowns)/len(knowns):>10.4f}")

    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
