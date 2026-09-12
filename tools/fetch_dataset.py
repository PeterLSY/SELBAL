"""Download a registered dataset into the directory its spec points at.

    python tools/fetch_dataset.py cifar100
    python tools/fetch_dataset.py --list

Every dataset in sesil.data.DATASETS is fetchable this way; the training
entry points never download implicitly, so a typo in --dataset fails fast
instead of silently pulling a few hundred MB mid-run.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sesil.data import DATASETS, download_dataset, get_spec  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(prog="fetch_dataset.py")
    p.add_argument("dataset", nargs="?", choices=sorted(DATASETS))
    p.add_argument("--list", action="store_true", help="show the registry and exit")
    args = p.parse_args(argv)

    if args.list or not args.dataset:
        print(f"{'name':<10} {'classes':>7} {'train':>7}  dir")
        for name in sorted(DATASETS):
            s = DATASETS[name]
            print(f"{name:<10} {s.num_classes:>7} {s.train_size:>7}  {s.dir}")
        return 0

    spec = get_spec(args.dataset)
    print(f"Fetching {spec.name} into {spec.dir} ...")
    where = download_dataset(spec)
    print(f"Done: {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
