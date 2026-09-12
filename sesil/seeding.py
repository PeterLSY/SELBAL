"""Seed control.

The legacy scripts set the three global seeds at import time with a literal 0
(evolutionary_wavg_training.py:16-18), so the seed could not be varied without
editing the file. Here it is a parameter.

Caveat, stated plainly: `set_all_seeds` covers weight init, dropout, and the
mating draws in `probabilistic_choice`. It does NOT by itself make a
multi-worker DataLoader deterministic. Loaders this package builds (pretraining)
get an explicit generator + worker_init_fn; loaders built inside
`utils.prepare_data` do not, because that would mean editing utils.py. The
cifar10 dataset config currently uses num_workers=0 (datasets/configs.py:42),
which sidesteps the issue -- if that is ever raised above 0, evolution-stage
shuffling becomes only approximately reproducible.
"""

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = False):
    """Seed python, numpy and torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loader_generator(seed: int) -> torch.Generator:
    """Generator for a DataLoader so shuffling is tied to the run seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int):
    """Give every dataloader worker a distinct, derived seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def derive(seed: int, *parts) -> int:
    """Derive a stable sub-seed, e.g. one per population member.

    Uses a fixed hash so it does not depend on PYTHONHASHSEED.
    """
    h = seed
    for p in parts:
        s = str(p).encode()
        acc = 0
        for b in s:
            acc = (acc * 131 + b) % (2**31 - 1)
        h = (h * 1000003 + acc) % (2**31 - 1)
    return h
