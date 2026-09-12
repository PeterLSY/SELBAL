"""Dataset registry and loaders.

Consolidates the three near-identical copies in the legacy tree
(train_cifar_models.py:51, evolutionary_wavg_training.py:685,
evolutionary_permute_training.py:691).

Every dataset the pipeline can run on is one `DatasetSpec` in `DATASETS`; the
name is what `--dataset` takes. Adding one is a single entry, provided the
images are 32x32 (resnet20's stride schedule assumes it) and the wrapper looks
like a torchvision classification dataset with a `.targets` list.

Two invariants a new entry has to respect:

  * `dir` must point at the SAME copy of the data that `utils.prepare_data`
    hands the evolution stage (via `evo_name` -> datasets/configs.py), and
    `mean`/`std` must match what datasets/cifar.py normalises with. The
    pretrain half and the evolution half feed the same weights: normalise them
    differently and the population silently degrades between stages.
  * `train_size` defines the full-set-equivalent epoch for that dataset -- it
    is the denominator every budget number on the x-axis is quoted against.
"""

from dataclasses import dataclass, replace
from typing import Callable, List, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from .seeding import loader_generator, worker_init_fn

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]


def _train_flag(cls):
    """Builder for torchvision datasets taking train=True/False (CIFAR, MNIST)."""
    def build(root, train, transform, download=False):
        return cls(root=root, train=train, download=download, transform=transform)
    build.cls = cls
    return build


def _split_kwarg(cls, train="train", test="test"):
    """Builder for datasets taking split='train'/'test' (SVHN, STL-10)."""
    def build(root, train_flag, transform, download=False):
        return cls(
            root=root, split=train if train_flag else test,
            download=download, transform=transform,
        )
    build.cls = cls
    return build


def targets_of(dataset):
    """Label list for a dataset, whatever the wrapper calls it.

    torchvision is not consistent: CIFAR/MNIST expose `.targets`, SVHN and
    STL-10 expose `.labels`. Both the subset filter here and the evolution
    stage's class-split loaders need one accessor.
    """
    for attr in ("targets", "labels"):
        t = getattr(dataset, attr, None)
        if t is not None:
            return [int(x) for x in t]
    raise AttributeError(
        f"{type(dataset).__name__} exposes neither .targets nor .labels"
    )


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    wrapper: Callable          # builder: (root, train, transform, download)
    dir: str
    num_classes: int
    train_size: int            # samples in the train split = one full-set epoch
    mean: List[float]          # 0-255 scale, 3 channels
    std: List[float]
    evo_name: str              # entry name in datasets/configs.py
    batch_size: int = 500
    # Input adaptation. resnet20 wants 32x32x3, so anything else is converted
    # here rather than in the model.
    resize: Optional[int] = None   # e.g. STL-10's 96x96 -> 32
    to_rgb: bool = False           # replicate a 1-channel image to 3
    # Horizontal flip is a label-preserving augmentation for objects but NOT
    # for digits/characters (a mirrored 2 is not a 2), so it is per dataset.
    hflip: bool = True


DATASETS = {
    "cifar10": DatasetSpec(
        name="cifar10",
        wrapper=_train_flag(torchvision.datasets.CIFAR10),
        dir="./data/cifar-10-python",
        num_classes=10,
        train_size=50000,
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
        evo_name="cifar10",
    ),
    # Same normalisation constants as CIFAR-10 on purpose: datasets/cifar.py
    # normalises with those globally, and matching it matters more than using
    # CIFAR-100's own statistics (which differ by well under one intensity
    # level anyway).
    #
    # dir is './data', NOT './data/cifar-100-python': torchvision appends the
    # 'cifar-100-python' component itself. (The directory of that name already
    # in the tree holds a copy of the CIFAR-10 batches -- a legacy misnaming,
    # see the note in the module docstring of the old loaders.)
    "cifar100": DatasetSpec(
        name="cifar100",
        wrapper=_train_flag(torchvision.datasets.CIFAR100),
        dir="./data",
        num_classes=100,
        train_size=50000,
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
        evo_name="cifar100",
    ),
    # 32x32 RGB house numbers: same shape as CIFAR-10 and the same 10-class
    # budget arithmetic, but a genuinely different image distribution -- the
    # natural second benchmark for the method. No hflip (mirrored digits).
    "svhn": DatasetSpec(
        name="svhn",
        wrapper=_split_kwarg(torchvision.datasets.SVHN),
        dir="./data/svhn",
        num_classes=10,
        train_size=73257,
        mean=[111.61, 113.16, 120.57],
        std=[50.50, 51.26, 50.24],
        evo_name="svhn",
        hflip=False,
    ),
    # 28x28 greyscale, replicated to 3 channels and padded/cropped to 32 by the
    # transform, so resnet20x4 is unchanged.
    "mnist": DatasetSpec(
        name="mnist",
        wrapper=_train_flag(torchvision.datasets.MNIST),
        dir="./data/mnist",
        num_classes=10,
        train_size=60000,
        mean=[33.32, 33.32, 33.32],
        std=[78.57, 78.57, 78.57],
        evo_name="mnist",
        resize=32,
        to_rgb=True,
        hflip=False,
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        wrapper=_train_flag(torchvision.datasets.FashionMNIST),
        dir="./data/fashion_mnist",
        num_classes=10,
        train_size=60000,
        mean=[73.15, 73.15, 73.15],
        std=[89.865, 89.865, 89.865],
        evo_name="fashion_mnist",
        resize=32,
        to_rgb=True,
    ),
    # 96x96 downsampled to 32. Only 5000 labelled train images, so a 3-class
    # subset is 1500 -- deliberately budget-starved, which is the regime the
    # SSL core is supposed to help most in. (STL-10 also ships 100k unlabelled
    # images; the SSL stage currently reads the labelled train split only.)
    "stl10": DatasetSpec(
        name="stl10",
        wrapper=_split_kwarg(torchvision.datasets.STL10),
        dir="./data/stl10",
        num_classes=10,
        train_size=5000,
        mean=[113.9, 112.2, 103.7],
        std=[66.4, 65.4, 69.4],
        evo_name="stl10",
        resize=32,
    ),
}

DEFAULT_DATASET = "cifar10"


def legacy_config(spec):
    """The dict shape `utils.prepare_data` expects for this dataset.

    `type: 'generic'` routes it to datasets/generic.py, which reads the spec
    back out via `spec_name`. CIFAR-10/100 are NOT built this way -- they keep
    their hand-written entries in datasets/configs.py and the original
    `type: 'cifar'` path, so existing runs are untouched.
    """
    return {
        "dir": spec.dir,
        "num_classes": spec.num_classes,
        "spec_name": spec.name,
        "batch_size": spec.batch_size,
        "type": "generic",
        "shuffle_train": True,
        "shuffle_test": False,
        "num_workers": 0,
    }


def register_legacy_configs():
    """Expose every non-CIFAR spec to the evolution stage.

    `utils.prepare_data` resolves a dataset by `getattr(datasets.configs, name)`,
    so the specs are injected as module attributes rather than duplicated by
    hand in datasets/configs.py. Existing attributes are never overwritten:
    cifar10/cifar100 keep the entries already written there.
    """
    import datasets.configs as config_module

    for spec in DATASETS.values():
        if not hasattr(config_module, spec.evo_name):
            setattr(config_module, spec.evo_name, legacy_config(spec))


def get_spec(dataset):
    """Resolve a --dataset name (or an already-resolved spec) to a DatasetSpec."""
    if isinstance(dataset, DatasetSpec):
        return dataset
    if dataset is None:
        dataset = DEFAULT_DATASET
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown --dataset {dataset!r}; choose from {sorted(DATASETS)}"
        ) from None


def _build(spec, root, train, transform):
    """Instantiate the dataset, turning 'not downloaded' into a usable message."""
    try:
        return spec.wrapper(root, train, transform, download=False)
    except (RuntimeError, FileNotFoundError) as e:
        raise RuntimeError(
            f"{spec.name} is not present under {root!r} ({e}). "
            f"Fetch it once with:  python tools/fetch_dataset.py {spec.name}"
        ) from None


def download_dataset(dataset):
    """Download a registered dataset into its configured directory."""
    import os

    spec = get_spec(dataset)
    os.makedirs(spec.dir, exist_ok=True)
    for train in (True, False):
        spec.wrapper(spec.dir, train, None, download=True)
    return spec.dir


# Back-compat: modules that predate the registry import these directly.
DATA_DIR = DATASETS[DEFAULT_DATASET].dir
FULL_TRAIN_SIZE = DATASETS[DEFAULT_DATASET].train_size


def cycle_batches(loader):
    """Endless iterator over `loader`, reshuffled on every wrap.

    An epoch of `updates_per_epoch` steps no longer coincides with a pass over
    the data, so the iterator has to survive the epoch boundary: restarting the
    loader every epoch would train on its first N batches and never see the rest.
    """
    while True:
        for batch in loader:
            yield batch


def updates_for(loader, updates_per_epoch, full_train_size=None):
    """How many updates make up ONE pass over `loader` under the epoch unit.

    `updates_per_epoch` counts updates per FULL-SET-equivalent epoch, so a
    loader over a class subset gets its proportional share (15000/50000 -> 0.3x
    the updates). That keeps the epoch axis update-fair across stages: every
    stage spends the same number of gradient steps per plotted epoch, whatever
    its dataset or batch size. 0 (or None) = legacy behaviour, one true pass.

    `full_train_size` defaults to the size the loader's own dataset was built
    from when known (`_full_train_size`, stamped on by the loaders below), so a
    subset loader scales against ITS dataset's full split rather than
    CIFAR-10's.
    """
    if not updates_per_epoch:
        return len(loader)
    if full_train_size is None:
        full_train_size = getattr(loader, "_full_train_size", FULL_TRAIN_SIZE)
    frac = len(loader.dataset) / full_train_size
    return max(1, round(updates_per_epoch * frac))


def _adapt(spec):
    """Leading transforms that bring any input to 32x32x3."""
    steps = []
    if spec.resize:
        steps.append(T.Resize(spec.resize))
    if spec.to_rgb:
        steps.append(T.Grayscale(num_output_channels=3))
    return steps


def _transforms(spec):
    normalize = T.Normalize(np.array(spec.mean) / 255, np.array(spec.std) / 255)
    train_steps = _adapt(spec)
    if spec.hflip:
        train_steps.append(T.RandomHorizontalFlip())
    train_steps += [T.RandomCrop(32, padding=4), T.ToTensor(), normalize]
    test_tf = T.Compose(_adapt(spec) + [T.ToTensor(), normalize])
    return T.Compose(train_steps), test_tf


class SubsetKeepLabels(torch.utils.data.Dataset):
    """Keep only samples whose label is in `valid_classes`, labels UNCHANGED.

    Labels are deliberately NOT remapped to 0..k-1. Verified against the
    reference population: those checkpoints have a 10-wide `linear` layer and
    predict in the original CIFAR-10 label space (a 3-class expert on {0,7,8}
    scores 0.930 against original labels and 0.321 against remapped ones). The
    merge stage also requires every expert to share one output space -- which is
    why every model carries a head as wide as the dataset's class count.
    """

    def __init__(self, dataset, valid_classes):
        self.dataset = dataset
        self.valid_classes = set(int(c) for c in valid_classes)
        self.indices = [
            i for i, t in enumerate(targets_of(dataset)) if t in self.valid_classes
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def _stamp(loader, spec):
    """Record the dataset's full train size on the loader for `updates_for`."""
    loader._full_train_size = spec.train_size
    return loader


def get_full_loaders(
    dataset=DEFAULT_DATASET, batch_size=500, num_workers=0, seed=0, data_dir=None
):
    """Full train/test loaders over every class (used by the mutation stage)."""
    spec = get_spec(dataset)
    train_tf, test_tf = _transforms(spec)
    root = data_dir or spec.dir
    train_dset = _build(spec, root, True, train_tf)
    test_dset = _build(spec, root, False, test_tf)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator(seed),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return _stamp(train_loader, spec), _stamp(test_loader, spec)


def get_cifar10_loaders(batch_size=500, num_workers=0, seed=0, data_dir=DATA_DIR):
    """Back-compat alias for the CIFAR-10 full loaders."""
    return get_full_loaders(
        "cifar10", batch_size=batch_size, num_workers=num_workers,
        seed=seed, data_dir=data_dir,
    )


def get_subset_loaders(
    classes, dataset=DEFAULT_DATASET, batch_size=500, num_workers=0, seed=0,
    data_dir=None
):
    """Train loader restricted to `classes`; test loader over the same classes.

    Both keep the dataset's original label values.
    """
    spec = get_spec(dataset)
    train_tf, test_tf = _transforms(spec)
    root = data_dir or spec.dir
    base_train = _build(spec, root, True, train_tf)
    base_test = _build(spec, root, False, test_tf)
    train_set = SubsetKeepLabels(base_train, classes)
    test_set = SubsetKeepLabels(base_test, classes)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator(seed),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return _stamp(train_loader, spec), _stamp(test_loader, spec)
