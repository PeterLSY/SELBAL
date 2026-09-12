"""Evolution-stage loaders for any dataset in sesil.data.DATASETS.

datasets/cifar.py is hardwired to the CIFAR shape: `train=True/False` in the
constructor, `.targets` for labels, `.classes` for names, and one global
normalisation. That covers CIFAR-10/100 and nothing else, so this module is the
same contract driven entirely by the DatasetSpec that the pretrain half already
uses -- one registry, both halves.

It is selected by `type: 'generic'` in the config dict (see
`DatasetSpec.legacy_config`), which `utils.prepare_data` dispatches on. The
CIFAR entries keep `type: 'cifar'` and the original code path, so nothing about
the existing runs changes.
"""

import numpy as np
import torch
import torchvision.transforms as T

from sesil.data import get_spec, targets_of


def _transforms(spec):
    """Same adaptation the pretrain half applies: whatever -> 32x32x3."""
    lead = []
    if spec.resize:
        lead.append(T.Resize(spec.resize))
    if spec.to_rgb:
        lead.append(T.Grayscale(num_output_channels=3))
    normalize = T.Normalize(np.array(spec.mean) / 255, np.array(spec.std) / 255)

    train_steps = list(lead)
    if spec.hflip:
        train_steps.append(T.RandomHorizontalFlip())
    train_steps += [T.RandomCrop(32, padding=4), T.ToTensor(), normalize]
    return T.Compose(train_steps), T.Compose(lead + [T.ToTensor(), normalize])


def _class_names(dset, spec):
    """`.classes` if the wrapper has it, else plain indices as strings.

    Only the LENGTH matters downstream (utils.prepare_experiment_config sizes
    the head with it, evaluation counts classes with it); CLIP-head evaluation
    would want real names, but this pipeline is logits-only.
    """
    names = getattr(dset, "classes", None)
    if names is not None and len(names) == spec.num_classes:
        return list(names)
    return [str(i) for i in range(spec.num_classes)]


def _split_loaders(dset, config, shuffle):
    """One loader per entry in config['class_splits'], plus the remapping."""
    loaders, grouped = [], np.zeros(config["num_classes"], dtype=int)
    labels = targets_of(dset)
    for splits in config["class_splits"]:
        wanted = set(int(c) for c in splits)
        idx = [i for i, lab in enumerate(labels) if lab in wanted]
        loaders.append(
            torch.utils.data.DataLoader(
                torch.utils.data.Subset(dset, idx),
                batch_size=config["batch_size"],
                shuffle=shuffle,
                num_workers=config["num_workers"],
            )
        )
        grouped[list(splits)] = np.arange(len(splits))
    return loaders, torch.from_numpy(grouped)


def prepare_train_loaders(config):
    spec = get_spec(config["spec_name"])
    train_tf, test_tf = _transforms(spec)
    tf = test_tf if "no_transform" in config else train_tf
    dset = spec.wrapper(config["dir"], True, tf, download=False)

    loaders = {
        "full": torch.utils.data.DataLoader(
            dset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle_train"],
            num_workers=config["num_workers"],
        )
    }
    if "class_splits" in config:
        splits, remapping = _split_loaders(dset, config, config["shuffle_train"])
        loaders["splits"] = splits
        loaders["label_remapping"] = remapping
        loaders["class_splits"] = config["class_splits"]
    return loaders


def prepare_test_loaders(config):
    spec = get_spec(config["spec_name"])
    _, test_tf = _transforms(spec)
    dset = spec.wrapper(config["dir"], False, test_tf, download=False)

    loaders = {
        "full": torch.utils.data.DataLoader(
            dset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle_test"],
            num_workers=config["num_workers"],
        )
    }
    if "class_splits" in config:
        loaders["splits"], _ = _split_loaders(dset, config, False)
    loaders["class_names"] = _class_names(dset, spec)
    return loaders
