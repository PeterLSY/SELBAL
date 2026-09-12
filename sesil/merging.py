"""Stage 3 -- merge mated pairs into offspring.

Carried over from evolutionary_wavg_training.py:518-674. This is the plugin
seam: `merging_fn` is a name resolved through `utils.get_merging_fn`, so wavg /
permute / zipit differ only in that one string (see sesil/config.py).

The aggressive VRAM cleanup is taken from the permute variant
(evolutionary_permute_training.py:637-676), which had it and wavg did not --
same results, lower peak memory.
"""

import gc
from copy import deepcopy

import torch
from tqdm.auto import tqdm

from model_merger import ModelMerge
from utils import (
    CONCEPT_TASKS,
    decode_labels,
    flatten_nested_dict,
    get_merging_fn,
    inject_pair,
    prepare_experiment_config,
    reset_bn_stats,
    write_to_csv,
)

from .evaluation import evaluate_model, inject_model, sort_model_name_unique


def build_unique_key(labels, seen_keys):
    """Short, unique key from labels via rotation, then an index suffix."""
    sorted_labels = sorted(set(labels))
    base_key = "_".join(sorted_labels)

    key = base_key
    rotation = 0
    while key in seen_keys:
        rotation += 1
        rotated = (
            sorted_labels[rotation % len(sorted_labels):]
            + sorted_labels[: rotation % len(sorted_labels)]
        )
        key = "_".join(rotated)

        if rotation >= len(sorted_labels):
            suffix = 1
            while f"{base_key}_{suffix}" in seen_keys:
                suffix += 1
            key = f"{base_key}_{suffix}"
            break

    seen_keys.add(key)
    return key


def run_merging(merging_fn, node_config, experiment_config, pairs, loners, device, csv_file):
    """Merge every pair; pass loners through untouched. Returns {key: model}."""
    offsprings = {}
    seen_keys = set()

    # ---------------- pairs ----------------
    for pair in tqdm(pairs, desc="Evaluating Pairs..."):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(experiment_config)

        train_loader = config["data"]["train"]["full"]
        base_models = [
            reset_bn_stats(base_model, train_loader)
            for base_model in config["models"]["bases"]
        ]

        config["node"] = node_config
        Grapher = config["graph"]
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]

        Merge = ModelMerge(*graphs, device=device)

        Merge.transform(
            deepcopy(config["models"]["new"]),
            train_loader,
            transform_fn=get_merging_fn(merging_fn),
            metric_classes=config["metric_fns"],
            stop_at=node_config["stop_node"],
            **node_config["params"],
        )

        reset_bn_stats(Merge, train_loader)

        results = evaluate_model(experiment_config["eval_type"], Merge, config)
        for idx, split in enumerate(pair):
            results[f"Split {CONCEPT_TASKS[idx]}"] = sort_model_name_unique(
                decode_labels(split)
            )
        results["Time"] = Merge.compute_transform_time
        results["Merging Fn"] = merging_fn
        results["Model Name"] = config["model"]["name"]
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)
        print(results)

        pair = tuple(decode_labels(p) for p in pair)
        labels = "_".join(pair).split("_")
        merged_key = build_unique_key(labels, seen_keys)

        for g_ in Merge.graphs:
            g_.clear_hooks()        # stop re-capturing activations on every forward
            g_.intermediates = {}   # free ~1-2 GB of stored batch activations
        Merge.metrics = None        # free per-node covariance/mean metrics
        Merge.graphs = []           # drop graph wrappers (weights live in head_models)

        offsprings[merged_key] = Merge.cpu()

        del base_models, graphs, Merge, config
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------- loners ----------------
    for loner in tqdm(loners, desc="Evaluating Loners..."):
        experiment_config = inject_model(experiment_config, loner)
        config = prepare_experiment_config(experiment_config)

        train_loader = config["data"]["train"]["full"]
        base_model = reset_bn_stats(config["models"]["bases"][0], train_loader)
        config["node"] = node_config

        results = evaluate_model(experiment_config["eval_type"], base_model, config)
        results["Split"] = sort_model_name_unique(decode_labels(loner))
        results["Time"] = 0.0
        results["Merging Fn"] = "None"
        results["Model Name"] = config["model"]["name"]
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)
        print(results)

        labels = decode_labels(loner).split("_")
        loner_key = build_unique_key(labels, seen_keys)
        offsprings[loner_key] = base_model.cpu()

        del config
        gc.collect()
        torch.cuda.empty_cache()

    return offsprings
