"""Stage 1 -- evaluate the current population (and shared eval helpers).

Logic carried over verbatim from evolutionary_wavg_training.py:22-200 and
:381-515. The only changes: `prepare_experiment_config` is now called on the
config that was passed in rather than on a module-level global `raw_config`
(wavg:170 and :600 -- equivalent, because the caller passed that same object
and `inject_model`/`inject_pair` mutate it in place), and the CSV path is
supplied by the caller.
"""

import os
from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm

from utils import (
    CONCEPT_TASKS,
    autocast,
    decode_labels,
    flatten_nested_dict,
    get_device,
    prepare_experiment_config,
    reset_bn_stats,
    split_str_to_ints,
    write_to_csv,
)


def sort_model_name_unique(name: str) -> str:
    nums = [int(x) for x in name.split("_")]
    nums = sorted(set(nums))
    return "_".join(map(str, nums))


def inject_model(config, model, ignore_bases=False):
    model_name = config["model"]["name"]
    config["dataset"]["class_splits"] = [split_str_to_ints(decode_labels(model))]
    if not ignore_bases:
        config["model"]["bases"] = [
            os.path.join(config["model"]["dir"], model, f"{model_name}_v0.pth.tar")
        ]
    return config


def evaluate_fitness(model_id, model, config, num_classes=None):
    """Per-class accuracy over the classes this model was trained on.

    Note the label space: predictions are compared against ORIGINAL dataset
    labels, which is why the population must carry a full-width head.
    `num_classes` defaults to the loaded dataset's class list.
    """
    device = next(model.parameters()).device
    test_loader = config["data"]["test"]["full"]

    if num_classes is None:
        num_classes = len(config["data"]["test"]["class_names"])
    acc_per_class = [0.0] * num_classes

    subset_labels = sorted(set(int(x) for x in model_id.split("_")))
    print("Consider subset labels: ", subset_labels)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            mask = torch.zeros_like(labels, dtype=torch.bool)
            for lab in subset_labels:
                mask |= labels == lab
            if mask.sum() == 0:
                continue

            images_masked = images[mask]
            labels_masked = labels[mask]

            outputs = model(images_masked)
            _, preds = outputs.max(1)

            for lab in subset_labels:
                lab_mask = labels_masked == lab
                if lab_mask.sum() > 0:
                    acc_per_class[lab] += (
                        (preds[lab_mask] == labels_masked[lab_mask]).sum().item()
                    )

    total_per_class = torch.zeros(num_classes)
    for images, labels in test_loader:
        for lab in subset_labels:
            total_per_class[lab] += (labels == lab).sum().item()

    for lab in subset_labels:
        if total_per_class[lab] > 0:
            acc_per_class[lab] /= total_per_class[lab]

    joint_acc = sum(acc_per_class[lab] for lab in subset_labels) / len(subset_labels)

    return {
        "Joint": sum(acc_per_class) / len(acc_per_class),
        "Per Task Avg": joint_acc,
        "Model Name": model_id,
        "Per Class": acc_per_class.copy(),
    }


def run_evolutionary_evaluation(
    node_config, experiment_config, model_ids, device, csv_file, num_classes=None
):
    population_info = []
    for model_id in model_ids:
        print(model_id)
        print(decode_labels(model_id))

        experiment_config = inject_model(experiment_config, model_id)
        config = prepare_experiment_config(experiment_config)

        train_loader = config["data"]["train"]["full"]
        base_model = [
            reset_bn_stats(base_model, train_loader)
            for base_model in config["models"]["bases"]
        ]
        config["node"] = node_config

        reset_bn_stats(base_model[0], train_loader)

        results = evaluate_fitness(
            decode_labels(model_id), base_model[0], config, num_classes=num_classes
        )
        print(results)
        results["Model Name"] = sort_model_name_unique(decode_labels(model_id))
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)
        results["Model Name"] = model_id
        population_info.append(results)

    return population_info


# --------------------------------------------------------------------------
# Merged-model evaluation (multi-head aware)
# --------------------------------------------------------------------------
def evaluate_logits_alltasks(model, loader, splits, num_classes):
    model.eval()
    correct = 0
    total = 0

    splits = [list(split) for split in splits]
    print("check splits: ", splits)
    totals = [0] * num_classes
    corrects = [0] * num_classes

    device = get_device(model)

    all_splits = torch.tensor(
        [cls for split in splits for cls in split], device=device, dtype=torch.long
    )
    print(all_splits)

    task_map = {}
    for i, split in enumerate(splits):
        for _cls in split:
            task_map[_cls] = i

    task_map = [task_map.get(_cls, -1) for _cls in range(num_classes)]
    task_map = torch.tensor(task_map, device=device, dtype=torch.long)

    splits_tensor = [torch.tensor(s, device=device, dtype=torch.long) for s in splits]

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, "Evaluating multihead head model"):
            inputs, labels = inputs.to(device), labels.to(device)

            class_selector = torch.isin(labels, all_splits)
            inputs, labels = inputs[class_selector], labels[class_selector]

            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue

            task_idx = task_map[labels]
            outputs = model(inputs)

            if isinstance(outputs, list):
                for i, split in enumerate(splits_tensor):
                    exclude_labels = torch.tensor(
                        [c for c in all_splits.tolist() if c not in split.tolist()],
                        device=device,
                        dtype=torch.long,
                    )
                    if exclude_labels.numel() > 0:
                        outputs[i][:, exclude_labels] = -torch.inf
                outputs = torch.stack(outputs, dim=1)
                outputs2 = outputs.softmax(dim=-1).to(outputs.dtype).max(dim=-2)[0]
                outputs2[:, all_splits] += 2
                outputs = outputs[range(batch_size), task_idx, :]
            else:
                outputs2 = outputs.clone()
                for split in splits_tensor:
                    outputs2[:, split] = (
                        torch.softmax(outputs2[:, split], dim=-1).to(outputs.dtype) + 2
                    )
            outputs2 = outputs2.argmax(dim=-1)

            preds = []
            for out_vec, split in zip(
                outputs, [splits_tensor[i] for i in task_idx.tolist()]
            ):
                idx = out_vec[split].argmax()
                preds.append(split[idx].item())
            outputs = torch.tensor(preds, device=device, dtype=torch.long)

            for gt, p, p2 in zip(labels, outputs, outputs2):
                totals[gt] += 1
                if gt == p:
                    corrects[gt] += 1
                if gt == p2:
                    correct += 1
                total += 1

    print("acc: ", np.asarray(corrects) / np.asarray(totals))
    split_accs = [0] * len(splits)

    for i, split in enumerate(splits):
        print("last split: ", split)
        split_total = 0
        for _cls in split:
            split_accs[i] += corrects[_cls]
            split_total += totals[_cls]
        split_accs[i] /= max(split_total, 1e-4)

    print("split_acc: ", split_accs)

    return (
        correct / total,
        sum(split_accs) / len(split_accs),
        split_accs,
        np.asarray(corrects) / np.asarray(totals),
    )


def evaluate_model(eval_type, model, config, **opt_kwargs):
    """Evaluate methods on arbitrary experiment kinds."""
    if opt_kwargs.get("opt_dataloader", None) is not None:
        loader = opt_kwargs["opt_dataloader"]
        num_classes = opt_kwargs["opt_classes"]
    else:
        loader = config["data"]["test"]["full"]
        num_classes = len(config["data"]["test"]["class_names"])

    if eval_type == "logits":
        acc_overall, acc_avg, pertask_acc, perclass_acc = evaluate_logits_alltasks(
            model,
            loader,
            splits=config["dataset"]["class_splits"],
            num_classes=num_classes,
        )
        print(acc_overall)
        print(acc_avg)
        print(pertask_acc)
        print(perclass_acc)
    else:
        raise ValueError(f"Invalid eval_type: {eval_type}! Must be 'logits'.")

    results = {
        "Joint": acc_overall,
        "Per Task Avg": acc_avg,
        "Per class Acc": perclass_acc,
    }
    for task_idx, task_acc in enumerate(pertask_acc):
        results[f"Task {CONCEPT_TASKS[task_idx]}"] = task_acc

    return results
