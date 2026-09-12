"""All-pairs merge outcomes for ANY population directory (proposal Stage 0b).

Same measurement as collect_merge_data.py, which is hardwired to the original
independent-init population. This one takes the population as an argument, so
the shared-core populations can be measured too.

    python tools/collect_merge_data_for.py \
        --population runs_exp2d/seed0/population \
        --out merge_dataset_exp2d_permute.csv

The original collect_merge_data.py is left untouched, so the pilot numbers in
the proposal stay reproducible exactly as they were produced.

Rows are written incrementally: a crash or a stop loses only the pair in
flight. Re-running skips pairs already present in --out.
"""

import argparse
import gc
import itertools
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from copy import deepcopy  # noqa: E402

from utils import *  # noqa: E402,F403
from model_merger import ModelMerge  # noqa: E402
from training_scripts.evolutionary_permute_training import (  # noqa: E402
    evaluate_model, evaluate_fitness, inject_model, sort_model_name_unique,
)

# stop_node decides HOW MUCH of the network is actually merged. resnet20x4's
# graph has 12 mergeable nodes: 5/11/17/21 (64ch), 26/32/38/42 (128ch),
# 47/53/59/63 (256ch). SESiL's default 21 merges only the first stage -- every
# layer after it is DUPLICATED into two branches, not combined. None = merge
# everything.
DEFAULT_STOP_NODE = 21


def _floats(xs):
    out = []
    for x in xs:
        v = float(x)
        out.append(None if v != v else v)
    return out


def eval_parent(raw_config, model_id):
    inject_model(raw_config, model_id)
    config = prepare_experiment_config(raw_config)  # noqa: F405
    train_loader = config["data"]["train"]["full"]
    base = [reset_bn_stats(m, train_loader) for m in config["models"]["bases"]][0]  # noqa: F405
    res = evaluate_fitness(decode_labels(model_id), base, config)  # noqa: F405
    del base, config
    gc.collect(); torch.cuda.empty_cache()
    return res


def merge_pair(raw_config, pair, merging_fn, device, bn_reset=True,
               stop_node=DEFAULT_STOP_NODE):
    inject_pair(raw_config, pair)  # noqa: F405
    config = prepare_experiment_config(raw_config)  # noqa: F405
    train_loader = config["data"]["train"]["full"]
    base_models = [reset_bn_stats(m, train_loader) for m in config["models"]["bases"]]  # noqa: F405
    node_config = {"stop_node": stop_node, "params": {"a": 0.0001, "b": 0.075}}
    config["node"] = node_config
    Grapher = config["graph"]
    graphs = [Grapher(deepcopy(m)).graphify() for m in base_models]

    Merge = ModelMerge(*graphs, device=device)
    Merge.transform(
        deepcopy(config["models"]["new"]),
        train_loader,
        transform_fn=get_merging_fn(merging_fn),  # noqa: F405
        metric_classes=config["metric_fns"],
        stop_at=stop_node,
        **node_config["params"],
    )
    # The post-merge BN recalibration. Skipping it isolates how much of the
    # merge outcome is weight interference versus BN statistics being
    # re-estimated on real data.
    if bn_reset:
        reset_bn_stats(Merge, train_loader)  # noqa: F405

    res = evaluate_model(raw_config["eval_type"], Merge, config)

    # What the composite scores is NOT what propagates. ModelMerge.forward
    # returns one output per branch, so `res` describes "shared trunk + both
    # parents' branches". save_offspring writes head_models[0] -- a single
    # network -- and that is what the next generation inherits. Measure it too.
    saved = None
    if hasattr(Merge, "head_models"):
        saved = evaluate_model(raw_config["eval_type"], Merge.head_models[0], config)

    del Merge, graphs, base_models, config
    gc.collect(); torch.cuda.empty_cache()
    return res, saved


def build_parser():
    p = argparse.ArgumentParser(prog="collect_merge_data_for.py")
    p.add_argument("--population", required=True,
                   help="directory of <hash>/<model>_v0.pth.tar")
    p.add_argument("--out", required=True)
    p.add_argument("--merging-fn", default="match_tensors_permute",
                   choices=["match_tensors_permute", "match_tensors_identity",
                            "match_tensors_zipit"])
    p.add_argument("--config-name", default="cifar_evolution_resnet20")
    p.add_argument("--dataset", default=None,
                   help="override the config's dataset name (e.g. cifar100)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stop-node", type=int, default=DEFAULT_STOP_NODE,
                   help="merge the graph up to this node; -1 merges the whole "
                        "network (nodes: 5,11,17,21 | 26,32,38,42 | 47,53,59,63)")
    p.add_argument("--no-bn-reset", action="store_true",
                   help="skip the post-merge BN recalibration, to separate "
                        "weight interference from BN re-estimation")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    raw_config = get_config_from_name(args.config_name, device=args.device)  # noqa: F405
    raw_config["model"]["dir"] = args.population
    if args.dataset:
        from sesil.data import register_legacy_configs, get_spec
        register_legacy_configs()
        raw_config["dataset"]["name"] = get_spec(args.dataset).evo_name

    model_ids = sorted(d for d in os.listdir(args.population)
                       if os.path.isdir(os.path.join(args.population, d)))
    pairs = list(itertools.combinations(model_ids, 2))
    print(f"Population: {args.population}")
    print(f"{len(model_ids)} models -> {len(pairs)} pairs, merger {args.merging_fn}\n")

    rows, done = [], set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        rows = prev.to_dict("records")
        done = {(r["parent_a"], r["parent_b"]) for r in rows}
        print(f"resuming: {len(done)} pairs already in {args.out}\n")

    parents = {}
    with torch.no_grad():
        for mid in model_ids:
            r = eval_parent(raw_config, mid)
            parents[mid] = r
            print(f"parent {mid} ({decode_labels(mid)}): "  # noqa: F405
                  f"per-task {float(r['Per Task Avg']):.4f}")

        for a, b in pairs:
            if (a, b) in done:
                continue
            t0 = time.time()
            try:
                res, saved = merge_pair(
                    raw_config, (a, b), args.merging_fn, args.device,
                    bn_reset=not args.no_bn_reset,
                    stop_node=(None if args.stop_node < 0 else args.stop_node))
                row = {
                    "parent_a": a,
                    "parent_b": b,
                    "labels_a": sort_model_name_unique(decode_labels(a)),  # noqa: F405
                    "labels_b": sort_model_name_unique(decode_labels(b)),  # noqa: F405
                    "a_per_class": json.dumps(_floats(parents[a]["Per Class"])),
                    "b_per_class": json.dumps(_floats(parents[b]["Per Class"])),
                    "a_per_task": float(parents[a]["Per Task Avg"]),
                    "b_per_task": float(parents[b]["Per Task Avg"]),
                    "child_joint": float(res["Joint"]),
                    "child_per_task": float(res["Per Task Avg"]),
                    "child_task_a": float(res.get("Task A", float("nan"))),
                    "child_task_b": float(res.get("Task B", float("nan"))),
                    "child_per_class": json.dumps(_floats(res["Per class Acc"])),
                    "saved_joint": (float(saved["Joint"]) if saved else None),
                    "saved_per_task": (float(saved["Per Task Avg"]) if saved else None),
                    "merging_fn": args.merging_fn,
                    "population": args.population,
                    "bn_reset": not args.no_bn_reset,
                    "stop_node": args.stop_node,
                    "time_s": round(time.time() - t0, 1),
                }
                rows.append(row)
                print(f"[{len(rows)}/{len(pairs)}] {row['labels_a']} x {row['labels_b']}: "
                      f"joint={row['child_joint']:.4f} "
                      f"per_task={row['child_per_task']:.4f} ({row['time_s']}s)")
            except Exception as e:  # noqa: BLE001
                print(f"FAILED pair ({a}, {b}): {type(e).__name__}: {e}")
                gc.collect(); torch.cuda.empty_cache()

            if rows:
                pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"\nDone. {len(rows)}/{len(pairs)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
