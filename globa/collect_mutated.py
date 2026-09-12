"""Stage 0b against the outcome that actually propagates: H0 on POST-mutation
children.

Every H0 measurement so far correlated GLOBA diagnostics with the child as
merged. In SESiL the object that enters the next generation is the child
AFTER mutation (2 epochs on the union of the parents' classes), and the
sandbox showed mutation reshuffles the ranking. This script builds each of
the 45 children with the operator, mutates it exactly as sesil.pipeline does
(finetune_merged_model, same loaders, same epochs/lr), scores it the same way,
and stores the per-pair diagnostics next to both scores.

    python -m globa.collect_mutated --preset average --head label --out merge_dataset_exp2d_globa_mutated.csv

Reuses SESiL's own evaluation, BN reset, loaders and mutation code; nothing in
sesil/ is modified.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import random
import sys
import time
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from utils import *  # noqa: E402,F403
from sesil.evaluation import evaluate_model, evaluate_fitness, inject_model, sort_model_name_unique  # noqa: E402
from sesil.data import get_full_loaders, register_legacy_configs  # noqa: E402
from sesil.mutation import finetune_merged_model  # noqa: E402
from globa.collect import aggregate_stats, classes_of, eval_parent, load_sd, _floats  # noqa: E402
from globa.operator import MergeConfig, merge  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.collect_mutated")
    ap.add_argument("--population", default="runs_exp2d/seed0/population")
    ap.add_argument("--core", default="runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
    ap.add_argument("--preset", default="average",
                    help="operator preset, or 'copy-a' for SESiL's own child (parent A's weights, "
                         "what save_offspring writes) run through the identical mutation/scoring")
    ap.add_argument("--head", default="label")
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--mutation-epochs", type=int, default=2)
    ap.add_argument("--mutation-lr", type=float, default=1e-3)
    ap.add_argument("--updates-per-epoch", type=int, default=0)
    ap.add_argument("--out", default="merge_dataset_exp2d_globa_mutated.csv")
    ap.add_argument("--config-name", default="cifar_evolution_resnet20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    register_legacy_configs()
    raw_config = get_config_from_name(args.config_name, device=args.device)  # noqa: F405
    raw_config["model"]["dir"] = args.population
    core_sd = load_sd(args.core, "cpu")
    copy_a = args.preset == "copy-a"
    cfg = None if copy_a else MergeConfig.preset(args.preset, eta=args.eta, head=args.head)
    tag = f"copy-a+mut{args.mutation_epochs}" if copy_a else f"{args.preset}@{args.eta:.2f}@{args.head}+mut{args.mutation_epochs}"
    # the loaders sesil.pipeline.run_evolution hands to finetune_merged_model
    trainloader, testloader = get_full_loaders("cifar10", batch_size=500, num_workers=0, seed=args.seed)

    model_ids = sorted(d for d in os.listdir(args.population) if os.path.isdir(os.path.join(args.population, d)))
    pairs = list(itertools.combinations(model_ids, 2))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(model_ids)} models -> {len(pairs)} pairs, {tag}\n")

    rows, done = [], set()
    if os.path.exists(args.out):
        rows = pd.read_csv(args.out).to_dict("records")
        done = {(r["parent_a"], r["parent_b"]) for r in rows}

    parents = {}
    with torch.no_grad():
        for mid in model_ids:
            parents[mid] = eval_parent(raw_config, mid)
    for a, b in pairs:
        if (a, b) in done:
            continue
        t0 = time.time()
        try:
            with torch.no_grad():
                inject_pair(raw_config, (a, b))  # noqa: F405
                config = prepare_experiment_config(raw_config)  # noqa: F405
                train_loader = config["data"]["train"]["full"]
                sd_a = {k: v.detach().cpu() for k, v in config["models"]["bases"][0].state_dict().items()}
                sd_b = {k: v.detach().cpu() for k, v in config["models"]["bases"][1].state_dict().items()}
                if copy_a:
                    child_sd, stats = sd_a, []
                else:
                    child_sd, stats = merge(sd_a, sd_b, core_sd, cfg, classes_of(a), classes_of(b), return_stats=True)
                child = deepcopy(config["models"]["new"]); child.load_state_dict(child_sd); child = child.to(args.device)
                reset_bn_stats(child, train_loader)  # noqa: F405
                pre = evaluate_model(raw_config["eval_type"], child, config)
            # SESiL's mutation, verbatim call
            child, _ = finetune_merged_model(child, trainloader, testloader, epochs=args.mutation_epochs,
                                             lr=args.mutation_lr, updates_per_epoch=args.updates_per_epoch)
            with torch.no_grad():
                reset_bn_stats(child, train_loader)  # noqa: F405
                post = evaluate_model(raw_config["eval_type"], child, config)
            agg = aggregate_stats(stats) if stats else {}
            row = {"parent_a": a, "parent_b": b,
                   "labels_a": sort_model_name_unique(decode_labels(a)), "labels_b": sort_model_name_unique(decode_labels(b)),  # noqa: F405
                   "a_per_class": json.dumps(_floats(parents[a]["Per Class"])), "b_per_class": json.dumps(_floats(parents[b]["Per Class"])),
                   "a_per_task": float(parents[a]["Per Task Avg"]), "b_per_task": float(parents[b]["Per Task Avg"]),
                   "pre_joint": float(pre["Joint"]), "pre_per_task": float(pre["Per Task Avg"]),
                   "pre_per_class": json.dumps(_floats(pre["Per class Acc"])),
                   "post_joint": float(post["Joint"]), "post_per_task": float(post["Per Task Avg"]),
                   "post_task_a": float(post.get("Task A", float("nan"))), "post_task_b": float(post.get("Task B", float("nan"))),
                   "post_per_class": json.dumps(_floats(post["Per class Acc"])),
                   "regime": tag, **agg, "time_s": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"[{len(rows)}/{len(pairs)}] {row['labels_a']} x {row['labels_b']}  pre {row['pre_joint']:.4f}/{row['pre_per_task']:.4f}"
                  f"  post {row['post_joint']:.4f}/{row['post_per_task']:.4f}  ({row['time_s']}s)")
            del child, config; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"FAILED ({a}, {b}): {type(e).__name__}: {e}")
            gc.collect(); torch.cuda.empty_cache()
        if rows:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\nDone. {len(df)} rows -> {args.out}")
    if len(df) > 5 and "type_A" in df:
        from scipy.stats import spearmanr
        df["mean_parent"] = 0.5 * (df.a_per_task + df.b_per_task)
        df["excess_post"] = df.post_per_task - df.mean_parent
        print(f"\npre  joint {df.pre_joint.mean():.4f} per_task {df.pre_per_task.mean():.4f}")
        print(f"post joint {df.post_joint.mean():.4f} per_task {df.post_per_task.mean():.4f}")
        print("\nH0 on POST-mutation outcomes, Spearman rho (p):")
        print(f"  {'feature':16s} {'post_joint':>14s} {'post_per_task':>14s} {'excess_post':>14s}")
        for f in ["type_A", "type_B", "type_C", "type_E", "type_D_plus", "type_D_minus", "typed_frac_mean", "norm_ratio_med", "mean_parent", "pre_per_task"]:
            out = []
            for tgt in ["post_joint", "post_per_task", "excess_post"]:
                r, p = spearmanr(df[f], df[tgt]); out.append(f"{r:+.3f} ({p:.2f})")
            print(f"  {f:16s} " + " ".join(f"{o:>14s}" for o in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
