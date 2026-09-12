"""Phase 2: all-pairs offline merge outcomes with the GLOBA operator (v2).

Same measurement path as tools/collect_merge_data_for.py -- same config
machinery, BN recalibration, evaluate_model and CSV columns -- so rows compare
one-to-one with merge_dataset_exp2d_saved.csv. Only the merge differs: the
child is built by globa.merge and loaded into a plain resnet20, so what is
scored IS what would propagate.

Regimes are `preset@eta@head`. Defaults cross the two questions Stage -1
left open for conv: does the label-aware head remove the disjoint-class loss
(Phase 2 v1: -0.379), and does type-weighting the backbone add anything once
the head is fixed.

    python -m globa.collect --out merge_dataset_exp2d_globa_v2.csv

Rows are written incrementally; re-running resumes. tools/ and sesil/ are
imported, never modified.
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
from sesil.evaluation import (  # noqa: E402
    evaluate_model, evaluate_fitness, inject_model, sort_model_name_unique,
)
from globa.operator import TYPES, MergeConfig, merge  # noqa: E402

DEFAULT_REGIMES = ["average@0.80@average", "average@0.80@label",
                   "single-full@0.80@average", "single-full@0.80@label"]


def parse_regime(s: str):
    name, eta, head = s.split("@")
    return name, float(eta), head


def classes_of(model_id: str):
    return set(int(x) for x in decode_labels(model_id).split("_"))  # noqa: F405


def _floats(xs):
    return [None if float(x) != float(x) else float(x) for x in xs]


def load_sd(path, device="cpu"):
    sd = torch.load(path, map_location=device)
    return sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd


def eval_parent(raw_config, model_id):
    inject_model(raw_config, model_id)
    config = prepare_experiment_config(raw_config)  # noqa: F405
    train_loader = config["data"]["train"]["full"]
    base = [reset_bn_stats(m, train_loader) for m in config["models"]["bases"]][0]  # noqa: F405
    res = evaluate_fitness(decode_labels(model_id), base, config)  # noqa: F405
    del base, config
    gc.collect(); torch.cuda.empty_cache()
    return res


def aggregate_stats(stats: dict) -> dict:
    w_sum, shares, ratios, typed = 0.0, {t: 0.0 for t in TYPES}, [], []
    for s in stats.values():
        w = s["norm_p"] ** 2 + s["norm_q"] ** 2
        for t in TYPES:
            shares[t] += s["energy_frac"][t] * w
        w_sum += w
        ratios.append(s["norm_ratio"])
        typed += [s["typed_frac_p"], s["typed_frac_q"]]
    ratios.sort()
    out = {f"type_{t}": (shares[t] / w_sum if w_sum else 0.0) for t in TYPES}
    out.update({"norm_ratio_min": ratios[0], "norm_ratio_med": ratios[len(ratios) // 2],
                "norm_ratio_max": ratios[-1], "typed_frac_mean": float(np.mean(typed)),
                "layers_analysed": len(stats)})
    return out


def merge_pair_globa(raw_config, pair, core_sd, cfg, device):
    inject_pair(raw_config, pair)  # noqa: F405
    config = prepare_experiment_config(raw_config)  # noqa: F405
    train_loader = config["data"]["train"]["full"]
    sd_a = {k: v.detach().cpu() for k, v in config["models"]["bases"][0].state_dict().items()}
    sd_b = {k: v.detach().cpu() for k, v in config["models"]["bases"][1].state_dict().items()}
    child_sd, stats = merge(sd_a, sd_b, core_sd, cfg, classes_of(pair[0]), classes_of(pair[1]),
                            return_stats=True)
    child = deepcopy(config["models"]["new"])
    child.load_state_dict(child_sd)
    child = child.to(device)
    reset_bn_stats(child, train_loader)  # noqa: F405
    res = evaluate_model(raw_config["eval_type"], child, config)
    del child, config
    gc.collect(); torch.cuda.empty_cache()
    return res, aggregate_stats(stats)


def build_parser():
    p = argparse.ArgumentParser(prog="globa.collect")
    p.add_argument("--population", default="runs_exp2d/seed0/population")
    p.add_argument("--core", default="runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
    p.add_argument("--out", default="merge_dataset_exp2d_globa_v2.csv")
    p.add_argument("--regimes", nargs="+", default=DEFAULT_REGIMES, help="preset@eta@head")
    p.add_argument("--reference", default="merge_dataset_exp2d_saved.csv")
    p.add_argument("--config-name", default="cifar_evolution_resnet20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    raw_config = get_config_from_name(args.config_name, device=args.device)  # noqa: F405
    raw_config["model"]["dir"] = args.population
    core_sd = load_sd(args.core, "cpu")

    model_ids = sorted(d for d in os.listdir(args.population) if os.path.isdir(os.path.join(args.population, d)))
    pairs = list(itertools.combinations(model_ids, 2))
    if args.limit:
        pairs = pairs[:args.limit]
    regimes = [parse_regime(r) for r in args.regimes]
    print(f"Population: {args.population}\nCore: {args.core}")
    print(f"{len(model_ids)} models -> {len(pairs)} pairs x {len(regimes)} regimes {args.regimes}\n")

    ref = {}
    if args.reference and os.path.exists(args.reference):
        rdf = pd.read_csv(args.reference)
        col = "saved_joint" if "saved_joint" in rdf.columns else "child_joint"
        ref = {(r.parent_a, r.parent_b): float(getattr(r, col)) for r in rdf.itertuples()}
        print(f"reference: {args.reference} [{col}] ({len(ref)} pairs)\n")

    rows, done = [], set()
    if os.path.exists(args.out):
        rows = pd.read_csv(args.out).to_dict("records")
        done = {(r["parent_a"], r["parent_b"], r["regime"]) for r in rows}
        print(f"resuming: {len(done)} rows already in {args.out}\n")

    parents = {}
    with torch.no_grad():
        for mid in model_ids:
            parents[mid] = eval_parent(raw_config, mid)
            print(f"parent {mid} ({decode_labels(mid)}): per-task {float(parents[mid]['Per Task Avg']):.4f}")  # noqa: F405
        print()
        n_total = len(pairs) * len(regimes)
        for a, b in pairs:
            for rname, eta, head in regimes:
                tag = f"{rname}@{eta:.2f}@{head}"
                if (a, b, tag) in done:
                    continue
                t0 = time.time()
                try:
                    cfg = MergeConfig.preset(rname, eta=eta, head=head)
                    res, agg = merge_pair_globa(raw_config, (a, b), core_sd, cfg, args.device)
                    row = {
                        "parent_a": a, "parent_b": b,
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
                        "regime": tag, "preset": rname, "eta": eta, "head": head,
                        **agg,
                        "merging_fn": "globa_v2", "population": args.population, "core": args.core,
                        "time_s": round(time.time() - t0, 1),
                    }
                    rows.append(row)
                    refs = f"  saved={ref[(a, b)]:.4f}" if (a, b) in ref else ""
                    print(f"[{len(rows)}/{n_total}] {row['labels_a']} x {row['labels_b']} {tag:30s} "
                          f"joint={row['child_joint']:.4f} per_task={row['child_per_task']:.4f}{refs} ({row['time_s']}s)")
                except Exception as e:  # noqa: BLE001
                    print(f"FAILED ({a}, {b}, {tag}): {type(e).__name__}: {e}")
                    gc.collect(); torch.cuda.empty_cache()
                if rows:
                    pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\nDone. {len(df)} rows -> {args.out}\n")
    if len(df):
        print(df.groupby("regime")[["child_joint", "child_per_task", "type_A", "type_E",
                                    "type_D_plus", "type_D_minus", "typed_frac_mean",
                                    "norm_ratio_med"]].mean().round(4).to_string())
        if ref:
            common = [k for k in ref if k in set(zip(df.parent_a, df.parent_b))]
            if common:
                print(f"\nreference saved_joint (parent A + BN reset) mean over {len(common)} pairs: "
                      f"{np.mean([ref[k] for k in common]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
