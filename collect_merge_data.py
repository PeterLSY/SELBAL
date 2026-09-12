"""
collect_merge_data.py  —  Phase 1: merge-outcome dataset for the thesis predictor.

Place this file in the SESiL repo ROOT (next to utils.py), then run:

    conda activate sesil
    python collect_merge_data.py

For every pair (A, B) in the initial population it:
  1. evaluates both parents' per-class accuracy,
  2. merges them with MERGING_FN (same path as run_auxiliary_experiment),
  3. evaluates the child,
  4. appends one row to OUT_CSV (saved incrementally — a crash loses nothing),
  5. frees ALL GPU memory before the next pair (no offspring retention).

10 models -> 45 pairs -> roughly 1 hour if per-pair time stays ~70 s.
Watch the printed time_s: it should stay flat. If it climbs, the offspring CPU-offloading fix has regressed.
"""

import os, gc, time, json, random, itertools
import torch
import numpy as np
import pandas as pd
from copy import deepcopy

from utils import *
from model_merger import ModelMerge
# Safe to import: the training script's main block is guarded by __name__ == "__main__"
from training_scripts.evolutionary_permute_training import (
    evaluate_model, evaluate_fitness, inject_model, sort_model_name_unique,
)

torch.manual_seed(0); random.seed(0); np.random.seed(0)

# ---------------- configuration (mirrors the training script) ----------------
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_NAME = 'cifar_evolution_resnet20'
MERGING_FN  = 'match_tensors_permute'   # rerun later with other merging fns
NODE_CONFIG = {'stop_node': 21, 'params': {'a': .0001, 'b': .075}}
OUT_CSV     = f'merge_dataset_{MERGING_FN}.csv'

raw_config = get_config_from_name(CONFIG_NAME, device=DEVICE)
MODEL_DIR  = raw_config['model']['dir']


def _floats(xs):
    """List of floats; NaN -> None so it survives JSON."""
    out = []
    for x in xs:
        v = float(x)
        out.append(None if v != v else v)
    return out


def eval_parent(model_id):
    """Per-class fitness of one parent (mirrors run_evolutionary_evaluation)."""
    inject_model(raw_config, model_id)                 # in-place, like the repo does
    config = prepare_experiment_config(raw_config)
    train_loader = config['data']['train']['full']
    base = [reset_bn_stats(m, train_loader) for m in config['models']['bases']][0]
    res = evaluate_fitness(decode_labels(model_id), base, config)
    del base, config
    gc.collect(); torch.cuda.empty_cache()
    return res


def merge_pair(pair):
    """Merge one pair and evaluate the child (mirrors run_auxiliary_experiment)."""
    inject_pair(raw_config, pair)                      # in-place, like the repo does
    config = prepare_experiment_config(raw_config)
    train_loader = config['data']['train']['full']
    base_models = [reset_bn_stats(m, train_loader) for m in config['models']['bases']]
    config['node'] = NODE_CONFIG
    Grapher = config['graph']
    graphs = [Grapher(deepcopy(m)).graphify() for m in base_models]

    Merge = ModelMerge(*graphs, device=DEVICE)
    Merge.transform(
        deepcopy(config['models']['new']),
        train_loader,
        transform_fn=get_merging_fn(MERGING_FN),
        metric_classes=config['metric_fns'],
        stop_at=NODE_CONFIG['stop_node'],
        **NODE_CONFIG['params'],
    )
    reset_bn_stats(Merge, train_loader)

    res = evaluate_model(raw_config['eval_type'], Merge, config)

    del Merge, graphs, base_models, config
    gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    model_ids = sorted(os.listdir(MODEL_DIR))
    n_pairs = len(model_ids) * (len(model_ids) - 1) // 2
    print(f'Population dir: {MODEL_DIR}')
    print(f'{len(model_ids)} models -> {n_pairs} pairs\n')

    # ---- 1) parent fitness --------------------------------------------------
    parents = {}
    with torch.no_grad():
        for mid in model_ids:
            r = eval_parent(mid)
            parents[mid] = r
            print(f'parent {mid} ({decode_labels(mid)}): '
                  f'per-task {float(r["Per Task Avg"]):.4f}')

    # ---- 2) all pairwise merges ---------------------------------------------
    rows = []
    with torch.no_grad():
        for a, b in itertools.combinations(model_ids, 2):
            t0 = time.time()
            try:
                res = merge_pair((a, b))
                row = {
                    'parent_a': a,
                    'parent_b': b,
                    'labels_a': sort_model_name_unique(decode_labels(a)),
                    'labels_b': sort_model_name_unique(decode_labels(b)),
                    'a_per_class': json.dumps(_floats(parents[a]['Per Class'])),
                    'b_per_class': json.dumps(_floats(parents[b]['Per Class'])),
                    'a_per_task': float(parents[a]['Per Task Avg']),
                    'b_per_task': float(parents[b]['Per Task Avg']),
                    'child_joint': float(res['Joint']),
                    'child_per_task': float(res['Per Task Avg']),
                    'child_task_a': float(res.get('Task A', float('nan'))),
                    'child_task_b': float(res.get('Task B', float('nan'))),
                    'child_per_class': json.dumps(_floats(res['Per class Acc'])),
                    'merging_fn': MERGING_FN,
                    'time_s': round(time.time() - t0, 1),
                }
                rows.append(row)
                print(f'[{len(rows)}/{n_pairs}] {row["labels_a"]} x {row["labels_b"]}: '
                      f'joint={row["child_joint"]:.4f} '
                      f'per_task={row["child_per_task"]:.4f} ({row["time_s"]}s)')
            except Exception as e:
                print(f'FAILED pair ({a}, {b}): {type(e).__name__}: {e}')
                gc.collect(); torch.cuda.empty_cache()

            if rows:
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f'\nDone. {len(rows)}/{n_pairs} rows written to {OUT_CSV}')


if __name__ == '__main__':
    main()