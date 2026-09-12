"""External evolution driver: SESiL's loop with a pluggable crossover.

This is `sesil.pipeline.run_generation` / `run_evolution` re-stated OUTSIDE
sesil/, importing every stage from sesil as a library and replacing only the
merge step. `sesil/` is not modified: `git diff sesil/ run_sesil.py` is empty.

    # validation: SESiL's own merge as the crossover -> must reproduce run_sesil.py
    python -m globa.run --out-root runs_phase3_ext --merge sesil --generations 1

    # the GLOBA operator as the crossover
    python -m globa.run --out-root runs_globa_sf --merge globa \
        --preset single-full --head label --generations 25

Layout mirrors RunPaths: <out_root>/seed<seed>/population is the gen-0
population (copied from --init-population if absent), generations go to
<out_root>/seed<seed>/<method-name>/gen_N/, CSVs alongside as in SESiL.

Merge-stage rows are written with SESiL's own columns so tools/plot_*.py and
tools/eval_generations.py read them unchanged. For GLOBA runs an extra
globa_stats.jsonl records per-merge type shares and the norm ratio.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from utils import (  # noqa: E402
    CONCEPT_TASKS, decode_labels, encode_labels, flatten_nested_dict,
    get_config_from_name, inject_pair, prepare_experiment_config, reset_bn_stats,
    write_to_csv,
)
from models.resnets import resnet20  # noqa: E402
from sesil.config import RunConfig  # noqa: E402
from sesil.data import get_full_loaders, register_legacy_configs  # noqa: E402
from sesil.evaluation import evaluate_model, inject_model, run_evolutionary_evaluation, sort_model_name_unique  # noqa: E402
from sesil.mapping_guard import mapping_guard  # noqa: E402
from sesil.mating import mate_with_population_info  # noqa: E402
from sesil.merging import build_unique_key, run_merging  # noqa: E402
from sesil.mutation import finetune_merged_model  # noqa: E402
from sesil.paths import RunPaths  # noqa: E402
from sesil.pipeline import list_population, save_offspring  # noqa: E402
from sesil.seeding import set_all_seeds  # noqa: E402
from globa.operator import TYPES, MergeConfig, merge  # noqa: E402


def classes_of(model_id: str):
    return set(int(x) for x in decode_labels(model_id).split("_"))


def load_sd(path, device="cpu"):
    sd = torch.load(path, map_location=device)
    return sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd


def _agg(stats):
    w_sum, sh, ratios, typed = 0.0, {t: 0.0 for t in TYPES}, [], []
    for s in stats.values():
        w = s["norm_p"] ** 2 + s["norm_q"] ** 2
        for t in TYPES:
            sh[t] += s["energy_frac"][t] * w
        w_sum += w; ratios.append(s["norm_ratio"]); typed += [s["typed_frac_p"], s["typed_frac_q"]]
    ratios.sort()
    return {**{f"type_{t}": sh[t] / w_sum if w_sum else 0.0 for t in TYPES},
            "norm_ratio_med": ratios[len(ratios) // 2], "norm_ratio_max": ratios[-1],
            "typed_frac_mean": sum(typed) / len(typed)}


# --------------------------------------------------------------------------
# the pluggable stage 3
# --------------------------------------------------------------------------
def merge_step_sesil(cfg, raw_config, pairs, loners, csv_file, **_):
    """Test double: SESiL's own crossover, untouched."""
    return run_merging(cfg.merging_fn, cfg.node_config, raw_config, pairs, loners, cfg.device, csv_file)


def merge_step_sesil_full(cfg, raw_config, pairs, loners, csv_file, tag, **_):
    """SESiL's own merger (wavg / permute / zipit via ZipIt!'s ModelMerge) applied to the
    WHOLE network (stop_at=None), and the merged network itself -- not head_models[0] --
    becomes the offspring. Everything else (metrics, matching, BN reset, loners) is
    sesil.merging.run_merging verbatim. This is the fair "standard merger" line:
    it differs from SESiL only in stop_at and in what is saved."""
    from copy import deepcopy
    from model_merger import ModelMerge
    from utils import get_merging_fn
    offsprings, seen_keys = {}, set()
    node_config = dict(cfg.node_config); node_config["stop_node"] = None
    device, model_name = cfg.device, raw_config["model"]["name"]

    for pair in pairs:
        raw_config = inject_pair(raw_config, pair)
        config = prepare_experiment_config(raw_config)
        train_loader = config["data"]["train"]["full"]
        base_models = [reset_bn_stats(m, train_loader) for m in config["models"]["bases"]]
        config["node"] = node_config
        graphs = [config["graph"](deepcopy(m)).graphify() for m in base_models]
        Merge = ModelMerge(*graphs, device=device)
        Merge.transform(deepcopy(config["models"]["new"]), train_loader,
                        transform_fn=get_merging_fn(cfg.merging_fn), metric_classes=config["metric_fns"],
                        stop_at=None, **node_config["params"])
        # the merged network is a plain resnet20 (no hooks when stop_at is None)
        child = resnet20(w=cfg.model_width, num_classes=cfg.num_classes)
        child.load_state_dict(Merge.merged_model.state_dict())
        child = child.to(device)
        reset_bn_stats(child, train_loader)

        results = evaluate_model(raw_config["eval_type"], child, config)
        for idx, split in enumerate(pair):
            results[f"Split {CONCEPT_TASKS[idx]}"] = sort_model_name_unique(decode_labels(split))
        results["Time"] = Merge.compute_transform_time
        results["Merging Fn"] = tag
        results["Model Name"] = model_name
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)

        labels = "_".join(decode_labels(p) for p in pair).split("_")
        offsprings[build_unique_key(labels, seen_keys)] = child.cpu()
        del Merge, graphs, base_models, child, config
        gc.collect(); torch.cuda.empty_cache()

    # loners: verbatim sesil.merging.run_merging, sharing seen_keys
    for loner in loners:
        raw_config = inject_model(raw_config, loner)
        config = prepare_experiment_config(raw_config)
        train_loader = config["data"]["train"]["full"]
        base_model = reset_bn_stats(config["models"]["bases"][0], train_loader)
        config["node"] = node_config
        results = evaluate_model(raw_config["eval_type"], base_model, config)
        results["Split"] = sort_model_name_unique(decode_labels(loner))
        results["Time"] = 0.0
        results["Merging Fn"] = "None"
        results["Model Name"] = model_name
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)
        offsprings[build_unique_key(decode_labels(loner).split("_"), seen_keys)] = base_model.cpu()
        del config
        gc.collect(); torch.cuda.empty_cache()
    return offsprings


def merge_step_globa(cfg, raw_config, pairs, loners, csv_file, gcfg, core_sd, tag, stats_path, gen, **_):
    """GLOBA operator for pairs; loners handled exactly as sesil.merging does.

    gcfg=None is the copy-A control: the child is parent A's weights in a whole
    resnet, mutated end to end. It differs from SESiL's own path only in the
    save artefact (SESiL keeps parent A's stage 1 and mutates stages 2-3 on the
    merged trunk's features); the crossover is the same "no mixing".
    """
    offsprings, seen_keys = {}, set()
    node_config, device = cfg.node_config, cfg.device
    model_name = raw_config["model"]["name"]

    for pair in pairs:
        raw_config = inject_pair(raw_config, pair)
        config = prepare_experiment_config(raw_config)
        train_loader = config["data"]["train"]["full"]
        sd_a = {k: v.detach().cpu() for k, v in config["models"]["bases"][0].state_dict().items()}
        sd_b = {k: v.detach().cpu() for k, v in config["models"]["bases"][1].state_dict().items()}
        if gcfg is None:
            child_sd, stats = sd_a, {}
        else:
            child_sd, stats = merge(sd_a, sd_b, core_sd, gcfg, classes_of(pair[0]), classes_of(pair[1]),
                                    return_stats=True)
        child = resnet20(w=cfg.model_width, num_classes=cfg.num_classes)
        child.load_state_dict(child_sd)
        child = child.to(device)
        reset_bn_stats(child, train_loader)

        results = evaluate_model(raw_config["eval_type"], child, config)
        for idx, split in enumerate(pair):
            results[f"Split {CONCEPT_TASKS[idx]}"] = sort_model_name_unique(decode_labels(split))
        results["Time"] = 0.0
        results["Merging Fn"] = tag
        results["Model Name"] = model_name
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)

        if stats:
            with open(stats_path, "a") as f:
                f.write(json.dumps({"gen": gen, "pair": list(pair), **_agg(stats)}) + "\n")

        labels = "_".join(decode_labels(p) for p in pair).split("_")
        offsprings[build_unique_key(labels, seen_keys)] = child.cpu()
        del child, config
        gc.collect(); torch.cuda.empty_cache()

    # loners: verbatim sesil.merging.run_merging lines 121-144, sharing seen_keys
    for loner in loners:
        raw_config = inject_model(raw_config, loner)
        config = prepare_experiment_config(raw_config)
        train_loader = config["data"]["train"]["full"]
        base_model = reset_bn_stats(config["models"]["bases"][0], train_loader)
        config["node"] = node_config
        results = evaluate_model(raw_config["eval_type"], base_model, config)
        results["Split"] = sort_model_name_unique(decode_labels(loner))
        results["Time"] = 0.0
        results["Merging Fn"] = "None"
        results["Model Name"] = model_name
        results.update(flatten_nested_dict(node_config, sep=" "))
        write_to_csv(results, csv_file=csv_file)
        offsprings[build_unique_key(decode_labels(loner).split("_"), seen_keys)] = base_model.cpu()
        del config
        gc.collect(); torch.cuda.empty_cache()
    return offsprings


# --------------------------------------------------------------------------
# sesil.pipeline.run_generation / run_evolution, restated
# --------------------------------------------------------------------------
def run_generation(gen, cfg, paths, raw_config, trainloader, testloader, merge_step, **mkw):
    model_dir = raw_config["model"]["dir"]
    print(f"\n ============== Generation {gen} ==============\n population dir: {model_dir}\n")
    csv_file = paths.csv_for(gen)
    models_ids = list_population(model_dir)
    raw_config["dataset"].update(cfg.node_config.get("dataset", {}))

    with torch.no_grad():
        population_info = run_evolutionary_evaluation(
            node_config=cfg.node_config, experiment_config=raw_config, model_ids=models_ids,
            device=cfg.device, csv_file=csv_file, num_classes=cfg.num_classes)
        pairs, loners = mate_with_population_info(population_info, cfg)
        print("Parents:\n", pairs); print("Loners:\n", loners)
        offsprings = merge_step(cfg=cfg, raw_config=raw_config, pairs=pairs, loners=loners,
                                csv_file=csv_file, gen=gen, **mkw)

    print(f"\nWe collected {len(offsprings)} offsprings.\n")
    gen_dir = paths.generation_dir(gen)
    mutated_accs = []
    for model_name in list(offsprings.keys()):
        model = offsprings.pop(model_name).to(cfg.device)
        print(f"Processing model: {model_name}")
        mutated_model, acc = finetune_merged_model(
            model, trainloader, testloader, epochs=cfg.mutation_epochs, lr=cfg.mutation_lr,
            updates_per_epoch=cfg.updates_per_epoch)
        mutated_accs.append(acc)
        with mapping_guard(cfg.mapping_file):
            hash_id = encode_labels(model_name, mapping_file=cfg.mapping_file)
        save_dir = paths.offspring_dir(gen, hash_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"resnet20x{cfg.model_width}_v{len(os.listdir(save_dir))}.pth.tar")
        save_offspring(mutated_model, save_path)
        print(f"Saved {model_name} -> {save_path}")
        mutated_model.cpu(); del mutated_model, model
        gc.collect(); torch.cuda.empty_cache()

    stats = {"generation": gen, "population": len(models_ids), "pairs": len(pairs) // 2,
             "loners": len(loners), "offspring": len(mutated_accs),
             "mutated_acc_mean": (sum(mutated_accs) / len(mutated_accs) if mutated_accs else None),
             "csv": csv_file}
    return stats, gen_dir


def run_evolution(cfg, paths, merge_step, extra_json, **mkw):
    set_all_seeds(cfg.seed)
    register_legacy_configs()
    raw_config = get_config_from_name(cfg.config_name, device=cfg.device)
    # resume: generation start_from+1 mates the population saved as gen_<start_from>
    raw_config["model"]["dir"] = paths.generation_dir(cfg.start_from) if cfg.start_from > 0 else paths.population_dir
    raw_config["dataset"]["name"] = cfg.spec.evo_name
    trainloader, testloader = get_full_loaders(cfg.dataset, batch_size=cfg.pretrain_batch_size,
                                               num_workers=cfg.pretrain_num_workers, seed=cfg.seed)
    history, t0 = [], time.time()
    for g in range(cfg.generations):
        gen = g + 1 + cfg.start_from
        stats, next_dir = run_generation(gen, cfg, paths, raw_config, trainloader, testloader, merge_step, **mkw)
        history.append(stats)
        raw_config["model"]["dir"] = next_dir
        with open(paths.run_json, "w") as f:
            json.dump({"config": cfg.__dict__, **extra_json, "history": history}, f, indent=2, default=str)
    print(f"\n[globa.run] {cfg.generations} generations in {time.time() - t0:.1f}s")
    return history


def main(argv=None):
    ap = argparse.ArgumentParser(prog="globa.run")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--method-name", default=None, help="directory name under seed/ (default: sesil | globa_<preset>_<head>)")
    ap.add_argument("--init-population", default="runs_exp2d/seed0/population",
                    help="copied to <out-root>/seed<seed>/population if that does not exist")
    ap.add_argument("--merge", choices=["sesil", "sesil-full", "globa", "copy-a"], default="globa",
                    help="sesil = SESiL's own crossover (partial zipping, saves head_models[0]); "
                         "sesil-full = SESiL's own merger on the whole network, merged network saved; "
                         "globa = the operator; copy-a = parent A as a whole network, fully mutated")
    ap.add_argument("--method", choices=["wavg", "permute", "zipit"], default="permute",
                    help="SESiL merging function for --merge sesil / sesil-full (ignored otherwise)")
    ap.add_argument("--preset", default="single-full")
    ap.add_argument("--eta", type=float, default=0.80)
    ap.add_argument("--head", choices=["label", "average"], default="label")
    ap.add_argument("--core", default="runs_ssl/simsiam_e50_seed0/backbone_e9.pth.tar")
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--start-from", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--num-agents", type=int, default=10)
    ap.add_argument("--classes-per-agent", type=int, default=3)
    ap.add_argument("--updates-per-epoch", type=int, default=0, help="0 = full pass, as exp2d")
    ap.add_argument("--mutation-epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    method_name = args.method_name or {"sesil": "sesil", "sesil-full": f"sesil-full_{args.method}",
                                       "copy-a": "copya"}.get(args.merge, f"globa_{args.preset}_{args.head}")
    cfg = RunConfig(seed=args.seed, method=args.method, dataset=args.dataset, n_agents=args.num_agents,
                    classes_per_agent=args.classes_per_agent, updates_per_epoch=args.updates_per_epoch,
                    generations=args.generations, start_from=args.start_from,
                    mutation_epochs=args.mutation_epochs, out_root=args.out_root, device=args.device,
                    skip_pretrain=True)
    paths = RunPaths(cfg.out_root, cfg.seed, method_name, cfg.model_width)
    if not os.path.isdir(paths.population_dir):
        print(f"copying population {args.init_population} -> {paths.population_dir}")
        shutil.copytree(args.init_population, paths.population_dir)
    paths.makedirs()
    print(cfg.describe()); print(f"  crossover        : {args.merge}"
          + {"globa": f" ({args.preset}@{args.eta}@{args.head}, core {args.core})",
             "sesil": f" (run_merging, SESiL's own, {args.method}, stop_node {cfg.stop_node})",
             "sesil-full": f" (SESiL's {args.method} on the whole network, stop_at=None, merged network saved)",
             "copy-a": " (parent A's weights, whole network)"}[args.merge])
    print(f"  output           : {paths.method_dir}\n")

    if args.merge == "sesil":
        history = run_evolution(cfg, paths, merge_step_sesil, {"crossover": "sesil"})
    elif args.merge == "sesil-full":
        tag = f"sesil-full:{cfg.merging_fn}"
        history = run_evolution(cfg, paths, merge_step_sesil_full, {"crossover": tag, "stop_node": None}, tag=tag)
    elif args.merge == "copy-a":
        history = run_evolution(cfg, paths, merge_step_globa, {"crossover": "copy-a"},
                                gcfg=None, core_sd=None, tag="copy-a", stats_path=None)
    else:
        gcfg = MergeConfig.preset(args.preset, eta=args.eta, head=args.head)
        core_sd = load_sd(args.core, "cpu")
        tag = f"globa:{args.preset}@{args.eta}@{args.head}"
        stats_path = os.path.join(paths.method_dir, "globa_stats.jsonl")
        history = run_evolution(cfg, paths, merge_step_globa,
                                {"crossover": tag, "globa": {**gcfg.__dict__, "alpha": dict(gcfg.alpha)}, "core": args.core},
                                gcfg=gcfg, core_sd=core_sd, tag=tag, stats_path=stats_path)
    for h in history:
        print(f"gen {h['generation']:>2}  pairs {h['pairs']}  loners {h['loners']}  "
              f"mutated acc {h['mutated_acc_mean']:.4f}" if h['mutated_acc_mean'] is not None else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
