"""The orchestrator: pretrain -> (evaluate -> mate -> merge -> mutate) x N.

This replaces the `__main__` block that was duplicated across
evolutionary_{wavg,permute,zipit}_training.py. Ordering and semantics are
unchanged; what differs is that nothing is hardcoded and every path is
namespaced by seed.

One deliberate behavioural change: the population listing uses `sorted(...)`
instead of bare `os.listdir` (legacy `find_runable_pairs`,
evolutionary_wavg_training.py:51-62, which had its pairing logic commented out
and just returned os.listdir). Listing order determines population_info order,
which feeds the mating draws, so leaving it to the filesystem would make a run
non-reproducible across machines even at a fixed seed.
"""

import gc
import json
import os
import time

import torch

from utils import encode_labels, get_config_from_name

from .evaluation import run_evolutionary_evaluation
from .mapping_guard import mapping_guard
from .mating import mate_with_population_info
from .merging import run_merging
from .mutation import finetune_merged_model
from .data import get_full_loaders, register_legacy_configs
from .seeding import set_all_seeds


def save_offspring(model, save_path, head_index=0):
    """Save an offspring, unwrapping ModelMerge if needed.

    A merged offspring is a ModelMerge, whose state_dict is prefixed
    `head_models.0.*`, `head_models.1.*`, `merged_model.*` -- resnet20 cannot
    load that, so the next generation's `prepare_resnets` would blow up with
    "Missing key(s) ... Unexpected key(s) head_models.0.conv1.weight".

    The legacy scripts handled this with their OWN save_model that pulls
    `head_models[head_index]` (evolutionary_wavg_training.py:835-841,
    evolutionary_permute_training.py:859-866). They shadow the `save_model`
    imported via `from utils import *`, and `utils.save_model` (utils.py:1011)
    does NOT unwrap. Importing the utils one here silently produced
    unloadable checkpoints -- generation N finished fine and generation N+1
    died on load.

    Loners are plain resnet20s and take the else branch.
    """
    if hasattr(model, "head_models"):
        sd = model.head_models[head_index].state_dict()
    else:
        sd = model.state_dict()
    torch.save(sd, save_path)


def list_population(model_dir):
    """Model ids in a generation directory, in a stable order."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"population directory not found: {model_dir}")
    ids = [
        name
        for name in sorted(os.listdir(model_dir))
        if os.path.isdir(os.path.join(model_dir, name))
    ]
    if not ids:
        raise RuntimeError(f"population directory is empty: {model_dir}")
    return ids


def run_generation(gen, cfg, paths, raw_config, trainloader, testloader):
    """One full generation. Returns (stats dict, next model_dir)."""
    model_dir = raw_config["model"]["dir"]
    print("")
    print(f" ============== Generation {gen} ==============")
    print(f" population dir: {model_dir}")
    print("")

    device = cfg.device
    node_config = cfg.node_config
    csv_file = paths.csv_for(gen)

    models_ids = list_population(model_dir)
    raw_config["dataset"].update(node_config.get("dataset", {}))

    with torch.no_grad():
        # -- stage 1: evaluate the population ---------------------------
        population_info = run_evolutionary_evaluation(
            node_config=node_config,
            experiment_config=raw_config,
            model_ids=models_ids,
            device=device,
            csv_file=csv_file,
            num_classes=cfg.num_classes,
        )

        # -- stage 2: score and pair ------------------------------------
        pairs, loners = mate_with_population_info(population_info, cfg)
        print("Parents:\n", pairs)
        print("Loners:\n", loners)

        # -- stage 3: merge ---------------------------------------------
        offsprings = run_merging(
            merging_fn=cfg.merging_fn,
            node_config=node_config,
            experiment_config=raw_config,
            pairs=pairs,
            loners=loners,
            device=device,
            csv_file=csv_file,
        )

    print(f"\nWe collected {len(offsprings)} offsprings.\n")

    # -- stage 4: mutate (finetune) and save as the next generation -----
    gen_dir = paths.generation_dir(gen)
    mutated_accs = []
    for model_name in list(offsprings.keys()):
        model = offsprings.pop(model_name)
        print(f"Processing model: {model_name}")

        model = model.to(device)
        mutated_model, acc = finetune_merged_model(
            model, trainloader, testloader,
            epochs=cfg.mutation_epochs, lr=cfg.mutation_lr,
            updates_per_epoch=cfg.updates_per_epoch,
        )
        mutated_accs.append(acc)

        with mapping_guard(cfg.mapping_file):
            hash_id = encode_labels(model_name, mapping_file=cfg.mapping_file)

        save_dir = paths.offspring_dir(gen, hash_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"resnet20x{cfg.model_width}_v{len(os.listdir(save_dir))}.pth.tar"
        )
        save_offspring(mutated_model, save_path)
        print(f"Saved {model_name} -> {save_path}")

        mutated_model.cpu()
        del mutated_model, model
        gc.collect()
        torch.cuda.empty_cache()

    stats = {
        "generation": gen,
        "population": len(models_ids),
        "pairs": len(pairs) // 2,
        "loners": len(loners),
        "offspring": len(mutated_accs),
        "mutated_acc_mean": (
            sum(mutated_accs) / len(mutated_accs) if mutated_accs else None
        ),
        "csv": csv_file,
    }
    return stats, gen_dir


def run_evolution(cfg, paths):
    """Run `cfg.generations` generations starting from the pretrained population."""
    set_all_seeds(cfg.seed)

    # Make every registered dataset visible to utils.prepare_data before it
    # resolves raw_config['dataset']['name'].
    register_legacy_configs()

    raw_config = get_config_from_name(cfg.config_name, device=cfg.device)
    raw_config["model"]["dir"] = paths.population_dir
    # The config file names a dataset; --dataset wins, so one evolution config
    # serves every dataset (the name keys into datasets/configs.py).
    raw_config["dataset"]["name"] = cfg.spec.evo_name

    trainloader, testloader = get_full_loaders(
        cfg.dataset,
        batch_size=cfg.pretrain_batch_size,
        num_workers=cfg.pretrain_num_workers,
        seed=cfg.seed,
    )

    history = []
    t0 = time.time()
    for g in range(cfg.generations):
        gen = g + 1 + cfg.start_from
        stats, next_dir = run_generation(
            gen, cfg, paths, raw_config, trainloader, testloader
        )
        history.append(stats)
        raw_config["model"]["dir"] = next_dir

        with open(paths.run_json, "w") as f:
            json.dump(
                {"config": cfg.__dict__, "history": history}, f, indent=2, default=str
            )

    print(f"\n[evolution] {cfg.generations} generations in {time.time() - t0:.1f}s")
    return history


def print_history(history):
    print()
    print(f"{'Gen':>4} | {'Pop':>4} | {'Pairs':>5} | {'Loners':>6} | {'Offspring':>9} | {'Mutated acc':>11}")
    print("-" * 60)
    for h in history:
        acc = "-" if h["mutated_acc_mean"] is None else f"{h['mutated_acc_mean']:.4f}"
        print(
            f"{h['generation']:>4} | {h['population']:>4} | {h['pairs']:>5} | "
            f"{h['loners']:>6} | {h['offspring']:>9} | {acc:>11}"
        )
