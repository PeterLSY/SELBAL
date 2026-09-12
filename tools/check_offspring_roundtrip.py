"""Regression test for the offspring save/load round-trip.

Catches the class of bug that killed exp1 at generation 2: a merged offspring
saved with the wrong state_dict layout finishes its own generation fine and
only explodes when the NEXT generation tries to load it. A single-generation
smoke test cannot see this -- the round-trip has to be checked explicitly.

Builds one real merged offspring, saves it both ways, and asserts:
  * sesil.pipeline.save_offspring  -> resnet20 loads it
  * utils.save_model               -> resnet20 REJECTS it (the original bug)

    python tools/check_offspring_roundtrip.py [--population DIR]
"""

import argparse
import os
import sys
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from model_merger import ModelMerge  # noqa: E402
from models.resnets import resnet20  # noqa: E402
from sesil.pipeline import save_offspring  # noqa: E402
from utils import (  # noqa: E402
    get_config_from_name,
    get_merging_fn,
    inject_pair,
    prepare_experiment_config,
    reset_bn_stats,
    save_model as utils_save_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="runs_weak/seed0/population")
    ap.add_argument("--merging-fn", default="match_tensors_permute")
    ap.add_argument("--out", default=None, help="scratch dir for the test files")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or os.path.join(REPO_ROOT, ".roundtrip_tmp")
    os.makedirs(out, exist_ok=True)

    members = sorted(
        d for d in os.listdir(args.population)
        if os.path.isdir(os.path.join(args.population, d))
    )
    if len(members) < 2:
        sys.exit(f"need >=2 members in {args.population}, found {len(members)}")
    pair = (members[0], members[1])
    print(f"pair: {pair}")

    raw = get_config_from_name("cifar_evolution_resnet20", device=device)
    raw["model"]["dir"] = args.population
    raw = inject_pair(raw, pair)
    config = prepare_experiment_config(raw)

    train_loader = config["data"]["train"]["full"]
    bases = [reset_bn_stats(b, train_loader) for b in config["models"]["bases"]]
    Grapher = config["graph"]
    graphs = [Grapher(deepcopy(b)).graphify() for b in bases]

    print("merging (this is the slow part) ...")
    Merge = ModelMerge(*graphs, device=device)
    Merge.transform(
        deepcopy(config["models"]["new"]),
        train_loader,
        transform_fn=get_merging_fn(args.merging_fn),
        metric_classes=config["metric_fns"],
        stop_at=21,
        a=0.0001,
        b=0.075,
    )
    reset_bn_stats(Merge, train_loader)
    print(f"merged: type={type(Merge).__name__}, "
          f"has head_models={hasattr(Merge, 'head_models')}")

    ok_path = os.path.join(out, "via_save_offspring.pth.tar")
    bad_path = os.path.join(out, "via_utils_save_model.pth.tar")
    save_offspring(Merge, ok_path)
    utils_save_model(Merge, bad_path)

    results = []

    # -- the fix: must load ------------------------------------------------
    sd = torch.load(ok_path, map_location="cpu")
    print(f"\nsave_offspring    -> {len(sd)} keys, first: {list(sd)[0]}")
    try:
        resnet20(w=4, num_classes=10).load_state_dict(sd)
        print("  PASS  resnet20 loads it")
        results.append(True)
    except RuntimeError as e:
        print(f"  FAIL  resnet20 rejected it: {str(e)[:120]}")
        results.append(False)

    # -- the bug: must NOT load -------------------------------------------
    sd_bad = torch.load(bad_path, map_location="cpu")
    print(f"\nutils.save_model  -> {len(sd_bad)} keys, first: {list(sd_bad)[0]}")
    try:
        resnet20(w=4, num_classes=10).load_state_dict(sd_bad)
        print("  FAIL  loaded unexpectedly -- ModelMerge no longer wraps? "
              "re-check save_offspring")
        results.append(False)
    except RuntimeError:
        print("  PASS  rejected, as expected (this was the exp1 gen-2 crash)")
        results.append(True)

    # -- loner path: plain resnet must round-trip too ----------------------
    plain = resnet20(w=4, num_classes=10)
    plain_path = os.path.join(out, "plain.pth.tar")
    save_offspring(plain, plain_path)
    try:
        resnet20(w=4, num_classes=10).load_state_dict(
            torch.load(plain_path, map_location="cpu")
        )
        print("\nloner path (plain resnet20)")
        print("  PASS  round-trips")
        results.append(True)
    except RuntimeError as e:
        print(f"\n  FAIL  loner path broken: {str(e)[:120]}")
        results.append(False)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
