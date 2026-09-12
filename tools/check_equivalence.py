"""Check that sesil/* reproduces the legacy scripts' behaviour.

Covers the deterministic, model-free logic -- scoring, mate selection, offspring
keying, name normalisation -- by running the legacy function and the new one on
identical inputs under identical seeds and diffing the results.

What this does NOT cover: the merge and evaluation stages, which need real
checkpoints and a GPU. Those were carried over verbatim (see the module
docstrings in sesil/merging.py and sesil/evaluation.py); to check them, run one
generation each way and diff the CSVs.

    python tools/check_equivalence.py
"""

import os
import random
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "training_scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import evolutionary_wavg_training as legacy  # noqa: E402
from sesil import mating as new_mating  # noqa: E402
from sesil import merging as new_merging  # noqa: E402
from sesil import evaluation as new_eval  # noqa: E402


class Cfg:
    """Stand-in for RunConfig with the legacy defaults."""

    mate_mode = "threshold"
    mate_threshold = 0.5
    mate_weight_extra = 1.0
    mate_weight_common = 0.1
    mate_max_retries = 100


def fake_population(n=10, seed=0):
    rng = random.Random(seed)
    pop = []
    for i in range(n):
        classes = rng.sample(range(10), 3)
        per_class = [0.0] * 10
        for c in classes:
            per_class[c] = rng.uniform(0.6, 0.99)
        pop.append(
            {
                "Model Name": "_".join(str(c) for c in classes),
                "Per Class": per_class,
                "Joint": sum(per_class) / 10,
                "Per Task Avg": sum(per_class[c] for c in classes) / 3,
            }
        )
    return pop


def seeded(fn, *a, seed=0, **kw):
    random.seed(seed)
    np.random.seed(seed)
    return fn(*a, **kw)


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        legacy: {want}")
        print(f"        new   : {got}")
    return ok


def main():
    results = []
    pop = fake_population()

    print("sort_model_name_unique")
    for s in ["8_7_0", "2_9_7", "0_0_1_9", "5"]:
        results.append(
            check(f"  {s}", new_eval.sort_model_name_unique(s),
                  legacy.sort_model_name_unique(s))
        )

    print("mating_score")
    fa, fb = pop[0]["Per Class"], pop[1]["Per Class"]
    results.append(
        check("  threshold mode", new_mating.mating_score(fa, fb),
              legacy.mating_score(fa, fb))
    )
    results.append(
        check("  soft mode", new_mating.mating_score(fa, fb, mode="soft"),
              legacy.mating_score(fa, fb, mode="soft"))
    )

    print("build_score_matrix")
    results.append(
        check("  full matrix",
              new_mating.build_score_matrix(pop, mode="threshold", threshold=0.5),
              legacy.build_score_matrix(pop, mode="threshold", threshold=0.5))
    )

    print("bidirectional_selection (seeded)")
    for seed in (0, 1, 7):
        scores = legacy.build_score_matrix(pop, mode="threshold", threshold=0.5)
        want = seeded(legacy.bidirectional_selection, scores, seed=seed)
        got = seeded(new_mating.bidirectional_selection, scores, seed=seed)
        results.append(check(f"  seed={seed}", got, want))

    print("mate_with_population_info (seeded, end to end)")
    for seed in (0, 3):
        want = seeded(legacy.mate_with_population_info, pop, seed=seed)
        got = seeded(new_mating.mate_with_population_info, pop, Cfg(), seed=seed)
        results.append(check(f"  seed={seed}", got, want))

    print("build_unique_key")
    for labels in (["0", "1", "2"], ["0", "1", "2", "1"], ["5", "5", "5"]):
        seen_a, seen_b = set(), set()
        # feed the same key twice to exercise the rotation/suffix path
        want = [legacy.build_unique_key(labels, seen_a) for _ in range(4)]
        got = [new_merging.build_unique_key(labels, seen_b) for _ in range(4)]
        results.append(check(f"  {labels}", got, want))

    print()
    n_pass, n = sum(results), len(results)
    print(f"{n_pass}/{n} checks passed")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
