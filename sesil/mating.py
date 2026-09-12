"""Stage 2 -- scoring and mate selection.

Carried over verbatim from evolutionary_wavg_training.py:205-377. The scoring
knobs that were hardcoded at the call site (wavg:369 passed
mode="threshold", threshold=0.5 and relied on mating_score's defaults for the
weights) are now parameters.
"""

import random

import numpy as np


def known_classes(fitness, threshold=0.5):
    """Return indices of classes this model knows (above threshold)."""
    return {i for i, acc in enumerate(fitness) if acc > threshold}


def mating_score(
    fitness_a,
    fitness_b,
    mode="threshold",
    threshold=0.5,
    weight_extra=1.0,
    weight_common=0.1,
):
    """How much A values B as a mate (directional)."""
    if mode == "threshold":
        known_a = known_classes(fitness_a, threshold)
        known_b = known_classes(fitness_b, threshold)

        extra_skills = known_b - known_a
        common_skills = known_a & known_b

        score = weight_extra * sum(fitness_b[i] for i in extra_skills)
        score += weight_common * sum(fitness_b[i] for i in common_skills)

    elif mode == "soft":
        score = 0.0
        for i, (acc_a, acc_b) in enumerate(zip(fitness_a, fitness_b)):
            if acc_b > 0:
                if acc_a < threshold:
                    score += weight_extra * acc_b
                else:
                    score += weight_common * acc_b
    else:
        raise ValueError("mode must be 'threshold' or 'soft'")

    return score


def build_score_matrix(models, **kwargs):
    """Matrix of directional mating scores."""
    scores = {}
    for model_a in models:
        fa = model_a["Per Class"]
        scores[model_a["Model Name"]] = {}
        for model_b in models:
            if model_a is model_b:
                continue
            fb = model_b["Per Class"]
            scores[model_a["Model Name"]][model_b["Model Name"]] = mating_score(
                fa, fb, **kwargs
            )
    return scores


def probabilistic_choice(score_dict):
    """Pick a mate from score_dict probabilistically."""
    if not score_dict:
        return None
    models = list(score_dict.keys())
    weights = np.array(list(score_dict.values()), dtype=float)
    if weights.sum() == 0:
        return random.choice(models)
    probs = weights / weights.sum()
    return np.random.choice(models, p=probs)


def bidirectional_selection(scores, max_retries=100, return_loners=True):
    """Bidirectional probabilistic mate selection, population size preserved.

    Each reciprocated pair is appended twice (two offspring per couple).
    """
    models = list(scores.keys())
    N = len(models)

    pairs = []
    paired = set()
    retries = 0

    while retries < max_retries:
        retries += 1
        choices = {m: probabilistic_choice(scores[m]) for m in models if m not in paired}

        for a, b in choices.items():
            if b is not None and choices.get(b) == a:
                if a not in paired and b not in paired:
                    pair = tuple(sorted((a, b)))
                    pairs.append(pair)
                    pairs.append(pair)
                    paired.update([a, b])

    loners = [m for m in models if m not in paired]

    assert 2 * (len(pairs) // 2) + len(loners) == N, (
        f"Population size mismatch: {2*(len(pairs)//2)+len(loners)} != {N}"
    )

    if return_loners:
        return pairs, loners
    return pairs


def mate_with_population_info(population_info, cfg):
    scores = build_score_matrix(
        population_info,
        mode=cfg.mate_mode,
        threshold=cfg.mate_threshold,
        weight_extra=cfg.mate_weight_extra,
        weight_common=cfg.mate_weight_common,
    )
    print("Scores:\n", scores)
    pairs, loners = bidirectional_selection(scores, max_retries=cfg.mate_max_retries)
    return pairs, loners
