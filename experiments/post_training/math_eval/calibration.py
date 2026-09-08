# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Empirical evaluation noise and explicitly assumed future-arm MDE sensitivity."""

import math
import statistics
from itertools import product


def _panel(panel, samples):
    seeds = sorted(panel)
    if len(seeds) < 2 or any(type(seed) is not int for seed in seeds):
        raise ValueError("Calibration requires at least two declared training seeds")
    questions = sorted(panel[seeds[0]])
    if len(questions) < 2:
        raise ValueError("Calibration requires at least two questions")
    for seed in seeds:
        if sorted(panel[seed]) != questions:
            raise ValueError("Training seeds have different question membership")
        for values in panel[seed].values():
            if len(values) != samples or any(value not in (0, 1) for value in values):
                raise ValueError("Every question needs exactly K binary completed outcomes")
    return seeds, questions


def stochastic_variances(by_k):
    """Summarize audited K=1/4/8 panels without assuming a variance decomposition.

    Each panel is training seed -> question hash -> response outcomes. Distinct
    K panels must have their own generation provenance checked by the caller.
    Question-mean variance includes finite-response noise. Aggregate seed variance
    also includes evaluation noise; these are not disjoint components to add.
    """
    if set(by_k) != {1, 4, 8}:
        raise ValueError("Calibration needs the declared K=1,4,8 panels")
    identity = None
    result = {}
    for k in (1, 4, 8):
        panel = by_k[k]
        seeds, questions = _panel(panel, k)
        if identity is not None and identity != (seeds, questions):
            raise ValueError("K panels have different training seeds or questions")
        identity = seeds, questions
        means = {seed: [statistics.mean(panel[seed][q]) for q in questions] for seed in seeds}
        aggregate = {seed: statistics.mean(values) for seed, values in means.items()}
        result[k] = {
            "questions": len(questions),
            "training_seeds": seeds,
            "seed_degrees_of_freedom": len(seeds) - 1,
            "per_seed_mean": aggregate,
            "per_seed_question_mean_variance": {seed: statistics.variance(values) for seed, values in means.items()},
            "per_seed_within_question_variance": {
                seed: None if k == 1 else statistics.mean(statistics.variance(panel[seed][q]) for q in questions)
                for seed in seeds
            },
            "aggregate_training_seed_sample_variance": statistics.variance(aggregate.values()),
            "question_mean_covariance_between_observed_seeds": {
                str(a): {str(b): statistics.covariance(means[a], means[b]) for b in seeds} for a in seeds
            },
            "variance_scope": "empirical; question and aggregate-seed variances include sampling noise",
            "future_seed_variance_certified": False,
        }
    return result


def greedy_repeat_variances(panel):
    """Describe five greedy passes; identical observations do not prove zero noise."""
    seeds, questions = _panel(panel, 5)
    result = {}
    for seed in seeds:
        pass_means = [statistics.mean(panel[seed][q][repeat] for q in questions) for repeat in range(5)]
        result[seed] = {
            "questions": len(questions),
            "pass_means": pass_means,
            "pass_mean_sample_variance": statistics.variance(pass_means),
            "questions_with_changed_outcome": sum(len(set(panel[seed][q])) > 1 for q in questions),
            "mean_pairwise_outcome_disagreement": statistics.mean(
                panel[seed][q][a] != panel[seed][q][b] for q in questions for a in range(5) for b in range(a)
            ),
            "zero_population_noise_certified": False,
        }
    return result


def assumed_mde(
    *,
    question_variances,
    run_variances,
    rho_question,
    rho_run,
    questions,
    training_seeds,
    family_size,
    familywise_alpha=0.05,
    power=0.8,
):
    """Normal planning approximation for caller-supplied disjoint components.

    The caller must justify the decomposition and marginal variance estimates.
    Raw outputs from stochastic_variances are not disjoint components. Covariance
    is assumed, never inferred for future arms from two observed anchor seeds.
    rho=0 is not guaranteed conservative; rho=-1 maximizes pairwise variance for
    fixed marginals. Bonferroni planning does not relabel intervals as Holm.
    """
    if any(type(value) is not int or value < 1 for value in (questions, training_seeds, family_size)):
        raise ValueError("Positive integer design sizes and comparison family are required")
    if not 0 < familywise_alpha < 1 or not 0.5 < power < 1:
        raise ValueError("Invalid familywise alpha or power")
    if any(not math.isfinite(rho) or not -1 <= rho <= 1 for rho in (rho_question, rho_run)):
        raise ValueError("Assumed correlations must be between -1 and 1")

    def difference_variance(values, rho):
        if len(values) != 2 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Supply two finite nonnegative marginal variances")
        a, b = values
        return max(0.0, a + b - 2 * rho * math.sqrt(a * b))

    question_part = difference_variance(question_variances, rho_question) / questions
    run_part = difference_variance(run_variances, rho_run) / training_seeds
    if question_part + run_part <= 0:
        raise ValueError("Zero assumed or observed variance cannot certify power or zero MDE")
    alpha = familywise_alpha / family_size
    normal = statistics.NormalDist()
    multiplier = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return {
        "mde": multiplier * math.sqrt(question_part + run_part),
        "question_difference_variance_of_mean": question_part,
        "run_difference_variance_of_mean": run_part,
        "infinite_question_mde_floor": multiplier * math.sqrt(run_part),
        "assumed_rho_question": rho_question,
        "assumed_rho_run": rho_run,
        "family_size": family_size,
        "planning_alpha": alpha,
        "power": power,
        "method": "normal approximation, assumed disjoint variances/covariances, Bonferroni planning",
        "power_certified": False,
    }


COMPONENTS = ("seed", "question", "interaction", "draw")
CORRELATION_GRID = (-1.0, 0.0, 0.5, 0.9)


def crossed_variance_components(panel, samples):
    """Estimate disjoint second moments in a balanced seed/question/draw model.

    Seed, question, interaction and draw terms are mutually uncorrelated working
    components. Draw residuals are conditionally uncorrelated, not merely
    exchangeable. Binary outcomes need not be Gaussian. These moment estimates
    and their nonnegative planning truncations are not variance upper bounds.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("Positive integer K required")
    seeds, questions = _panel(panel, samples)
    ns, nq, k = len(seeds), len(questions), samples
    cells = {seed: {q: statistics.mean(panel[seed][q]) for q in questions} for seed in seeds}
    seed_means = {seed: statistics.mean(cells[seed].values()) for seed in seeds}
    question_means = {q: statistics.mean(cells[seed][q] for seed in seeds) for q in questions}
    grand = statistics.mean(seed_means.values())
    ss = {
        "seed": nq * k * sum((value - grand) ** 2 for value in seed_means.values()),
        "question": ns * k * sum((value - grand) ** 2 for value in question_means.values()),
        "interaction": (
            k
            * sum(
                (cells[seed][q] - seed_means[seed] - question_means[q] + grand) ** 2 for seed in seeds for q in questions
            )
        ),
        "draw": sum((value - cells[seed][q]) ** 2 for seed in seeds for q in questions for value in panel[seed][q]),
    }
    dfs = {"seed": ns - 1, "question": nq - 1, "interaction": (ns - 1) * (nq - 1), "draw": ns * nq * (k - 1)}
    ms = {key: ss[key] / dfs[key] if dfs[key] else None for key in COMPONENTS}
    raw = {
        "seed": (ms["seed"] - ms["interaction"]) / (nq * k),
        "question": (ms["question"] - ms["interaction"]) / (ns * k),
        "interaction": None if k == 1 else (ms["interaction"] - ms["draw"]) / k,
        "draw": ms["draw"],
    }
    return {
        "questions": nq,
        "training_seeds": seeds,
        "samples": k,
        "mean": grand,
        "sums_of_squares": ss,
        "degrees_of_freedom": dfs,
        "mean_squares": ms,
        "raw_moments": raw,
        "planning_nonnegative_moments": {key: None if value is None else max(0.0, value) for key, value in raw.items()},
        "interaction_plus_draw_when_k1": ms["interaction"] if k == 1 else None,
        "assumptions": "mutually uncorrelated crossed components; conditionally uncorrelated draw residuals",
        "native_engine_shared_seed_uncertainty_estimated": False,
        "future_seed_variance_certified": False,
        "truncation_is_conservative": False,
    }


def crossed_mde_sensitivity(
    *, component_pairs, questions, training_seeds, samples, family_size, familywise_alpha=0.05, power=0.8
):
    """Persist all 256 assumed component-covariance scenarios for one contrast.

    Caller supplies explicitly identified nonnegative marginal moments for both
    future arms. Their applicability is a planning assumption. The contrast family
    size must be prespecified; this function never derives it from the panel size.
    """
    if set(component_pairs) != set(COMPONENTS):
        raise ValueError("All four identified component pairs are required")
    if any(type(value) is not int or value < 1 for value in (questions, training_seeds, samples, family_size)):
        raise ValueError("Positive integer design and prespecified family sizes required")
    if not 0 < familywise_alpha < 1 or not 0.5 < power < 1:
        raise ValueError("Invalid alpha or power")
    for pair in component_pairs.values():
        if len(pair) != 2 or any(value is None or not math.isfinite(value) or value < 0 for value in pair):
            raise ValueError("Finite identified nonnegative component pairs required")
    divisors = {
        "seed": training_seeds,
        "question": questions,
        "interaction": training_seeds * questions,
        "draw": training_seeds * questions * samples,
    }
    alpha = familywise_alpha / family_size
    multiplier = statistics.NormalDist().inv_cdf(1 - alpha / 2) + statistics.NormalDist().inv_cdf(power)
    rows = []
    for correlations in product(CORRELATION_GRID, repeat=4):
        rhos = dict(zip(COMPONENTS, correlations, strict=True))
        parts = {}
        for key, (a, b) in component_pairs.items():
            parts[key] = max(0.0, a + b - 2 * rhos[key] * math.sqrt(a * b)) / divisors[key]
        variance = sum(parts.values())
        rows.append(
            {
                "assumed_correlations": rhos,
                "variance_of_mean_components": parts,
                "mde": multiplier * math.sqrt(variance) if variance > 0 else None,
                "zero_variance_uninformative": variance == 0,
                "power_certified": False,
            }
        )
    return {
        "family_size": family_size,
        "planning_alpha": alpha,
        "power": power,
        "primary_rho0_assumption": next(row for row in rows if set(row["assumed_correlations"].values()) == {0.0}),
        "shared_rho_scenarios": [row for row in rows if len(set(row["assumed_correlations"].values())) == 1],
        "all_component_scenarios": rows,
        "component_divisors": divisors,
        "scope": "normal planning approximation; no covariance, power or future-seed certification",
    }
