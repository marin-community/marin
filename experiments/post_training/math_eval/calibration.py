# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Empirical evaluation noise and explicitly assumed future-arm MDE sensitivity."""

import math
import statistics


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
