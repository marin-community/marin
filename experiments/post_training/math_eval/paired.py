# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired question statistics with explicit fixed-seed or seed-population inference."""

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from experiments.post_training.async_rl_audit import percentile


class SeedInference(StrEnum):
    FIXED = "fixed-observed-seeds"
    POPULATION = "resample-seeds-and-questions"


@dataclass(frozen=True)
class PairedEstimate:
    reference: float
    candidate: float
    delta: float
    paired_question_se: float
    question_correlation: float | None
    interval: tuple[float, float]
    questions: int
    seeds: tuple[int, ...]
    seed_deltas: tuple[float, ...]
    inference: SeedInference
    confidence: float
    repetitions: int


def bootstrap(
    reference: Mapping[int, Mapping[str, Sequence[float]]],
    candidate: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    seed: int,
    repetitions: int,
    inference: SeedInference,
    alpha: float = 0.05,
) -> PairedEstimate:
    """Pair on question hashes, averaging repeated responses within each seed.

    Population inference jointly resamples seed pairs and question clusters;
    fixed inference reproduces the predecessor's conditional-seed estimand.
    The displayed question SE is conditional on observed seeds in both modes.
    """
    seeds = tuple(sorted(reference))
    if not seeds or set(seeds) != set(candidate):
        raise ValueError("Arms require the same nonempty training seed set")
    if inference == SeedInference.POPULATION and len(seeds) < 2:
        raise ValueError("Population inference requires multiple independent training seeds")
    if repetitions < 100 or not 0 < alpha < 1:
        raise ValueError("Bootstrap needs at least 100 draws and alpha between zero and one")
    questions = sorted(reference[seeds[0]])
    if len(questions) < 2:
        raise ValueError("Paired inference requires at least two questions")
    means = []
    for arm in (reference, candidate):
        rows = []
        for run_seed in seeds:
            if set(arm[run_seed]) != set(questions):
                raise ValueError("Arms and seeds must contain identical question hashes")
            values = []
            for question in questions:
                samples = arm[run_seed][question]
                if not samples or not all(math.isfinite(value) for value in samples):
                    raise ValueError("Each question requires finite per-response scores")
                values.append(statistics.mean(samples))
            rows.append(values)
        means.append(rows)
    differences = [[b - a for a, b in zip(left, right, strict=True)] for left, right in zip(*means, strict=True)]
    by_question = [statistics.mean(values) for values in zip(*differences, strict=True)]
    count = len(questions)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        question_indices = rng.choices(range(count), k=count)
        if inference == SeedInference.FIXED:
            draws.append(sum(by_question[index] for index in question_indices) / count)
        else:
            seed_indices = rng.choices(range(len(seeds)), k=len(seeds))
            draws.append(sum(differences[s][q] for s in seed_indices for q in question_indices) / (len(seeds) * count))
    draws.sort()
    left, right = [[statistics.mean(values) for values in zip(*arm, strict=True)] for arm in means]
    correlation = (
        statistics.correlation(left, right) if statistics.pvariance(left) and statistics.pvariance(right) else None
    )
    return PairedEstimate(
        reference=statistics.mean(left),
        candidate=statistics.mean(right),
        delta=statistics.mean(by_question),
        paired_question_se=statistics.stdev(by_question) / math.sqrt(count),
        question_correlation=correlation,
        interval=(percentile(draws, alpha / 2), percentile(draws, 1 - alpha / 2)),
        questions=count,
        seeds=seeds,
        seed_deltas=tuple(statistics.mean(row) for row in differences),
        inference=inference,
        confidence=1 - alpha,
        repetitions=repetitions,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values over a declared comparison family."""
    if not p_values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("Provide finite p-values for a nonempty comparison family")
    previous = 0.0
    result = {}
    for rank, (name, p_value) in enumerate(sorted(p_values.items(), key=lambda item: (item[1], item[0]))):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_value))
        result[name] = previous
    return result


def required_questions(*, delta: float, difference_variance: float, alpha: float, power: float) -> int:
    """Normal approximation for paired-question power (variance includes sampling noise)."""
    if delta <= 0 or difference_variance < 0 or not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("Invalid power-design inputs")
    normal = statistics.NormalDist()
    value = (normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) ** 2 * difference_variance / delta**2
    return math.ceil(value)
