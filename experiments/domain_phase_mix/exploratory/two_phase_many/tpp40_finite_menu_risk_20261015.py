# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Finite-action Gaussian-pair Bayes risk; no surrogate fitting or policy search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.special import log_ndtr, logsumexp


@dataclass(frozen=True)
class Interval:
    action: int
    lower: Fraction
    upper: Fraction | None


@dataclass(frozen=True)
class MenuRisk:
    bayes_regret: float
    log_bayes_regret: float
    overlap_bound: float
    log_overlap_bound: float
    compromise_sum_regret: float
    optimal_actions_zero: tuple[int, ...]
    optimal_actions_one: tuple[int, ...]
    shared_optimum: bool
    intervals: tuple[Interval, ...]
    interval_log_probabilities: tuple[tuple[float, float], ...]


def lower_envelope(intercepts: list[Fraction], slopes: list[Fraction]) -> tuple[Interval, ...]:
    """Return exact-rational minimizing lines on the positive likelihood-ratio axis."""
    if len(intercepts) != len(slopes) or not intercepts:
        raise ValueError("The two finite nonempty action lists must match")
    if any(value < 0 for value in intercepts + slopes):
        raise ValueError("Regrets must be nonnegative")
    ordered = sorted(range(len(slopes)), key=lambda i: (-slopes[i], intercepts[i], i))
    hull: list[tuple[int, Fraction | None]] = []
    previous_slope: Fraction | None = None
    for action in ordered:
        slope = slopes[action]
        if previous_slope is not None and slope == previous_slope:
            continue
        previous_slope = slope
        crossing: Fraction | None = None
        while hull:
            prior, prior_start = hull[-1]
            crossing = (intercepts[action] - intercepts[prior]) / (slopes[prior] - slope)
            if prior_start is None or crossing > prior_start:
                break
            hull.pop()
        hull.append((action, crossing if hull else None))
    intervals = []
    for position, (action, start) in enumerate(hull):
        end = hull[position + 1][1] if position + 1 < len(hull) else None
        if end is not None and end <= 0:
            continue
        lower = max(Fraction(0), start) if start is not None else Fraction(0)
        intervals.append(Interval(action, lower, end))
    return tuple(intervals)


def log_fraction(value: Fraction) -> float:
    if value < 0:
        raise ValueError("A negative fraction has no real logarithm")
    if value == 0:
        return -math.inf
    if Fraction(1, 2) <= value <= 2:
        # Subtracting the integer logarithms can erase a breakpoint close to one.
        return math.log1p(float(value - 1))
    return math.log(value.numerator) - math.log(value.denominator)


def log_normal_interval(lower: float, upper: float) -> float:
    """Evaluate standard-normal interval mass without subtracting two rounded CDFs."""
    if math.isnan(lower) or math.isnan(upper) or lower > upper:
        raise ValueError("Normal interval endpoints must be ordered")
    if lower == upper:
        return -math.inf
    if lower >= 0:
        large, small = float(log_ndtr(-lower)), float(log_ndtr(-upper))
    else:
        large, small = float(log_ndtr(upper)), float(log_ndtr(lower))
    if small == -math.inf:
        return large
    if small == large:
        raise ArithmeticError("Positive Gaussian interval is below log-CDF resolution")
    return large + math.log(-math.expm1(small - large))


def finite_menu_risk(truth: np.ndarray, alternative: np.ndarray, distance: float) -> MenuRisk:
    """Compute the equal-prior Bayes regret and overlap lower bound for a fixed pair.

    Action values are interpreted as their exact supplied floating-point numbers.
    Envelope breakpoints are rational; Gaussian probabilities remain floating point.
    The action menu is part of the decision problem, not a continuum certificate.
    """
    if truth.ndim != 1 or truth.shape != alternative.shape or len(truth) == 0:
        raise ValueError("Expected two matching nonempty action vectors")
    if not np.isfinite(truth).all() or not np.isfinite(alternative).all():
        raise ValueError("Action means must be finite")
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Gaussian distance must be finite and nonnegative")
    means = [[Fraction.from_float(float(value)) for value in row] for row in (truth, alternative)]
    regrets = [[value - min(row) for value in row] for row in means]
    optimal = [tuple(i for i, value in enumerate(row) if value == 0) for row in regrets]
    compromise = min(a + b for a, b in zip(*regrets, strict=True))
    log_overlap = log_fraction(compromise) + float(log_ndtr(-distance / 2))
    envelope = lower_envelope(*regrets)
    probabilities: list[tuple[float, float]] = []
    terms = []
    if distance == 0:
        log_risk = log_fraction(compromise) - math.log(2)
    else:
        for interval in envelope:
            lower = log_fraction(interval.lower)
            upper = math.inf if interval.upper is None else log_fraction(interval.upper)
            standardized = [
                (lower / distance + shift, upper / distance + shift) for shift in (distance / 2, -distance / 2)
            ]
            if lower >= upper or any(lo >= hi for lo, hi in standardized):
                raise ArithmeticError("Positive rational interval coalesced during Gaussian standardization")
            log_probability_zero, log_probability_one = [log_normal_interval(lo, hi) for lo, hi in standardized]
            probabilities.append((log_probability_zero, log_probability_one))
            for values, log_probability in zip(regrets, probabilities[-1], strict=True):
                terms.append(log_fraction(values[interval.action]) + log_probability - math.log(2))
        log_risk = float(logsumexp(terms))
    return MenuRisk(
        bayes_regret=math.exp(log_risk),
        log_bayes_regret=log_risk,
        overlap_bound=math.exp(log_overlap),
        log_overlap_bound=log_overlap,
        compromise_sum_regret=float(compromise),
        optimal_actions_zero=optimal[0],
        optimal_actions_one=optimal[1],
        shared_optimum=bool(set(optimal[0]) & set(optimal[1])),
        intervals=envelope,
        interval_log_probabilities=tuple(probabilities),
    )
