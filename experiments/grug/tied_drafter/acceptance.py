# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrated local progress forecasts and paired candidate decisions."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class ProgressForecast:
    loss_per_step: float
    upper_loss_per_step: float
    calibration_first_step: int
    calibration_last_step: int
    independent_groups: int


class DecisionReason(StrEnum):
    ACCEPT = "gain_covers_cost"
    UNCERTAIN = "gain_not_distinguishable_from_noise"
    COST = "gain_does_not_cover_cost"


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: DecisionReason
    mean_loss_gain: float
    standard_error: float
    lower_loss_gain: float
    estimated_equivalent_steps: float
    conservative_equivalent_steps: float
    required_equivalent_steps: float


def local_progress_forecast(curve: list[ProgressForecast], *, step: int, lookahead: int) -> ProgressForecast:
    """Use the current control segment and conservative slopes over the lookahead.

    A single earlier slope is unsuitable when ordinary progress decelerates.
    This function interpolates a measured control trajectory; it does not
    extrapolate beyond the available control window.
    """
    active = [
        part for part in curve if part.calibration_last_step > step and part.calibration_first_step < step + lookahead
    ]
    if not active or active[0].calibration_first_step > step or active[-1].calibration_last_step < step + lookahead:
        raise ValueError("Control curve does not cover the proposal lookahead")
    if any(a.calibration_last_step != b.calibration_first_step for a, b in pairwise(active)):
        raise ValueError("Control curve has a gap")
    return ProgressForecast(
        active[0].loss_per_step,
        max(part.upper_loss_per_step for part in active),
        active[0].calibration_first_step,
        active[-1].calibration_last_step,
        min(part.independent_groups for part in active),
    )


def fit_progress_forecast(steps: np.ndarray, group_losses: np.ndarray, *, confidence: float) -> ProgressForecast:
    """Fit a local slope per independent group, retaining each group's full time series.

    Calibration uses fixed groups across control checkpoints. The confidence
    bound reflects variation across groups, not an independence assumption for
    successive checkpoints. Curvature must be checked on held-out control steps.
    """
    steps = np.asarray(steps, dtype=np.float64)
    losses = np.asarray(group_losses, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] != len(steps) or losses.shape[1] < 2:
        raise ValueError("Need checkpoint-by-group losses with at least two independent groups")
    centered = steps - steps.mean()
    slopes = -(centered[:, None] * losses).sum(axis=0) / np.square(centered).sum()
    mean = float(slopes.mean())
    upper = mean + float(t.ppf(confidence, len(slopes) - 1) * slopes.std(ddof=1) / np.sqrt(len(slopes)))
    if not np.isfinite(upper) or mean <= 0:
        raise ValueError("Control calibration does not establish positive ordinary training progress")
    return ProgressForecast(mean, upper, int(steps.min()), int(steps.max()), len(slopes))


def grouped_loss(losses: np.ndarray, counts: np.ndarray, *, sequences_per_group: int) -> np.ndarray:
    """Aggregate adjacent sequences into weighted groups for uncertainty estimates."""
    losses = np.asarray(losses, dtype=np.float64).reshape(-1, sequences_per_group)
    counts = np.asarray(counts, dtype=np.float64).reshape(losses.shape)
    totals = counts.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every verification group must contain scored tokens")
    return (losses * counts).sum(axis=1) / totals


def assess_candidate(
    before: np.ndarray,
    after: np.ndarray,
    forecast: ProgressForecast,
    *,
    overhead_equivalent_steps: float,
    assumed_acceptance: float,
    repeatability_floor: float,
    confidence: float,
    candidate_scales: int,
) -> AcceptanceDecision:
    """Require conservative extra progress to cover cost per expected acceptance.

    Inputs are paired independent-group means. Bonferroni correction covers the
    predefined scale trials within this proposal. Forecast uncertainty is charged
    separately using the upper ordinary-progress slope.
    """
    gains = np.asarray(before, dtype=np.float64) - np.asarray(after, dtype=np.float64)
    if gains.ndim != 1 or gains.size < 2 or not np.all(np.isfinite(gains)):
        raise ValueError("Need finite paired group losses from at least two groups")
    mean = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(len(gains)))
    critical = t.ppf(1 - (1 - confidence) / candidate_scales, len(gains) - 1)
    lower = mean - max(float(critical * se), repeatability_floor)
    required = overhead_equivalent_steps / assumed_acceptance
    conservative_steps = lower / forecast.upper_loss_per_step
    if lower <= 0:
        reason = DecisionReason.UNCERTAIN
    elif conservative_steps <= required:
        reason = DecisionReason.COST
    else:
        reason = DecisionReason.ACCEPT
    return AcceptanceDecision(
        reason == DecisionReason.ACCEPT,
        reason,
        mean,
        se,
        lower,
        mean / forecast.loss_per_step,
        conservative_steps,
        required,
    )
