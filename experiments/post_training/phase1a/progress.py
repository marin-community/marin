# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Match checkpoint trajectories at confirmed upward crossings of reverse progress."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise, product
from statistics import mean
from statistics import quantiles as quantiles_from_samples

CONFIRMATION_DISTANCE = 0.05


@dataclass(frozen=True)
class ProgressPoint:
    step: int
    progress: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class Crossing:
    lower_step: int
    upper_step: int
    confirmation_step: int
    fraction: float
    metrics: Mapping[str, float]


def confirmed_crossings(points: Sequence[ProgressPoint], level: float) -> list[Crossing]:
    """Interpolate upward crossings confirmed by the following scored checkpoint.

    The checkpoint after the upper bracket must not fall more than five percentage
    points below the level; continued progress above it confirms the crossing. The
    guard rejects transient spikes that fall back. A terminal crossing cannot be
    confirmed. Points must include
    measured baseline metrics if interpolation from step zero is needed; unknown
    baseline metrics must not be filled with zero.
    """
    if any(left.step >= right.step for left, right in pairwise(points)):
        raise ValueError("progress points must have strictly increasing steps")
    crossings = []
    for lower, upper, confirmation in zip(points, points[1:], points[2:], strict=False):
        if not lower.progress < level <= upper.progress:
            continue
        if confirmation.progress < level - CONFIRMATION_DISTANCE - 1e-12:
            continue
        fraction = (level - lower.progress) / (upper.progress - lower.progress)
        metrics = {
            name: value + fraction * (upper.metrics[name] - value)
            for name, value in lower.metrics.items()
            if name in upper.metrics
        }
        crossings.append(Crossing(lower.step, upper.step, confirmation.step, fraction, metrics))
    return crossings


def metrics_at_step(points: Sequence[ProgressPoint], step: float) -> dict[str, float]:
    """Interpolate measured metrics in optimizer steps, without extrapolating."""
    for point in points:
        if step == point.step:
            return dict(point.metrics)
    for lower, upper in pairwise(points):
        if lower.step < step < upper.step:
            fraction = (step - lower.step) / (upper.step - lower.step)
            return {
                name: value + fraction * (upper.metrics[name] - value)
                for name, value in lower.metrics.items()
                if name in upper.metrics
            }
    raise ValueError(f"step {step} is outside the measured trajectory")


@dataclass(frozen=True)
class Interference:
    difference: float
    single_task_gain: float
    gain_fraction: float | None


def interference_at_crossing(
    crossing: Crossing,
    joint: Sequence[ProgressPoint],
    control: Sequence[ProgressPoint],
    baselines: Mapping[str, float],
) -> dict[str, Interference]:
    """Compare joint and control fidelity at the anchor's matched optimizer budget.

    The anchor is the exposure-matched single-task run's reverse-progress crossing.
    Evaluate the joint and either single-task control at that same fractional step.
    Matching the joint separately on the affected task would force reverse-form
    interference to zero by construction. Baseline keys select the fidelity forms
    to report; drift and routing are read independently from the matched trajectories.
    """
    step = crossing.lower_step + crossing.fraction * (crossing.upper_step - crossing.lower_step)
    joint_metrics = metrics_at_step(joint, step)
    control_metrics = metrics_at_step(control, step)
    result = {}
    for name, baseline in baselines.items():
        difference = joint_metrics[name] - control_metrics[name]
        gain = baseline - control_metrics[name]
        result[name] = Interference(difference, gain, difference / gain if gain != 0 else None)
    return result


@dataclass(frozen=True)
class SeedInterval:
    mean: float
    lower: float
    upper: float
    seeds: int


def seed_bootstrap_interval(values: Sequence[float]) -> SeedInterval:
    """Exact 95% percentile interval from resampling the observed seeds.

    Enumerating all n**n draws avoids Monte Carlo variation for the three-seed
    cells. For paired method contrasts, pass per-seed differences so the bootstrap
    preserves each seed's shared prompt ordering across methods.
    """
    if len(values) < 2:
        raise ValueError("a seed bootstrap requires at least two seeds")
    samples = sorted(mean(sample) for sample in product(values, repeat=len(values)))
    quantiles = quantiles_from_samples(samples, n=40, method="inclusive")
    return SeedInterval(mean(values), quantiles[0], quantiles[-1], len(values))
