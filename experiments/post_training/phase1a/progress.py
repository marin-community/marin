# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Match checkpoint trajectories at confirmed upward crossings of reverse progress."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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

    The checkpoint after the upper bracket must lie within five percentage points
    of the level. A terminal crossing cannot be confirmed. Points must include
    measured baseline metrics if interpolation from step zero is needed; unknown
    baseline metrics must not be filled with zero.
    """
    if any(left.step >= right.step for left, right in zip(points, points[1:])):
        raise ValueError("progress points must have strictly increasing steps")
    crossings = []
    for lower, upper, confirmation in zip(points, points[1:], points[2:]):
        if not lower.progress < level <= upper.progress:
            continue
        if abs(confirmation.progress - level) > CONFIRMATION_DISTANCE + 1e-12:
            continue
        fraction = (level - lower.progress) / (upper.progress - lower.progress)
        metrics = {
            name: value + fraction * (upper.metrics[name] - value)
            for name, value in lower.metrics.items()
            if name in upper.metrics
        }
        crossings.append(Crossing(lower.step, upper.step, confirmation.step, fraction, metrics))
    return crossings
