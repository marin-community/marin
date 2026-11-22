# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty bands and their trace-length budget contracts.

Bands place a task by its measured per-task solve rate; each band carries the sampling
budget its traces are generated under. The band names reuse the taskforge review
vocabulary (`model.review.BandOutcome`), but the edges here are the lane's own policy:
the taskforge defaults (`band.low: 0.25`, `band.high: 0.75`) govern task acceptance, not
trace budgets.
"""

from dataclasses import dataclass
from enum import StrEnum

EVAL_OUTPUT_CAP = 16_384
"""The generation cap every band respects: the eval context cap and the
``atlas-difficulty-v3`` hosted output budget. No shipped trace may exceed it."""


class Band(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNSOLVED = "unsolved"


@dataclass(frozen=True, slots=True)
class BandBudget:
    """The trace-length contract one band is generated under."""

    band: Band
    """Lower solve-rate edge of the band (inclusive), or ``None`` for the unsolved band."""
    solve_rate_min: float | None
    solve_rate_max: float | None
    """Upper solve-rate edge (exclusive); ``None`` means unbounded above."""
    target_tokens: tuple[int, int]
    """Expected assistant-token range of a kept trace; documentation and telemetry only."""
    cap_tokens: int
    """The per-request output cap. Every cap is at most ``EVAL_OUTPUT_CAP``."""
    attempts: int
    """How many traces to sample per task before routing to the graceful lane."""

    def __post_init__(self) -> None:
        if self.cap_tokens > EVAL_OUTPUT_CAP:
            raise ValueError(f"{self.band} cap {self.cap_tokens} exceeds the eval output cap {EVAL_OUTPUT_CAP}")
        if self.attempts < 1:
            raise ValueError(f"{self.band} must sample at least one attempt")


BAND_BUDGETS: tuple[BandBudget, ...] = (
    BandBudget(Band.EASY, 0.70, None, (500, 2_000), 4_096, 2),
    BandBudget(Band.MEDIUM, 0.30, 0.70, (3_000, 6_000), 8_192, 4),
    BandBudget(Band.HARD, None, 0.30, (6_000, 12_000), 16_384, 6),
    BandBudget(Band.UNSOLVED, 0.0, None, (6_000, 12_000), 16_384, 3),
)
"""The lane's band table. ``UNSOLVED`` is reached only through the graceful lane."""

BUDGET_BY_BAND: dict[Band, BandBudget] = {budget.band: budget for budget in BAND_BUDGETS}

BAND_ORDER: dict[Band, int] = {band: index for index, band in enumerate(Band)}
"""Canonical band order: easy, medium, hard, unsolved."""


def band_for(solve_rate: float) -> Band:
    """Place one per-task solve rate into its band."""
    if not 0.0 <= solve_rate <= 1.0:
        raise ValueError(f"Solve rate must be within [0, 1], got {solve_rate}")
    if solve_rate == 0.0:
        return Band.UNSOLVED
    if solve_rate >= 0.70:
        return Band.EASY
    if solve_rate >= 0.30:
        return Band.MEDIUM
    return Band.HARD


def budget_for(band: Band) -> BandBudget:
    """The budget contract of one band."""
    return BUDGET_BY_BAND[band]
