# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TailSFT-style coverage changes between evaluation checkpoints.

The diagnostic in issue #8876 estimates each problem's pass@1 from repeated binary
attempts, transforms that estimate to pass@k under independent sampling, and compares
checkpoint pairs only on problems whose base-model pass@k lies in a fixed open interval.

This module contains only the reduction. Producing repeated samples and loading them
from an evaluation archive remain responsibilities of the evaluation runner and archive
adapter respectively.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemAttempts:
    """Binary outcomes summarized for one problem at one checkpoint."""

    successes: int
    attempts: int

    def __post_init__(self) -> None:
        if isinstance(self.successes, bool) or not isinstance(self.successes, int):
            raise TypeError("successes must be an integer")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise TypeError("attempts must be an integer")
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")
        if not 0 <= self.successes <= self.attempts:
            raise ValueError("successes must be between zero and attempts")

    @property
    def pass_at_1(self) -> float:
        """The per-attempt empirical success rate."""
        return self.successes / self.attempts


@dataclass(frozen=True)
class CoverageRatio:
    """Loss and gain over the base-reachable problem set for one transition."""

    reachable_problem_ids: tuple[str, ...]
    loss: float
    gain: float
    ratio: float | None
    mean_pass_at_k_change: float | None


def pass_at_k(pass_at_1: float, k: int) -> float:
    """Transform a per-attempt success probability to independent-sampling pass@k.

    This is the ``1 - (1 - p)^k`` plug-in transform specified by issue #8876, not
    the finite-sample unbiased pass@k estimator.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if isinstance(pass_at_1, bool) or not isinstance(pass_at_1, (int, float)):
        raise TypeError("pass_at_1 must be a number")
    if not math.isfinite(pass_at_1) or not 0.0 <= pass_at_1 <= 1.0:
        raise ValueError("pass_at_1 must be finite and between zero and one")
    return 1.0 - (1.0 - pass_at_1) ** k


def tail_sft_coverage_ratio(
    base: Mapping[str, ProblemAttempts],
    before: Mapping[str, ProblemAttempts],
    after: Mapping[str, ProblemAttempts],
    *,
    k: int = 16,
    expected_attempts: int = 16,
    reachable_lower: float = 0.05,
    reachable_upper: float = 0.95,
) -> CoverageRatio:
    """Compare one checkpoint transition on the base-reachable problem set.

    The reachable interval is open. Every supplied problem must have the fixed number
    of attempts so a transition cannot silently mix evaluation budgets. ``ratio`` is
    ``None`` when gain is zero, and the mean change is ``None`` when the reachable set
    is empty.
    """
    if isinstance(expected_attempts, bool) or not isinstance(expected_attempts, int) or expected_attempts <= 0:
        raise ValueError("expected_attempts must be a positive integer")
    if not 0.0 <= reachable_lower < reachable_upper <= 1.0:
        raise ValueError("reachable bounds must satisfy 0 <= lower < upper <= 1")

    for checkpoint, problems in (("base", base), ("before", before), ("after", after)):
        mismatched = sorted(problem_id for problem_id, value in problems.items() if value.attempts != expected_attempts)
        if mismatched:
            raise ValueError(f"{checkpoint} problems do not have exactly {expected_attempts} attempts: {mismatched}")

    reachable = tuple(
        sorted(
            problem_id
            for problem_id, attempts in base.items()
            if reachable_lower < pass_at_k(attempts.pass_at_1, k) < reachable_upper
        )
    )
    missing_before = sorted(set(reachable).difference(before))
    missing_after = sorted(set(reachable).difference(after))
    if missing_before or missing_after:
        raise ValueError(
            "base-reachable problems are missing from a transition: " f"before={missing_before}, after={missing_after}"
        )

    changes = [
        pass_at_k(after[problem_id].pass_at_1, k) - pass_at_k(before[problem_id].pass_at_1, k)
        for problem_id in reachable
    ]
    loss = math.fsum(max(-change, 0.0) for change in changes)
    gain = math.fsum(max(change, 0.0) for change in changes)
    return CoverageRatio(
        reachable_problem_ids=reachable,
        loss=loss,
        gain=gain,
        ratio=loss / gain if gain > 0.0 else None,
        mean_pass_at_k_change=math.fsum(changes) / len(changes) if changes else None,
    )
