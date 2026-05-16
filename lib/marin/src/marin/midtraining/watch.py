# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Startup-proof predicates and post-launch monitoring helpers.

Each mode owns its expected and forbidden startup lines (see
:meth:`CptMode.expected_startup_lines` etc.). The babysitter scans the
TPU child's stdout and refuses any forbidden line.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from marin.midtraining.spec import ResolvedMidtrainSpec

_W_AND_B_STEP_REGRESSION_PATTERN = re.compile(r"Step (\d+) is less than current W&B step (\d+)")
_RESUME_STEP_PATTERN = re.compile(r"Resuming training from step (\d+)")


@dataclass(frozen=True)
class StartupProof:
    """Outcome of scanning a startup-log buffer."""

    matched_expected: tuple[str, ...]
    missing_expected: tuple[str, ...]
    forbidden_seen: tuple[str, ...]
    detected_step: int | None
    expected_min_step: int | None

    @property
    def healthy(self) -> bool:
        if self.forbidden_seen or self.missing_expected:
            return False
        if self.expected_min_step is not None:
            if self.detected_step is None:
                return False
            if self.detected_step < self.expected_min_step:
                return False
        return True


def evaluate_startup(resolved: ResolvedMidtrainSpec, log_lines: Iterable[str]) -> StartupProof:
    """Scan ``log_lines`` and return a :class:`StartupProof` for the resolved spec."""
    mode = resolved.spec.mode
    expected = mode.expected_startup_lines()
    forbidden = mode.forbidden_startup_lines()
    seen_text = "\n".join(log_lines)

    matched = tuple(line for line in expected if line in seen_text)
    missing = tuple(line for line in expected if line not in seen_text)
    forbidden_seen = tuple(line for line in forbidden if line in seen_text)

    detected_step = _extract_resume_step(seen_text)
    expected_min_step = _expected_min_step(resolved)

    if _W_AND_B_STEP_REGRESSION_PATTERN.search(seen_text):
        forbidden_seen = (*forbidden_seen, "W&B step regression")

    return StartupProof(
        matched_expected=matched,
        missing_expected=missing,
        forbidden_seen=forbidden_seen,
        detected_step=detected_step,
        expected_min_step=expected_min_step,
    )


def _extract_resume_step(text: str) -> int | None:
    match = _RESUME_STEP_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _expected_min_step(resolved: ResolvedMidtrainSpec) -> int | None:
    spec = resolved.spec
    if spec.expected_min_step is not None:
        return spec.expected_min_step
    return spec.mode.expected_min_step()
