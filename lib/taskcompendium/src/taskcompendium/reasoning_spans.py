# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning spans in a model reply, and the grading rule for malformed ones.

A reply may hold one reasoning span before its answer, in any of the delimiter syntaxes below. A
closing marker with no opening marker closes a span that the chat template opened in the prompt.
"""

import re
from collections.abc import Sequence
from enum import StrEnum

from taskcompendium.grading_result import GradeResult, Outcome

REASONING_DELIMITERS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|start_think|>", "<|end_think|>"),
)
LENGTH_STOP_REASON = "length"
MALFORMED_REASONING_ERROR = "Malformed reasoning span"
REASONING_VIOLATIONS_DIAGNOSTIC = "reasoning_violations"

_OPENING_MARKERS = frozenset(opening for opening, _ in REASONING_DELIMITERS)
_REASONING_MARKER = re.compile("|".join(re.escape(marker) for pair in REASONING_DELIMITERS for marker in pair))


class ReasoningViolation(StrEnum):
    """A malformed reasoning span in one model reply."""

    UNCLOSED = "unclosed"
    SECOND_SPAN = "second_span"
    STRAY_END = "stray_end"


class ReasoningSpanRule(StrEnum):
    """The grading rule for an attempt with a malformed reasoning span.

    GRADE scores the final answer as usual. REJECT grades the attempt as a submission failure
    without running its grader.
    """

    GRADE = "grade"
    REJECT = "reject"


def reasoning_span_violation(text: str, stop_reason: str) -> ReasoningViolation | None:
    """Name the malformed reasoning span in a reply decoded with its special tokens.

    A span left open by a length stop is truncation, not a format error, so it is not a violation.
    """
    openings = [match in _OPENING_MARKERS for match in _REASONING_MARKER.findall(text)]
    if not openings:
        return None
    if openings[-1]:
        return None if stop_reason == LENGTH_STOP_REASON else ReasoningViolation.UNCLOSED
    spans = sum(openings) + (not openings[0])
    if spans > 1:
        return ReasoningViolation.SECOND_SPAN
    if openings.count(False) > spans:
        return ReasoningViolation.STRAY_END
    return None


def malformed_reasoning_grade(violations: Sequence[ReasoningViolation]) -> GradeResult:
    """The reward-0 submission failure of an attempt whose turns have these violations."""
    if not violations:
        raise ValueError("A malformed reasoning grade needs at least one violation")
    return GradeResult(
        Outcome.SUBMISSION_FAILURE,
        0.0,
        MALFORMED_REASONING_ERROR,
        passed=False,
        diagnostics={REASONING_VIOLATIONS_DIAGNOSTIC: [str(violation) for violation in violations]},
    )
