# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify reasoning-span markup in completed assistant text.

A reply may hold one reasoning span before its answer, in any of the delimiter syntaxes below. When
the chat template opens a span in the prompt, a reply's first closing marker may close it without an
opening marker; otherwise that closing marker is stray.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

REASONING_DELIMITERS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|start_think|>", "<|end_think|>"),
)
_LENGTH_STOP_REASON = "length"
COMPLETED_STOP_REASON = "stop"

_CLOSING_MARKER = dict(REASONING_DELIMITERS)
_REASONING_MARKER = re.compile("|".join(re.escape(marker) for pair in REASONING_DELIMITERS for marker in pair))


class ReasoningViolation(StrEnum):
    """A malformed reasoning span in one model reply."""

    UNCLOSED = "unclosed"
    SECOND_SPAN = "second_span"
    STRAY_END = "stray_end"


@dataclass(frozen=True)
class ReasoningReply:
    """A sampled reply's text, reasoning markers retained, and the reason its generation stopped."""

    text: str
    stop_reason: str


class ReasoningSpanError(NamedTuple):
    violation: ReasoningViolation
    marker_index: int | None
    """Position of the second opening or stray closing marker; ``None`` for a span left open."""


def first_reasoning_violation(
    markers: Sequence[str], stop_reason: str, *, prompt_opens_span: bool
) -> ReasoningSpanError | None:
    """Find the first malformed marker among a reply's reasoning markers, in order.

    A closing marker must match the opening marker of its span. A length stop excuses an open span.
    ``prompt_opens_span`` says whether the prompt ends inside a span, which only a leading closing
    marker may close.
    """
    if markers and markers[0] not in _CLOSING_MARKER and not prompt_opens_span:
        return ReasoningSpanError(ReasoningViolation.STRAY_END, 0)
    expected_close = _CLOSING_MARKER.get(markers[0]) if markers else None
    for index, marker in enumerate(markers[1:], start=1):
        if marker == expected_close:
            expected_close = None
        elif marker in _CLOSING_MARKER:
            return ReasoningSpanError(ReasoningViolation.SECOND_SPAN, index)
        else:
            return ReasoningSpanError(ReasoningViolation.STRAY_END, index)
    if expected_close is not None and stop_reason != _LENGTH_STOP_REASON:
        return ReasoningSpanError(ReasoningViolation.UNCLOSED, None)
    return None


def reasoning_span_violation(text: str, stop_reason: str, *, prompt_opens_span: bool) -> ReasoningViolation | None:
    """Name the malformed reasoning span in a reply decoded with its special tokens."""
    error = first_reasoning_violation(_REASONING_MARKER.findall(text), stop_reason, prompt_opens_span=prompt_opens_span)
    return None if error is None else error.violation
