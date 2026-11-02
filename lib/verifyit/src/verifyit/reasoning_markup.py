# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify reasoning-span markup in completed assistant text.

A reply may hold one reasoning span before its answer, in any of the delimiter syntaxes below. A
closing marker with no opening marker closes a span that the chat template opened in the prompt.
"""

import re
from enum import StrEnum

REASONING_DELIMITERS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|start_think|>", "<|end_think|>"),
)
_LENGTH_STOP_REASON = "length"

_CLOSING_MARKER = dict(REASONING_DELIMITERS)
_REASONING_MARKER = re.compile("|".join(re.escape(marker) for pair in REASONING_DELIMITERS for marker in pair))


class ReasoningViolation(StrEnum):
    """A malformed reasoning span in one model reply."""

    UNCLOSED = "unclosed"
    SECOND_SPAN = "second_span"
    STRAY_END = "stray_end"


def reasoning_span_violation(text: str, stop_reason: str) -> ReasoningViolation | None:
    """Name the malformed reasoning span in a reply decoded with its special tokens.

    A closing marker must match the opening marker of its span. A span left open by a length stop is
    truncation, so it is not a violation by itself.
    """
    spans = 0
    stray_end = False
    expected_close: str | None = None
    for index, marker in enumerate(_REASONING_MARKER.findall(text)):
        if marker in _CLOSING_MARKER:
            spans += 1
            expected_close = _CLOSING_MARKER[marker]
        elif index == 0:
            spans += 1
        elif marker == expected_close:
            expected_close = None
        else:
            stray_end = True
    if expected_close is not None and stop_reason != _LENGTH_STOP_REASON:
        return ReasoningViolation.UNCLOSED
    if spans > 1:
        return ReasoningViolation.SECOND_SPAN
    if stray_end:
        return ReasoningViolation.STRAY_END
    return None
