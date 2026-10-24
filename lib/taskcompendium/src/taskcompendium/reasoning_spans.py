# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reasoning spans in a model reply.

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

_OPENING_MARKERS = frozenset(opening for opening, _ in REASONING_DELIMITERS)
_REASONING_MARKER = re.compile("|".join(re.escape(marker) for pair in REASONING_DELIMITERS for marker in pair))


class ReasoningViolation(StrEnum):
    """A malformed reasoning span in one model reply."""

    UNCLOSED = "unclosed"
    SECOND_SPAN = "second_span"
    STRAY_END = "stray_end"


def reasoning_span_violation(text: str, stop_reason: str) -> ReasoningViolation | None:
    """Name the malformed reasoning span in a reply decoded with its special tokens.

    A span left open by a length stop is truncation, so it is not a violation.
    """
    openings = [match in _OPENING_MARKERS for match in _REASONING_MARKER.findall(text)]
    if not openings:
        return None
    if openings[-1]:
        return None if stop_reason == _LENGTH_STOP_REASON else ReasoningViolation.UNCLOSED
    spans = sum(openings) + (not openings[0])
    if spans > 1:
        return ReasoningViolation.SECOND_SPAN
    if openings.count(False) > spans:
        return ReasoningViolation.STRAY_END
    return None
