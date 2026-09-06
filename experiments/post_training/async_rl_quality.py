# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Supplemental numeric-answer extraction for retained RL evaluation responses.

This candidate contract is not a semantic grader or a replacement reward. Callers
must supply the final assistant segment, isolated using the pinned tokenizer's
thinking boundary. Extraction never receives the reference answer. Qualification
against independently labelled development responses precedes study use.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from fractions import Fraction

QUALITY_VERSION = "numeric-answer-candidate-1"
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMERIC_VALUE = re.compile(rf"{NUMBER}(?:/{NUMBER})?")
# Capture the whole non-whitespace candidate before validating its grammar. A
# prefix match would silently turn malformed values such as 1,23 into 1.
MARKER = re.compile(r"####\s*\$?([^\s\"'`<>]+)")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
FINAL_PROSE = re.compile(r"\b(?:the\s+)?(?:final\s+)?answer\b\s*(?:is\b|=|:)\s*\$?([^\s\"'`<>]+)", re.I)
ROLE_TURN = re.compile(r"(?:^|\n)\s*(?:Human|User|Assistant|System):|<\|(?:start_header_id|im_start)\|>", re.I)


class AnswerStatus(StrEnum):
    EXTRACTED = "extracted"
    MISSING = "missing"
    MALFORMED = "malformed"
    CONFLICTING = "conflicting"
    AMBIGUOUS = "ambiguous"
    ROLE_CONTINUATION = "role_continuation"


@dataclass(frozen=True)
class NumericAnswer:
    status: AnswerStatus
    value: str | None
    candidate_count: int
    distinct_values: int
    characters_after_last_candidate: int | None


def normalize_numeric_answer(text: str) -> str | None:
    """Normalize an explicit scalar to an exact rational, rejecting partial parses."""
    text = text.strip().removeprefix("$")
    if len(text) > 128 or NUMERIC_VALUE.fullmatch(text) is None:
        return None
    parts = text.replace(",", "").split("/")
    try:
        decimals = [Decimal(part) for part in parts]
    except InvalidOperation:
        return None
    if any(abs(int(number.as_tuple().exponent)) > 1000 for number in decimals):
        return None
    value = Fraction(decimals[0])
    if len(parts) == 2:
        denominator = Fraction(decimals[1])
        if not denominator:
            return None
        value /= denominator
    return str(value)


def extract_numeric_answer(final_assistant_text: str) -> NumericAnswer:
    """Extract explicit answer candidates; retain disagreement and parse failures.

    Supported forms are #### scalars, boxed scalars, concluding answer prose and
    a final line consisting solely of a scalar. Repeated equal values are allowed.
    Different explicit values remain conflicting even if one matches a reference.
    Unmarked arithmetic in reasoning is never searched for a reference value.
    """
    candidates: list[tuple[str, int]] = []
    for pattern in (MARKER, BOXED, FINAL_PROSE):
        for match in pattern.finditer(final_assistant_text):
            if pattern is FINAL_PROSE and match.group(1).startswith(("####", "\\boxed")):
                continue
            candidates.append((match.group(1).rstrip(".,;!?)"), match.end()))
    # A bare numeric final line is useful for models omitting the requested marker.
    # It cannot promote an arbitrary number from earlier explanatory prose.
    lines = final_assistant_text.rstrip().splitlines()
    if lines and normalize_numeric_answer(lines[-1]) is not None:
        candidates.append((lines[-1], len(final_assistant_text.rstrip())))
    ambiguous = any(
        re.match(
            r"\s+(?:or\b|and\s+\$?[+-]?(?:\d|\.\d)|[+*/=\-]\s*\$?[+-]?(?:\d|\.\d))", final_assistant_text[end:], re.I
        )
        or (
            final_assistant_text[end - 1 : end] == ","
            and re.match(r"\s*\$?[+-]?(?:\d|\.\d)", final_assistant_text[end:])
        )
        for _, end in candidates
    )
    normalized = [normalize_numeric_answer(value) for value, _ in candidates]
    values = set(normalized) - {None}
    tail = len(final_assistant_text) - max(end for _, end in candidates) if candidates else None
    status = AnswerStatus.EXTRACTED
    if ROLE_TURN.search(final_assistant_text):
        status = AnswerStatus.ROLE_CONTINUATION
    elif not candidates:
        status = AnswerStatus.MISSING
    elif ambiguous:
        status = AnswerStatus.AMBIGUOUS
    elif None in normalized:
        status = AnswerStatus.MALFORMED
    elif len(values) > 1:
        status = AnswerStatus.CONFLICTING
    value = next(iter(values)) if status == AnswerStatus.EXTRACTED else None
    return NumericAnswer(status, value, len(candidates), len(values), tail)
