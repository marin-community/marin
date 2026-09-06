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

QUALITY_VERSION = "numeric-answer-candidate-2"
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMERIC_VALUE = re.compile(rf"{NUMBER}(?:/{NUMBER})?")
# Capture the whole non-whitespace candidate before validating its grammar. A
# prefix match would silently turn malformed values such as 1,23 into 1.
MARKER = re.compile(r"####\s*")
BOXED = re.compile(r"\\boxed\{([^}]*)(\}|$)")
FINAL_PROSE = re.compile(r"\b(?:the\s+)?(?:final\s+)?answer\b(?:\*\*)?\s*(is\b\s*:|is\b|=|:)\s*", re.I)
ROLE_TURN = re.compile(r"(?:^|\n)\s*(?:Human|User|Assistant|System):|<\|(?:start_header_id|im_start)\|>", re.I)
WRAPPERS = (
    (r"\(", r"\)"),
    (r"\[", r"\]"),
    ("**", "**"),
    ("$$", "$$"),
    ("$", "$"),
    ("<", ">"),
    ('"', '"'),
    ("'", "'"),
    ("`", "`"),
)


def numeric_candidate(text: str, start: int) -> tuple[str, int]:
    """Read a whole wrapped value or lexical token, never a valid numeric prefix."""
    while start < len(text) and text[start].isspace():
        start += 1
    for opening, closing in WRAPPERS:
        if text.startswith(opening, start):
            end = text.find(closing, start + len(opening))
            if end < 0:
                return text[start:], len(text)
            end += len(closing)
            suffix = re.match(r"\w+", text[end:])
            if suffix:
                end += suffix.end()
            return text[start:end], end
    match = re.match(r"[^\s\"'`<>$\\]+", text[start:])
    return (match.group(), start + match.end()) if match else ("", start)


def unwrap_candidate(value: str) -> str:
    value = value.strip()
    # A single sentence punctuation mark is presentation, not part of the scalar.
    if value.endswith(tuple(".,;!?")):
        value = value[:-1]
    while True:
        for opening, closing in WRAPPERS:
            if value.startswith(opening) and value.endswith(closing) and len(value) >= len(opening) + len(closing):
                value = value[len(opening) : -len(closing)].strip()
                break
        else:
            return value


def empty_format_instruction(text: str, start: int, end: int) -> bool:
    """Ignore an empty box only when the surrounding prose identifies a template."""
    before, after = text[max(0, start - 100) : start], text[end : end + 100]
    return bool(
        re.search(r"(?:do not use|not to use|specif\w* no|without|no)\s*$", before, re.I)
        or re.match(r"\s+is\s+(?:not used|included here|not the answer)\b", after, re.I)
    )


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
    for match in BOXED.finditer(final_assistant_text):
        if (
            match.group(2)
            and not match.group(1).strip()
            and empty_format_instruction(final_assistant_text, match.start(), match.end())
        ):
            continue
        suffix = re.match(r"\w+", final_assistant_text[match.end() :])
        value = match.group(1) + (suffix.group() if suffix else "")
        candidates.append((value if match.group(2) else "{" + value, match.end()))
    unfinished = False
    for pattern in (MARKER, FINAL_PROSE):
        for match in pattern.finditer(final_assistant_text):
            start = match.end()
            if pattern is FINAL_PROSE and match.group(1) == ":" and final_assistant_text.startswith("**", start):
                start += 2  # Closing emphasis on a heading, not on its following value.
            remainder = final_assistant_text[start:].lstrip()
            if BOXED.match(remainder):
                continue
            if pattern is FINAL_PROSE and re.match(r"expected to be\b", remainder, re.I):
                continue  # A description of the expected format, not an asserted value.
            raw, end = numeric_candidate(final_assistant_text, start)
            value = unwrap_candidate(raw)
            if value.startswith("####") or BOXED.fullmatch(value):
                continue  # The nested explicit candidate is checked independently.
            if (
                pattern is FINAL_PROSE
                and match.group(1) == ":"
                and "\n" in final_assistant_text[match.start() : start]
                and re.match(r"(?:The|To|We|There|This|Let's)\b", value)
            ):
                continue  # A heading introducing prose, not a scalar assertion.
            unfinished |= not value or not raw.strip("\"'`*$<> \\()[]")
            candidates.append((value, end))
    # A bare numeric final line is useful for models omitting the requested marker.
    # It cannot promote an arbitrary number from earlier explanatory prose.
    lines = final_assistant_text.rstrip().splitlines()
    if lines and normalize_numeric_answer(lines[-1]) is not None:
        candidates.append((lines[-1], len(final_assistant_text.rstrip())))
    ambiguous = any(
        re.match(
            r"(?:\s+(?:or\b|and\s+\$?[+-]?(?:\d|\.\d))|\s*[+*/=\-]\s*\$?[+-]?(?:\d|\.\d))",
            final_assistant_text[end:],
            re.I,
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
    reconsidered = bool(
        candidates
        and re.search(
            r"\b(?:let me think again|let me reconsider|this is confusing|that (?:is|seems) wrong)\b",
            final_assistant_text[max(end for _, end in candidates) :],
            re.I,
        )
    )
    status = AnswerStatus.EXTRACTED
    if ROLE_TURN.search(final_assistant_text):
        status = AnswerStatus.ROLE_CONTINUATION
    elif not candidates:
        status = AnswerStatus.MISSING
    elif ambiguous or unfinished or reconsidered:
        status = AnswerStatus.AMBIGUOUS
    elif None in normalized:
        status = AnswerStatus.MALFORMED
    elif len(values) > 1:
        status = AnswerStatus.CONFLICTING
    value = next(iter(values)) if status == AnswerStatus.EXTRACTED else None
    return NumericAnswer(status, value, len(candidates), len(values), tail)
