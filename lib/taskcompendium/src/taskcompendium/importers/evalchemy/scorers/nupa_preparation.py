# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Representation extraction from Marin Evalchemy's MIT-licensed NUPA-Loose scorer.

This preserves the independently written source protocol; it does not copy the
GPL-licensed Number Cookbook evaluator.
"""

import re
from enum import StrEnum


class NupaAnswerFormat(StrEnum):
    INTEGER = "Integer"
    FLOAT = "Float"
    FRACTION = "Fraction"
    SCIENTIFIC = "ScientificNotation"


_ANSWER_MARKER = re.compile(r"(?i)^(?:the\s+answer\s+is|so\s+the\s+answer\s+is)\s+")
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_INLINE_MATH = re.compile(r"\\\(([^()]*)\\\)")
_EQUATION_TAIL = re.compile(r"=\s*([^\s=]+)\s*$")
_ANSWER_TAIL = re.compile(
    r"(?i)(?:^|\n)\s*(?:(?:so|therefore|thus)[, ]+)?(?:the\s+)?(?:final\s+)?"
    r"(?:answer|result|output)\s*(?:is\s*)?[:=]?\s*([^\s]+)\s*\Z"
)
_PATTERNS = {
    NupaAnswerFormat.INTEGER: re.compile(r"^\d+"),
    NupaAnswerFormat.FLOAT: re.compile(r"^\d+\.\d+"),
    NupaAnswerFormat.FRACTION: re.compile(r"^\d+/\d+"),
    NupaAnswerFormat.SCIENTIFIC: re.compile(r"^\d+\.\d+[eE][+-]?\d+"),
}
_SEPARATORS = {
    NupaAnswerFormat.INTEGER: (),
    NupaAnswerFormat.FLOAT: (".",),
    NupaAnswerFormat.FRACTION: ("/",),
    NupaAnswerFormat.SCIENTIFIC: (".", "e"),
}


def _normalize_answer(answer: str) -> str:
    return answer.replace("+", "").replace("-", "").replace("E", "e")


def full_answer(candidate: str, answer_format: str) -> str | None:
    if _PATTERNS[NupaAnswerFormat(answer_format)].fullmatch(candidate) is None:
        return None
    return _normalize_answer(candidate)


def extract_answer(text: str | None, answer_format: str) -> str | None:
    """Extract direct or final marked numbers with the source's representation rules."""
    representation = NupaAnswerFormat(answer_format)
    if text is None:
        return None
    if "<|end_think|>" in text:
        text = text.rsplit("<|end_think|>", 1)[1]
    if "<|start_think|>" in text:
        return None
    stripped = text.strip()
    tail = stripped.rstrip(" \t\r\n.,;:!?$")
    for pattern in (_BOXED, _INLINE_MATH):
        matches = list(pattern.finditer(tail))
        if matches and not tail[matches[-1].end() :].strip(" \t\r\n.,;:!?$\\)"):
            answer = full_answer(matches[-1].group(1).strip(), answer_format)
            if answer is not None:
                return answer
    for pattern in (_EQUATION_TAIL, _ANSWER_TAIL):
        match = pattern.search(tail)
        if match is not None:
            answer = full_answer(match.group(1), answer_format)
            if answer is not None:
                return answer
    direct = _ANSWER_MARKER.sub("", stripped, count=1)
    match = _PATTERNS[representation].match(direct)
    if match is None:
        return full_answer(tail.splitlines()[-1].strip() if tail else "", answer_format)
    return _normalize_answer(match.group())


def digit_parts(answer: str, answer_format: str) -> tuple[str, ...]:
    """Keep leading/trailing zeros and omit signs exactly as the source digit metric does."""
    separators = _SEPARATORS[NupaAnswerFormat(answer_format)]
    parts = [answer]
    for separator in separators:
        parts = [piece for part in parts for piece in part.replace("E", "e").split(separator, 1)]
    expected = len(separators) + 1
    if len(parts) != expected:
        return ("",) * expected
    return tuple("".join(character for character in part if character.isdigit()) for part in parts)
