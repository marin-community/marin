"""NUPA-Loose answer extraction and digit-component preparation.

These helpers reproduce Evalchemy's NUPA-Loose permissive extraction and
component semantics (marin-community/evalchemy, ``eval/chat_benchmarks/NUPA-Loose``)
so RL rewards match the NUPA5K-Loose policy eval. Correctness itself is graded
by ``verifyit.adapters.evalchemy_nupa.grade_nupa_answer``; these callbacks only
prepare representations.
"""

from __future__ import annotations

import json
import re

INTEGER = "Integer"
FLOAT = "Float"
FRACTION = "Fraction"
SCIENTIFIC = "ScientificNotation"
ANSWER_FORMATS = (INTEGER, FLOAT, FRACTION, SCIENTIFIC)

_ANSWER_MARKER_RE = re.compile(r"(?i)^(?:the\s+answer\s+is|so\s+the\s+answer\s+is)\s+")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_INLINE_MATH_RE = re.compile(r"\\\(([^()]*)\\\)")
_EQUATION_TAIL_RE = re.compile(r"=\s*([^\s=]+)\s*$")
_ANSWER_TAIL_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:(?:so|therefore|thus)[, ]+)?(?:the\s+)?(?:final\s+)?"
    r"(?:answer|result|output)\s*(?:is\s*)?[:=]?\s*([^\s]+)\s*\Z"
)
_ANSWER_PATTERNS = {
    INTEGER: re.compile(r"^\d+"),
    FLOAT: re.compile(r"^\d+\.\d+"),
    FRACTION: re.compile(r"^\d+/\d+"),
    SCIENTIFIC: re.compile(r"^\d+\.\d+[eE][+-]?\d+"),
}
_SEPARATORS = {
    INTEGER: (),
    FLOAT: (".",),
    FRACTION: ("/",),
    SCIENTIFIC: (".", "e"),
}


def parse_ground_truth(ground_truth: object) -> tuple[str, str]:
    """Return the (answer, answer_format) pair encoded in a reward-spec ground truth."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as error:
            raise ValueError(f"NUPA ground truth must be a JSON object: {ground_truth!r}") from error
    if not isinstance(ground_truth, dict):
        raise ValueError(f"NUPA ground truth must be a JSON object: {ground_truth!r}")
    answer, answer_format = ground_truth.get("answer"), ground_truth.get("answer_format")
    if not isinstance(answer, str) or not answer:
        raise ValueError("NUPA ground truth requires a nonempty 'answer' string")
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"Unsupported NUPA answer format: {answer_format!r}")
    return answer, answer_format


def extract_answer(text: str | None, answer_format: str) -> str | None:
    """Extract a direct or clearly marked final numeric answer."""
    if answer_format not in _ANSWER_PATTERNS:
        raise ValueError(f"Unsupported NUPA answer format: {answer_format}")
    if text is None:
        return None
    if "<|end_think|>" in text:
        text = text.rsplit("<|end_think|>", 1)[1]
    if "<|start_think|>" in text:
        return None
    stripped = text.strip()
    tail = stripped.rstrip(" \t\r\n.,;:!?$")

    for pattern in (_BOXED_RE, _INLINE_MATH_RE):
        match = list(pattern.finditer(tail))
        if match and not tail[match[-1].end() :].strip(" \t\r\n.,;:!?$\\)"):
            answer = full_answer(match[-1].group(1).strip(), answer_format)
            if answer is not None:
                return answer

    for pattern in (_EQUATION_TAIL_RE, _ANSWER_TAIL_RE):
        match = pattern.search(tail)
        if match is not None:
            answer = full_answer(match.group(1), answer_format)
            if answer is not None:
                return answer

    direct = _ANSWER_MARKER_RE.sub("", stripped, count=1)
    match = _ANSWER_PATTERNS[answer_format].match(direct)
    if match is None:
        return full_answer(tail.splitlines()[-1].strip() if tail else "", answer_format)
    return _normalize_answer(match.group())


def full_answer(candidate: str, answer_format: str) -> str | None:
    """Return the normalized candidate when it fully matches the answer format."""
    if _ANSWER_PATTERNS[answer_format].fullmatch(candidate) is None:
        return None
    return _normalize_answer(candidate)


def digit_parts(answer: str, answer_format: str) -> tuple[str, ...]:
    """Split an answer into its digit components, dropping signs and separators."""
    if answer_format not in _SEPARATORS:
        raise ValueError(f"Unsupported NUPA answer format: {answer_format}")
    parts = [answer]
    for separator in _SEPARATORS[answer_format]:
        parts = [piece for part in parts for piece in part.replace("E", "e").split(separator, 1)]
    expected_parts = len(_SEPARATORS[answer_format]) + 1
    if len(parts) != expected_parts:
        return tuple("" for _ in range(expected_parts))
    return tuple("".join(character for character in part if character.isdigit()) for part in parts)


def _normalize_answer(answer: str) -> str:
    return answer.replace("+", "").replace("-", "").replace("E", "e")
