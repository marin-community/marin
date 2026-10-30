"""Extract short answers from chat completions with one shared policy.

Tasks use :func:`short_answer_filter` for their score and retain
:func:`marked_short_answer_filter` as a strict prompt-contract signal. Keeping
the extraction and format classification together makes the policy reusable
and lets it improve without each benchmark growing a bespoke parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from eval.generation_stops import truncate_at_stop

INVALID_SHORT_ANSWER = "[invalid]"
_ANSWER_MARKER = re.compile(
    r"(?im)^[ \t]*(?:(?:final)[ \t]+)?(?:answer|a)[ \t]*:[ \t]*(?P<answer>\S[^\r\n]*)"
)
_EXPLICIT_ANSWER = re.compile(
    r"(?im)\b(?:the[ \t]+)?(?:final[ \t]+)?answer[ \t]+is[ \t]+(?P<answer>\S[^\r\n]*)"
)
_LEADING_ANSWER_MARKER = re.compile(
    r"(?i)^(?:(?:final)[ \t]+)?(?:answer|a)[ \t]*:[ \t]*"
)
_BARE_ANSWER_MAX_CHARS = 256


class ShortAnswerFormat(str, Enum):
    """How a completion rendered its final short answer."""

    CONTRACT = "contract"
    EXPLICIT = "explicit"
    BARE = "bare"
    INVALID = "invalid"


@dataclass(frozen=True)
class ShortAnswerExtraction:
    """One extracted answer and the format used to present it."""

    answer: str
    format: ShortAnswerFormat


_INVALID_EXTRACTION = ShortAnswerExtraction(
    answer=INVALID_SHORT_ANSWER,
    format=ShortAnswerFormat.INVALID,
)


def _clean_answer(answer: str) -> str:
    """Remove answer presentation syntax without changing answer words."""
    answer = answer.strip().rstrip(".").strip()
    while marker := _LEADING_ANSWER_MARKER.match(answer):
        answer = answer[marker.end() :].strip()
    if len(answer) >= 2 and (answer[0], answer[-1]) in {("(", ")"), ("[", "]")}:
        answer = answer[1:-1].strip()
    return answer.rstrip(".").strip()


def _extraction(answer: str, answer_format: ShortAnswerFormat) -> ShortAnswerExtraction:
    """Return a valid extraction when its cleaned answer has content."""
    answer = _clean_answer(answer)
    if not answer:
        return _INVALID_EXTRACTION
    return ShortAnswerExtraction(answer=answer, format=answer_format)


def _last_bare_line(response: str) -> str | None:
    """Return a plausible final bare answer line, if the response has one."""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    answer = lines[-1]
    if (
        len(answer) > _BARE_ANSWER_MAX_CHARS
        or answer.startswith("<")
        or _LEADING_ANSWER_MARKER.fullmatch(answer)
    ):
        return None
    return answer


def extract_short_answer(response: object) -> ShortAnswerExtraction:
    """Extract the final contracted, explicit, or bare short answer.

    The last answer signal wins so a completion can reason, revise, and still
    provide a final answer. Bare answers are limited to a short final line to
    avoid scoring an unstructured transcript as an answer.
    """
    if not isinstance(response, str):
        return _INVALID_EXTRACTION

    response = truncate_at_stop(response)
    candidates = [
        (match.start(), match["answer"], ShortAnswerFormat.CONTRACT)
        for match in _ANSWER_MARKER.finditer(response)
    ]
    candidates.extend(
        (match.start(), match["answer"], ShortAnswerFormat.EXPLICIT)
        for match in _EXPLICIT_ANSWER.finditer(response)
    )
    if candidates:
        _, answer, answer_format = max(candidates, key=lambda candidate: candidate[0])
        return _extraction(answer, answer_format)

    answer = _last_bare_line(response)
    return _extraction(answer, ShortAnswerFormat.BARE) if answer else _INVALID_EXTRACTION


def extract_marked_short_answer(response: object) -> str:
    """Return the last line-level marked answer or an invalid scoring value."""
    if not isinstance(response, str):
        return INVALID_SHORT_ANSWER
    matches = list(_ANSWER_MARKER.finditer(truncate_at_stop(response)))
    return _clean_answer(matches[-1]["answer"]) if matches else INVALID_SHORT_ANSWER


def marked_short_answer_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Extract contracted answers for the strict prompt-format metric."""
    del docs
    return [
        [extract_marked_short_answer(response) for response in responses]
        for responses in resps
    ]


def short_answer_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Extract short answers for the primary NQ-Open and TriviaQA metric."""
    del docs
    return [
        [extract_short_answer(response).answer for response in responses]
        for responses in resps
    ]
