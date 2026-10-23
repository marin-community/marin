"""Final-answer extraction shared by Ultra graders and model transport."""

import re
from enum import StrEnum

from skyrl_gym.envs.aime.utils import last_boxed_only_string, remove_boxed

REASONING_DELIMITERS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|start_think|>", "<|end_think|>"),
)
_OPENING_MARKERS = frozenset(opening for opening, _ in REASONING_DELIMITERS)
_REASONING_MARKER = re.compile("|".join(re.escape(marker) for pair in REASONING_DELIMITERS for marker in pair))


class ReasoningViolation(StrEnum):
    """A malformed reasoning span in one model response."""

    UNCLOSED = "unclosed"
    SECOND_SPAN = "second_span"
    STRAY_END = "stray_end"


def reasoning_span_violation(text: str) -> ReasoningViolation | None:
    """Name the malformed reasoning span in ``text``, decoded with special tokens kept.

    A response may have one span, in any delimiter syntax. A closing marker before any opening
    marker closes a span that the chat template opened in the prompt.
    """
    openings = [match in _OPENING_MARKERS for match in _REASONING_MARKER.findall(text)]
    if not openings:
        return None
    if openings[-1]:
        return ReasoningViolation.UNCLOSED
    spans = sum(openings) + (not openings[0])
    if spans > 1:
        return ReasoningViolation.SECOND_SPAN
    if openings.count(False) > spans:
        return ReasoningViolation.STRAY_END
    return None


def final_answer_text(text: str) -> str:
    """Strip explicit reasoning blocks, preserving unmarked text.

    An unfinished reasoning block has no final answer. A closing marker alone
    ends reasoning whose opening marker was supplied by the chat template.
    Unmarked prose cannot be reliably classified as reasoning.
    """
    for opening, closing in REASONING_DELIMITERS:
        text = re.sub(re.escape(opening) + ".*?" + re.escape(closing), "", text, flags=re.DOTALL)
    last_closing = max(
        (text.rfind(closing) + len(closing) for _, closing in REASONING_DELIMITERS if closing in text), default=0
    )
    text = text[last_closing:]
    if any(opening in text for opening, _ in REASONING_DELIMITERS):
        return ""
    return text.strip().removesuffix("<|eot_id|>").strip()


def last_boxed_answer(text: str) -> str | None:
    boxed = last_boxed_only_string(text)
    return None if boxed is None else remove_boxed(boxed).strip()


def final_verdict(text: str, labels: set[str]) -> str:
    """Accept one exact label on the last nonempty line."""
    cleaned = final_answer_text(text)
    last_line = cleaned.rsplit("\n", 1)[-1].strip()
    if last_line not in labels:
        raise ValueError(f"Invalid final judge verdict: {last_line!r}")
    return last_line
