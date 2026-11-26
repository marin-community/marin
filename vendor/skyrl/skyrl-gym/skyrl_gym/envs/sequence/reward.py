"""Score a space-separated reply against the exact sequence of items a row expects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

PREFIX_WEIGHT = 0.7
EXACT_WEIGHT = 0.3
JUNK_PENALTY = 0.25
TRUNCATION_PENALTY = 0.1

_TRAILING_SPECIAL = re.compile(r"(?:(?:<\|im_end\|>|</s>|<\|endoftext\|>|<\|eot_id\|>|<eos>)\s*)+$")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SequenceScore:
    reward: float
    exact: bool
    correct_prefix: int
    n_words: int
    truncated: bool


def strip_completion(text: str) -> str:
    return _TRAILING_SPECIAL.sub("", text).strip()


def reply_words(text: str) -> list[str]:
    stripped = strip_completion(text)
    return _WS.split(stripped) if stripped else []


def sequence_score(completion: str, items: Sequence[str], *, stop_reason: Optional[str] = None) -> SequenceScore:
    """Reward the longest correct prefix of ``items`` plus an exact-match bonus, minus junk and truncation.

    Shaping splits on any whitespace; the exact match requires the items joined by single spaces, as the prompt
    asks, so "2\n4\n6" earns the full prefix credit but not the bonus.
    """
    if not items:
        raise ValueError("a sequence row needs at least one item")
    words = reply_words(completion)
    prefix = 0
    while prefix < min(len(words), len(items)) and words[prefix] == items[prefix]:
        prefix += 1
    exact = strip_completion(completion) == " ".join(items)
    truncated = stop_reason == "length"
    reward = PREFIX_WEIGHT * prefix / len(items) + EXACT_WEIGHT * float(exact)
    if len(words) != prefix or not words:
        reward -= JUNK_PENALTY
    if truncated:
        reward -= TRUNCATION_PENALTY
    reward = max(-1.0, min(1.0, reward))
    return SequenceScore(reward=reward, exact=exact, correct_prefix=prefix, n_words=len(words), truncated=truncated)
