"""Score exact and partial counts of space-separated cats in a decoded reply."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

TARGET_WORD = "cat"

SHAPED_WEIGHT = 0.7
EXACT_WEIGHT = 0.3
NGRAM_WEIGHTS = {1: 0.5, 2: 0.3, 3: 0.2}
JUNK_PENALTY = 0.25
JUNK_PENALTY_PER_EXTRA_WORD = 0.02
JUNK_PENALTY_CAP = 0.45
EXTRA_CAT_PENALTY = 0.01
TRUNCATION_PENALTY = 0.1

_TRAILING_SPECIAL = re.compile(r"(?:(?:<\|im_end\|>|</s>|<\|endoftext\|>|<\|eot_id\|>)\s*)+$")
_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


@dataclass(frozen=True)
class CatCountScore:
    reward: float
    exact: bool
    cat_unigram_count: int
    cat_bigram_count: int
    cat_trigram_count: int
    n_words: int
    junk_words: int
    extra_cats: int
    truncated: bool


def strip_completion(text: str) -> str:
    """Remove trailing end-of-turn markers and outer whitespace."""
    return _TRAILING_SPECIAL.sub("", text).strip()


def shaping_words(text: str) -> list[str]:
    """Lenient word split for shaping: lowercase, any whitespace, edge punctuation removed."""
    words = [_EDGE_PUNCT.sub("", w) for w in _WS.split(strip_completion(text).lower())]
    return [w for w in words if w]


def cat_ngram_count(words: Sequence[str], k: int) -> int:
    """Count positions containing k consecutive target words."""
    run = 0
    count = 0
    for w in words:
        run = run + 1 if w == TARGET_WORD else 0
        if run >= k:
            count += 1
    return count


def closeness(observed: int, target: int) -> float:
    """1.0 at the target, falling linearly to 0.0 at an error of ``target`` or more."""
    return max(0.0, 1.0 - abs(observed - target) / target)


def cat_count_score(completion: str, n: int, *, stop_reason: Optional[str] = None) -> CatCountScore:
    """Score one assistant completion against its requested count and stop reason."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    words = shaping_words(completion)
    counts = {k: cat_ngram_count(words, k) for k in NGRAM_WEIGHTS}
    orders = [k for k in NGRAM_WEIGHTS if k <= n]
    weight_total = sum(NGRAM_WEIGHTS[k] for k in orders)
    shaped = sum(NGRAM_WEIGHTS[k] * closeness(counts[k], n - k + 1) for k in orders) / weight_total

    exact = strip_completion(completion) == " ".join([TARGET_WORD] * n)
    junk_words = sum(1 for w in words if w != TARGET_WORD)
    extra_cats = max(0, counts[1] - n)
    truncated = stop_reason == "length"

    reward = SHAPED_WEIGHT * shaped + EXACT_WEIGHT * float(exact)
    if junk_words:
        reward -= min(JUNK_PENALTY_CAP, JUNK_PENALTY + JUNK_PENALTY_PER_EXTRA_WORD * (junk_words - 1))
    elif not words:
        reward -= JUNK_PENALTY
    reward -= EXTRA_CAT_PENALTY * extra_cats
    if truncated:
        reward -= TRUNCATION_PENALTY
    reward = max(-1.0, min(1.0, reward))

    return CatCountScore(
        reward=reward,
        exact=exact,
        cat_unigram_count=counts[1],
        cat_bigram_count=counts[2],
        cat_trigram_count=counts[3],
        n_words=len(words),
        junk_words=junk_words,
        extra_cats=extra_cats,
        truncated=truncated,
    )
