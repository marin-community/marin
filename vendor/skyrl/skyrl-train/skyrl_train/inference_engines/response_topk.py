"""Keep exact behavior-policy top-K IDs without relying on decoded token strings."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


def select_response_topk(logprobs: Mapping[int, float], top_k: int) -> tuple[list[int], list[float]]:
    """Select the actual top K even when serving also reports the sampled token."""
    if top_k <= 0:
        raise ValueError("response top-K width must be positive")
    candidates = []
    for token_id, value in logprobs.items():
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError("response top-K requires exact non-negative token IDs")
        score = float(value)
        if not math.isfinite(score) or score > 0:
            raise ValueError("response top-K requires finite non-positive log probabilities")
        candidates.append((token_id, score))
    if len(candidates) < top_k:
        raise ValueError("vLLM returned fewer response candidates than the requested top-K")
    selected = sorted(candidates, key=lambda item: (-item[1], item[0]))[:top_k]
    return [token_id for token_id, _ in selected], [score for _, score in selected]


def select_chat_response_topk(items: list[dict[str, Any]], top_k: int) -> tuple[list[int], list[float]]:
    """Read vLLM's exact ``token_id:N`` chat representation, never decoded text."""
    scores = {}
    for item in items:
        token = item.get("token")
        match = re.fullmatch(r"token_id:([0-9]+)", token) if isinstance(token, str) else None
        if match is None:
            raise ValueError("chat response top-K omitted exact token IDs")
        token_id = int(match.group(1))
        if token_id in scores:
            raise ValueError("chat response top-K repeated a token ID")
        scores[token_id] = item.get("logprob")
    return select_response_topk(scores, top_k)
