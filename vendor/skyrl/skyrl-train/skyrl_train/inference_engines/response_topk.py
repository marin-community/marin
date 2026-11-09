"""Keep exact behavior-policy top-K IDs without relying on decoded token strings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple, Protocol

import numpy as np


# These fields travel only over the native Python/Ray chat boundary.
CHAT_TOPK_WIDTH_KEY = "response_topk_width"
CHOICE_TOPK_KEY = "response_topk"
CHAT_LOGPROB_FLOOR = -9999.0


class FlatResponseLogprobs(Protocol):
    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]


class ResponseTopK(NamedTuple):
    sampled_logprobs: list[float]
    topk_indices: np.ndarray
    topk_logprobs: np.ndarray


def _flat_response_arrays(
    flat: FlatResponseLogprobs, response_ids: list[int], top_k: int, *, minimum_logprob: float | None = None
):
    stride = top_k + 1
    starts = np.arange(len(response_ids), dtype=np.int64) * stride
    if (
        top_k <= 0
        or len(flat.token_ids) != len(response_ids) * stride
        or len(flat.logprobs) != len(flat.token_ids)
        or not np.array_equal(flat.start_indices, starts)
        or not np.array_equal(flat.end_indices, starts + stride)
    ):
        raise ValueError("vLLM flat response logprobs do not match the requested candidate width")
    ids = np.asarray(flat.token_ids, dtype=np.int64).reshape(-1, stride)
    scores = np.asarray(flat.logprobs, dtype=np.float32).reshape(-1, stride)
    if minimum_logprob is not None:
        scores = np.maximum(scores, minimum_logprob)
    if not np.array_equal(ids[:, 0], response_ids):
        raise ValueError("vLLM flat sampled token IDs do not match the response")
    if np.any(ids < 0) or not np.isfinite(scores).all() or np.any(scores > 0):
        raise ValueError("vLLM flat response requires nonnegative IDs and finite nonpositive logprobs")
    return ids, scores


def _response_topk_result(ids, scores, head_ids, head_scores):
    dtype = np.int32 if ids.size == 0 or ids.max() <= np.iinfo(np.int32).max else np.int64
    return ResponseTopK(
        scores[:, 0].tolist(), np.ascontiguousarray(head_ids, dtype=dtype), np.ascontiguousarray(head_scores)
    )


def select_flat_response_topk(
    flat: FlatResponseLogprobs,
    response_ids: list[int],
    top_k: int,
    requested_ids: list[int] | None = None,
) -> ResponseTopK:
    """Select chosen-token and top-K full-vocabulary log probabilities.

    Each row contains the sampled token followed by the requested candidates.
    The sampled token may duplicate a head entry. Preserve the dictionary
    path's ordering for tied probabilities, including a tied sampled token.
    """
    ids, scores = _flat_response_arrays(flat, response_ids, top_k)
    if requested_ids is not None:
        if len(requested_ids) != top_k or not np.all(ids[:, 1:] == np.asarray(requested_ids)):
            raise ValueError("vLLM flat response omitted requested token IDs or changed their order")
        head_ids, head_scores = ids[:, 1:], scores[:, 1:]
    else:
        # Deduplicate only the extra sampled entry; the exact top-K itself is unique.
        order_scores = scores.copy()
        order_scores[:, 0] = np.where(np.any(ids[:, :1] == ids[:, 1:], axis=1), -np.inf, scores[:, 0])
        order = np.lexsort((ids, -order_scores), axis=-1)[:, :top_k]
        head_ids = np.take_along_axis(ids, order, axis=1)
        head_scores = np.take_along_axis(scores, order, axis=1)
    return _response_topk_result(ids, scores, head_ids, head_scores)


def select_flat_chat_response_topk(flat: FlatResponseLogprobs, response_ids: list[int], top_k: int) -> ResponseTopK:
    """Preserve the native chat head, including tied boundary candidates.

    Chat collects K+1 natural candidates and emits at most K+1 distinct
    entries, including the sampled token. When that token is outside the
    head, the final natural candidate occupies the hidden display slot.
    """
    # The pinned chat formatter floors both sampled and head scores before
    # our native client sorts them. Flooring can create token-ID ties.
    ids, scores = _flat_response_arrays(flat, response_ids, top_k + 1, minimum_logprob=CHAT_LOGPROB_FLOOR)
    duplicate = np.any(ids[:, :1] == ids[:, 1:], axis=1)
    order_scores = scores.copy()
    order_scores[:, 0] = np.where(duplicate, -np.inf, scores[:, 0])
    order_scores[:, -1] = np.where(duplicate, scores[:, -1], -np.inf)
    order = np.lexsort((ids, -order_scores), axis=-1)[:, :top_k]
    return _response_topk_result(
        ids, scores, np.take_along_axis(ids, order, axis=1), np.take_along_axis(scores, order, axis=1)
    )


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
