# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bits-per-byte scoring primitives, ported from OLMo-Eval.

Pure functions (no model forward) so the parity-critical semantics — the byte
denominator, the prompt/continuation token boundary, and the BOS rule — are
unit-testable without a TPU. The model forward that produces the summed
continuation log-probability lives in ``run.py``.

Reference: OLMo-Eval ``BitsPerByteScorer`` (``scorers/base.py``),
``BPBMetricInstanceAvg`` (``metrics/base.py``), and
``encode_context_and_continuation`` (``inference/tokenizer_utils.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

LN2 = math.log(2.0)


def continuation_num_bytes(continuation: str) -> int:
    """Return the BPB denominator: the UTF-8 byte length of the gold continuation.

    This is the original continuation string (including any leading space), never
    the prompt and never the token count.
    """
    return len(continuation.encode("utf-8"))


def bits_per_byte(sum_logprob: float, num_bytes: int) -> float:
    """Return ``-sum_logprob / (num_bytes * ln2)``.

    ``sum_logprob`` is the sum of natural-log token log-probabilities over the
    continuation tokens. Matches OLMo-Eval's guard of returning 0.0 when there
    are no bytes.
    """
    if num_bytes == 0:
        return 0.0
    return -sum_logprob / (num_bytes * LN2)


def task_bpb(instance_logprobs: Sequence[float], instance_num_bytes: Sequence[int]) -> float:
    """Aggregate per-instance BPB into a task score: the unweighted instance mean.

    This is OLMo-Eval's ``BPBMetricInstanceAvg`` — each instance contributes its
    BPB with equal weight, NOT weighted by byte length (which would be the
    distinct ``BPBMetricByteAvg``).
    """
    if not instance_logprobs:
        raise ValueError("cannot aggregate BPB over zero instances")
    per_instance = [
        bits_per_byte(logprob, num_bytes)
        for logprob, num_bytes in zip(instance_logprobs, instance_num_bytes, strict=True)
    ]
    return sum(per_instance) / len(per_instance)


@dataclass(frozen=True)
class EncodedInstance:
    """Token layout for one scored instance.

    ``tokens`` is ``[bos?] + context_ids + continuation_ids`` and the continuation
    tokens are ``tokens[prompt_length:]``. ``num_bytes`` is the UTF-8 byte length
    of the original continuation (the BPB denominator).
    """

    tokens: tuple[int, ...]
    prompt_length: int
    num_continuation_tokens: int
    num_bytes: int


def _move_trailing_spaces(context: str, continuation: str) -> tuple[str, str]:
    """Move trailing spaces from the context into the continuation (pre-tokenization).

    Mirrors OLMo-Eval's lm-eval-parity rule (``tokenizer_utils.py``). Only the
    tokenization boundary is affected; the byte denominator uses the original
    continuation.
    """
    num_trailing = len(context) - len(context.rstrip())
    if num_trailing > 0:
        return context[:-num_trailing], context[-num_trailing:] + continuation
    return context, continuation


def encode_context_continuation(
    encode: Callable[[str], Sequence[int]],
    context: str,
    continuation: str,
    *,
    bos_token_id: int | None,
) -> EncodedInstance:
    """Tokenize one instance exactly as OLMo-Eval does.

    - trailing spaces on ``context`` are moved into the continuation before
      tokenization;
    - continuation tokens are ``encode(context + continuation)[len(encode(context)):]``
      (join-and-slice, robust to BPE merges across the boundary);
    - a BOS token is prepended to the context iff ``bos_token_id is not None``;
    - the byte denominator uses the original (pre-move) continuation.

    ``encode`` must tokenize a string WITHOUT adding special tokens (the caller
    binds ``add_special_tokens=False``), so BOS is controlled here explicitly.
    """
    num_bytes = continuation_num_bytes(continuation)
    enc_context, enc_continuation = _move_trailing_spaces(context, continuation)
    context_ids = tuple(encode(enc_context))
    whole_ids = tuple(encode(enc_context + enc_continuation))
    continuation_ids = whole_ids[len(context_ids) :]
    bos = (bos_token_id,) if bos_token_id is not None else ()
    tokens = (*bos, *context_ids, *continuation_ids)
    prompt_length = len(bos) + len(context_ids)
    return EncodedInstance(
        tokens=tokens,
        prompt_length=prompt_length,
        num_continuation_tokens=len(continuation_ids),
        num_bytes=num_bytes,
    )
