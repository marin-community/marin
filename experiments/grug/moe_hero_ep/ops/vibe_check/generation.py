# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded autoregressive decoding with one random stream per prompt."""

from collections.abc import Callable, Sequence

import numpy as np

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Completion, SamplingSpec, StopReason


def generate(
    spec: SamplingSpec,
    prompt_ids: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
    logits: Callable[[np.ndarray, np.ndarray], np.ndarray],
    decode: Callable[[list[int]], str],
) -> tuple[Completion, ...]:
    """Generate a fixed prompt bank, preserving text and token boundaries.

    ``logits`` receives padded token rows and the index of each row's last input token.
    The model backend must be dropless so filler rows cannot change prompt routing.
    """
    batch_size = spec.batch_size
    if len(prompt_ids) != len(spec.prompts) or batch_size < len(prompt_ids):
        raise ValueError("Prompt count does not fit the batch")
    if any(not ids or len(ids) > spec.context_length for ids in prompt_ids):
        raise ValueError("Each prompt must contain 1..context_length tokens")
    tokens = np.full((batch_size, spec.context_length), eos_token_id, dtype=np.int32)
    positions = np.zeros(batch_size, dtype=np.int32)
    for row in range(batch_size):
        ids = prompt_ids[row % len(prompt_ids)]
        tokens[row, : len(ids)] = ids
        positions[row] = len(ids) - 1
    random = [np.random.default_rng(prompt.seed) for prompt in spec.prompts]
    generated: list[list[int]] = [[] for _ in prompt_ids]
    stops: list[StopReason | None] = [
        StopReason.CONTEXT_LIMIT if len(ids) == spec.context_length else None for ids in prompt_ids
    ]
    for _ in range(spec.max_new_tokens):
        if all(stop is not None for stop in stops):
            break
        scores = np.asarray(logits(tokens, positions), dtype=np.float64)
        if scores.ndim != 2 or scores.shape[0] != batch_size or not np.isfinite(scores).all():
            raise ValueError("Model returned invalid logits")
        for row in range(len(prompt_ids)):
            if stops[row] is not None:
                continue
            if spec.temperature == 0:
                token = int(np.argmax(scores[row]))
            else:
                probabilities = np.exp((scores[row] - scores[row].max()) / spec.temperature)
                probabilities /= probabilities.sum()
                token = int(random[row].choice(scores.shape[1], p=probabilities))
            generated[row].append(token)
            positions[row] += 1
            tokens[row, positions[row]] = token
            if token == eos_token_id:
                stops[row] = StopReason.EOS
            elif positions[row] + 1 == spec.context_length:
                stops[row] = StopReason.CONTEXT_LIMIT
            elif len(generated[row]) == spec.max_new_tokens:
                stops[row] = StopReason.MAX_NEW_TOKENS
    results = []
    for row, prompt in enumerate(spec.prompts):
        assert stops[row] is not None
        results.append(
            Completion(
                prompt_id=prompt.id,
                prompt_token_ids=tuple(prompt_ids[row]),
                token_ids=tuple(generated[row]),
                text=decode(generated[row]),
                stop_reason=stops[row],
            )
        )
    return tuple(results)
