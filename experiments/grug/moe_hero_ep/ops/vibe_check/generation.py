# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded autoregressive decoding with one random stream per prompt."""

from collections.abc import Callable, Sequence

import numpy as np

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    TOP_TOKEN_COUNT,
    Completion,
    SamplingSpec,
    StopReason,
    TokenProbability,
    TokenScore,
)


def decode_scores(scores: Sequence[TokenScore], decode: Callable[[list[int]], str]) -> tuple[TokenScore, ...]:
    """Assign decoded text to tokens without splitting a Unicode character."""
    ids = [score.token_id for score in scores]
    text = decode(ids)
    end = 0
    decoded = []
    for index, score in enumerate(scores):
        # A byte token can end inside a UTF-8 character. Give the character to its final token.
        prefix = decode(ids[: index + 1]).rstrip("\ufffd") if index + 1 < len(ids) else text
        if not text.startswith(prefix) or len(prefix) < end:
            raise ValueError("Tokenizer decoding changed an earlier text fragment")
        decoded.append(score.model_copy(update={"text": text[end : len(prefix)]}))
        end = len(prefix)
    return tuple(decoded)


def scored_token(
    token_id: int,
    logprob: float,
    top_ids: Sequence[int],
    top_logprobs: Sequence[float],
    decode: Callable[[list[int]], str],
) -> TokenScore:
    """Build one score from the model distribution before temperature scaling."""
    return TokenScore(
        token_id=token_id,
        text="",
        logprob=logprob,
        top_tokens=tuple(
            TokenProbability(token_id=int(token), text=decode([int(token)]), logprob=float(value))
            for token, value in zip(top_ids, top_logprobs, strict=True)
        ),
    )


def score_expected(
    spec: SamplingSpec,
    prompt_ids: Sequence[Sequence[int]],
    expected_ids: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
    logprobs: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    decode: Callable[[list[int]], str],
) -> tuple[tuple[TokenScore, ...], ...]:
    """Score references with their own prefixes, including the first expected token.

    The backend receives full causal input rows, prediction positions, and target IDs.
    It returns target log probabilities, top token IDs, and top log probabilities.
    Expected text has no added EOS and must fit in the context without truncation.
    """
    if len(prompt_ids) != len(spec.prompts) or len(expected_ids) != len(prompt_ids):
        raise ValueError("Expected completion count differs from the prompt bank")
    for prompt, expected in zip(prompt_ids, expected_ids, strict=True):
        if not prompt or not expected or len(prompt) + len(expected) > spec.context_length:
            raise ValueError("Each prompt and expected completion must be nonempty and fit in the context")
    results = []
    for start in range(0, len(prompt_ids), spec.batch_size):
        prompts = prompt_ids[start : start + spec.batch_size]
        expected = expected_ids[start : start + spec.batch_size]
        width = max(map(len, expected))
        tokens = np.full((spec.batch_size, spec.context_length), eos_token_id, dtype=np.int32)
        positions = np.zeros((spec.batch_size, width), dtype=np.int32)
        targets = np.full((spec.batch_size, width), eos_token_id, dtype=np.int32)
        for row in range(spec.batch_size):
            prompt, continuation = prompts[row % len(prompts)], expected[row % len(expected)]
            full = [*prompt, *continuation]
            tokens[row, : len(full)] = full
            positions[row, : len(continuation)] = np.arange(len(prompt) - 1, len(full) - 1)
            targets[row, : len(continuation)] = continuation
        values, top_ids, top_values = logprobs(tokens, positions, targets)
        for row, ids in enumerate(expected):
            scores = [
                scored_token(token, float(values[row, index]), top_ids[row, index], top_values[row, index], decode)
                for index, token in enumerate(ids)
            ]
            results.append(decode_scores(scores, decode))
    return tuple(results)


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
    if len(prompt_ids) != len(spec.prompts):
        raise ValueError("Tokenized prompt count differs from the prompt bank")
    if any(not ids or len(ids) > spec.context_length for ids in prompt_ids):
        raise ValueError("Each prompt must contain 1..context_length tokens")
    results = []
    for start in range(0, len(prompt_ids), spec.batch_size):
        batch = slice(start, start + spec.batch_size)
        results.extend(
            _generate_batch(
                spec.model_copy(update={"prompts": spec.prompts[batch]}),
                prompt_ids[batch],
                eos_token_id=eos_token_id,
                logits=logits,
                decode=decode,
            )
        )
    return tuple(results)


def _generate_batch(
    spec: SamplingSpec,
    prompt_ids: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
    logits: Callable[[np.ndarray, np.ndarray], np.ndarray],
    decode: Callable[[list[int]], str],
) -> tuple[Completion, ...]:
    batch_size = spec.batch_size
    tokens = np.full((batch_size, spec.context_length), eos_token_id, dtype=np.int32)
    positions = np.zeros(batch_size, dtype=np.int32)
    for row in range(batch_size):
        ids = prompt_ids[row % len(prompt_ids)]
        tokens[row, : len(ids)] = ids
        positions[row] = len(ids) - 1
    random = [np.random.default_rng(prompt.seed) for prompt in spec.prompts]
    generated: list[list[int]] = [[] for _ in prompt_ids]
    token_scores: list[list[TokenScore]] = [[] for _ in prompt_ids]
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
            shifted = scores[row] - scores[row].max()
            logprobs = shifted - np.log(np.exp(shifted).sum())
            count = min(TOP_TOKEN_COUNT, len(logprobs))
            top_ids = np.argpartition(-logprobs, count - 1)[:count]
            top_ids = top_ids[np.argsort(-logprobs[top_ids], kind="stable")]
            token_scores[row].append(scored_token(token, float(logprobs[token]), top_ids, logprobs[top_ids], decode))
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
                token_scores=decode_scores(token_scores[row], decode),
            )
        )
    return tuple(results)
