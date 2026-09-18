# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded autoregressive decoding with one random stream per prompt."""

import logging
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import NamedTuple

import numpy as np
from rigging.timing import RateLimiter

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    TOP_TOKEN_COUNT,
    Completion,
    PromptCompletions,
    SamplingSpec,
    StopReason,
    TokenProbability,
    TokenScore,
)

logger = logging.getLogger(__name__)
PROGRESS_INTERVAL_SECONDS = 30.0


class BatchLogprobs[ArrayT](NamedTuple):
    token_logprobs: ArrayT
    top_token_ids: ArrayT
    top_logprobs: ArrayT


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
    *,
    prefix_ids: Sequence[int],
) -> TokenScore:
    """Build one score from the model distribution before temperature scaling."""
    prefix = list(prefix_ids)
    prefix_text = decode(prefix).rstrip("\ufffd")
    candidates = []
    for token, value in zip(top_ids, top_logprobs, strict=True):
        text = decode([*prefix, int(token)])
        if not text.startswith(prefix_text):
            raise ValueError("Candidate decoding changed an earlier text fragment")
        candidates.append(TokenProbability(token_id=int(token), text=text[len(prefix_text) :], logprob=float(value)))
    return TokenScore(
        token_id=token_id,
        text="",
        logprob=logprob,
        top_tokens=tuple(candidates),
    )


def score_expected(
    spec: SamplingSpec,
    prompt_ids: Sequence[Sequence[int]],
    expected_ids: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
    logprobs: Callable[[np.ndarray, np.ndarray, np.ndarray], BatchLogprobs[np.ndarray]],
    decode: Callable[[list[int]], str],
) -> tuple[tuple[TokenScore, ...], ...]:
    """Score references and a final EOS token with their own prefixes.

    The backend receives full causal input rows, prediction positions, and target IDs.
    It returns target log probabilities, top token IDs, and top log probabilities.
    The prompt, expected text, and added EOS must fit in the context without truncation.
    """
    if len(prompt_ids) != len(spec.prompts) or len(expected_ids) != len(prompt_ids):
        raise ValueError("Expected completion count differs from the prompt bank")
    for prompt, expected in zip(prompt_ids, expected_ids, strict=True):
        if not prompt or not expected or len(prompt) + len(expected) + 1 > spec.context_length:
            raise ValueError("Each prompt and expected completion must be nonempty and fit in the context with EOS")
    expected_ids = [(*ids, eos_token_id) for ids in expected_ids]
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
        logger.info(
            "Expected scoring: prompts %d-%d/%d, %d reference tokens; start model pass (includes compilation)",
            start + 1,
            start + len(prompts),
            len(prompt_ids),
            sum(map(len, expected)),
        )
        model_started = perf_counter()
        batch_scores = logprobs(tokens, positions, targets)
        logger.info(
            "Expected model pass completed in %.1f seconds; decode reference token probabilities",
            perf_counter() - model_started,
        )
        for row, ids in enumerate(expected):
            decode_started = perf_counter()
            scores = [
                scored_token(
                    token,
                    float(batch_scores.token_logprobs[row, index]),
                    batch_scores.top_token_ids[row, index],
                    batch_scores.top_logprobs[row, index],
                    decode,
                    prefix_ids=[*prompts[row], *ids[:index]],
                )
                for index, token in enumerate(ids)
            ]
            results.append(decode_scores(scores, decode))
            logger.info(
                "Expected scoring: %d/%d prompts completed; prompt=%s, tokens=%d, decoding=%.1f seconds",
                len(results),
                len(prompt_ids),
                spec.prompts[start + row].id,
                len(ids),
                perf_counter() - decode_started,
            )
    return tuple(results)


def generate(
    spec: SamplingSpec,
    prompt_ids: Sequence[Sequence[int]],
    *,
    eos_token_id: int,
    logits: Callable[[np.ndarray, np.ndarray], np.ndarray],
    decode: Callable[[list[int]], str],
) -> tuple[PromptCompletions, ...]:
    """Generate a fixed prompt bank, preserving text and token boundaries.

    ``logits`` receives padded token rows and the index of each row's last input token.
    The model backend must be dropless so filler rows cannot change prompt routing.
    """
    if len(prompt_ids) != len(spec.prompts):
        raise ValueError("Tokenized prompt count differs from the prompt bank")
    if any(not ids or len(ids) > spec.context_length for ids in prompt_ids):
        raise ValueError("Each prompt must contain 1..context_length tokens")
    samples: list[list[Completion]] = [[] for _ in prompt_ids]
    for sample_index in range(spec.completions_per_prompt):
        for start in range(0, len(prompt_ids), spec.batch_size):
            batch = slice(start, start + spec.batch_size)
            logger.info(
                "Generation: sample %d/%d, prompts %d-%d/%d, limit=%d new tokens per prompt",
                sample_index + 1,
                spec.completions_per_prompt,
                start + 1,
                min(start + spec.batch_size, len(prompt_ids)),
                len(prompt_ids),
                spec.max_new_tokens,
            )
            completed = _generate_batch(
                spec.model_copy(update={"prompts": spec.prompts[batch]}),
                prompt_ids[batch],
                sample_index=sample_index,
                eos_token_id=eos_token_id,
                logits=logits,
                decode=decode,
            )
            for row, completion in enumerate(completed, start):
                samples[row].append(completion)
    return tuple(
        PromptCompletions(prompt_id=prompt.id, prompt_token_ids=tuple(ids), samples=tuple(completions))
        for prompt, ids, completions in zip(spec.prompts, prompt_ids, samples, strict=True)
    )


def _generate_batch(
    spec: SamplingSpec,
    prompt_ids: Sequence[Sequence[int]],
    *,
    sample_index: int,
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
    random = [np.random.default_rng(prompt.seed + sample_index) for prompt in spec.prompts]
    generated: list[list[int]] = [[] for _ in prompt_ids]
    token_scores: list[list[TokenScore]] = [[] for _ in prompt_ids]
    stops: list[StopReason | None] = [
        StopReason.CONTEXT_LIMIT if len(ids) == spec.context_length else None for ids in prompt_ids
    ]
    started = perf_counter()
    progress = RateLimiter(PROGRESS_INTERVAL_SECONDS)
    model_seconds = 0.0
    decoding_seconds = 0.0
    logger.info("Generation: start first model pass (includes compilation); then select and decode token probabilities")
    for step in range(spec.max_new_tokens):
        if all(stop is not None for stop in stops):
            break
        model_started = perf_counter()
        scores = np.asarray(logits(tokens, positions), dtype=np.float64)
        model_seconds += perf_counter() - model_started
        if step == 0:
            logger.info(
                "Generation: first model pass completed in %.1f seconds; start token selection and decoding",
                model_seconds,
            )
        if scores.ndim != 2 or scores.shape[0] != batch_size or not np.isfinite(scores).all():
            raise ValueError("Model returned invalid logits")
        decode_started = perf_counter()
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
            token_scores[row].append(
                scored_token(
                    token,
                    float(logprobs[token]),
                    top_ids,
                    logprobs[top_ids],
                    decode,
                    prefix_ids=tokens[row, : positions[row] + 1].tolist(),
                )
            )
            generated[row].append(token)
            positions[row] += 1
            tokens[row, positions[row]] = token
            if token == eos_token_id:
                stops[row] = StopReason.EOS
            elif positions[row] + 1 == spec.context_length:
                stops[row] = StopReason.CONTEXT_LIMIT
            elif len(generated[row]) == spec.max_new_tokens:
                stops[row] = StopReason.MAX_NEW_TOKENS
            if stops[row] is not None:
                logger.info(
                    "Prompt finished: %s, sample=%d, tokens=%d, stop=%s",
                    spec.prompts[row].id,
                    sample_index + 1,
                    len(generated[row]),
                    stops[row],
                )
        decoding_seconds += perf_counter() - decode_started
        finished = sum(stop is not None for stop in stops)
        if progress.should_run() or finished == len(prompt_ids):
            elapsed = perf_counter() - started
            token_count = sum(map(len, generated))
            logger.info(
                "Generation: sample %d/%d, step %d/%d, prompts finished=%d/%d, generated tokens=%d, "
                "elapsed=%.1f seconds, tokens/sec=%.2f, model=%.1f seconds, token selection/decoding=%.1f seconds",
                sample_index + 1,
                spec.completions_per_prompt,
                step + 1,
                spec.max_new_tokens,
                finished,
                len(prompt_ids),
                token_count,
                elapsed,
                token_count / elapsed,
                model_seconds,
                decoding_seconds,
            )
    logger.info(
        "Generation loop completed in %.1f seconds; finalize text and token boundaries", perf_counter() - started
    )
    results = []
    for row, prompt in enumerate(spec.prompts):
        decode_started = perf_counter()
        assert stops[row] is not None
        results.append(
            Completion(
                sample_index=sample_index,
                seed=prompt.seed + sample_index,
                token_ids=tuple(generated[row]),
                text=decode(generated[row]),
                stop_reason=stops[row],
                token_scores=decode_scores(token_scores[row], decode),
            )
        )
        logger.info(
            "Completion finalized: %d/%d, prompt=%s, sample=%d, tokens=%d, stop=%s, decoding=%.1f seconds",
            row + 1,
            len(spec.prompts),
            prompt.id,
            sample_index + 1,
            len(generated[row]),
            stops[row],
            perf_counter() - decode_started,
        )
    return tuple(results)
