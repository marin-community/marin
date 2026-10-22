# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded prefill chunks for one Snowball sequence, followed by ordinary generation."""

import dataclasses
import functools

import haliax as hax
import jax
import jax.numpy as jnp

from levanter.inference.engine import _apply_prefill_work, _prefill_kernel
from levanter.inference.jit_scheduler import TokenQueue


@functools.partial(jax.jit, donate_argnums=0)
def _reserve_prompt(state, work):
    state = _apply_prefill_work(state, work)
    decode_state, _ = state.decode_state.allocate_for_seq(work.queue.queued_slot_ids, work.queue.queued_pos_ids)
    # All prompt pages are reserved, but none contains computed K/V yet. The page
    # allocator counts existing indices independently of these computed lengths.
    sequences = dataclasses.replace(decode_state.sequences, seq_lens=hax.zeros_like(decode_state.seq_lens))
    decode_state = dataclasses.replace(decode_state, sequences=sequences)
    return dataclasses.replace(state, decode_state=decode_state)


@functools.partial(jax.jit, donate_argnums=0)
def _cache_prefix(state, model, queue):
    decode_state, info = state.decode_state.allocate_for_seq(queue.queued_slot_ids, queue.queued_pos_ids)
    _, cache = model.decode(queue.queued_tokens, state.cache, info, queue.queued_pos_ids)
    return dataclasses.replace(state, cache=cache, decode_state=decode_state)


@functools.partial(jax.jit, donate_argnums=0, static_argnames=("max_seqs_in_prefill",))
def _finish_prompt(state, model, sampler, queue, max_seqs_in_prefill):
    # Only the final chunk reaches the production sampler and generation queue.
    return _prefill_kernel(state, model, sampler, queue, max_seqs_in_prefill)


def prefill_cached(engine, requests):
    if len(requests) != 1 or requests[0].n_generations != 1:
        raise ValueError("Snowball chunked prefill supports one sequence without clones")
    length = len(requests[0].prompt_tokens)
    if length == 0:
        raise ValueError("Snowball prefill requires a nonempty prompt")
    chunk_size = engine.config.max_prefill_size or engine.config.max_seq_len
    capacity = engine.config.max_queued_tokens
    if length > capacity:
        raise ValueError("Prompt exceeds configured admission queue capacity")
    capacity = ((capacity + chunk_size - 1) // chunk_size) * chunk_size
    work = engine._prefill_prompts(requests, buffer_size=capacity)
    if work is None:
        raise RuntimeError("Validated Snowball prompt was not admitted")
    state = _reserve_prompt(engine.gen_state, work)
    outputs = None
    for start in range(0, length, chunk_size):
        queue = TokenQueue(
            queued_tokens=work.queue.queued_tokens["position", hax.ds(start, chunk_size)],
            queued_slot_ids=work.queue.queued_slot_ids["position", hax.ds(start, chunk_size)],
            queued_pos_ids=work.queue.queued_pos_ids["position", hax.ds(start, chunk_size)],
            num_queued_tokens=jnp.array(min(chunk_size, length - start), jnp.int32),
        )
        if start + chunk_size < length:
            state = _cache_prefix(state, engine.model, queue)
        else:
            state, outputs = _finish_prompt(
                state, engine.model, engine.sampler, queue, engine.config.max_seqs_in_prefill
            )
    engine.gen_state = state
    assert outputs is not None
    return outputs
