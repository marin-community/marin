# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded packed prefill chunks, sampling each Snowball prompt only at its end."""

import dataclasses
import functools

import haliax as hax
import jax
import jax.numpy as jnp

from levanter.inference.engine import _apply_prefill_work, _prefill_kernel
from levanter.inference.jit_scheduler import TokenQueue, _DecodeOutputs


@functools.partial(jax.jit, donate_argnums=0)
def _reserve_prompt(state, work):
    state = _apply_prefill_work(state, work)
    decode_state, _ = state.decode_state.allocate_for_seq(work.queue.queued_slot_ids, work.queue.queued_pos_ids)
    prompt_ends = decode_state.seq_lens
    # All prompt pages are reserved, but none contains computed K/V yet. The page
    # allocator counts existing indices independently of these computed lengths.
    sequences = dataclasses.replace(decode_state.sequences, seq_lens=hax.zeros_like(decode_state.seq_lens))
    decode_state = dataclasses.replace(decode_state, sequences=sequences)
    return dataclasses.replace(state, decode_state=decode_state), prompt_ends


@functools.partial(jax.jit, donate_argnums=0)
def _cache_prefix(state, model, queue):
    decode_state, info = state.decode_state.allocate_for_seq(queue.queued_slot_ids, queue.queued_pos_ids)
    _, cache = model.decode(queue.queued_tokens, state.cache, info, queue.queued_pos_ids)
    return dataclasses.replace(state, cache=cache, decode_state=decode_state)


@functools.partial(jax.jit, donate_argnums=0, static_argnames=("max_seqs_in_prefill",))
def _finish_prompt(state, model, sampler, queue, max_seqs_in_prefill, prompt_ends):
    return _prefill_kernel(state, model, sampler, queue, max_seqs_in_prefill, sample_seq_lens=prompt_ends)


@jax.jit
def _append_outputs(outputs, new_outputs):
    return outputs.append(
        new_outputs.tokens,
        new_outputs.slot_ids,
        new_outputs.logprobs,
        new_outputs.num_tokens,
        new_outputs.finished,
    )


def prefill_cached(engine, requests):
    if (
        not requests
        or len(requests) > min(engine.config.max_seqs, engine.config.max_seqs_in_prefill)
        or any(r.n_generations != 1 for r in requests)
    ):
        raise ValueError("Snowball chunked prefill requires a batch fitting its sequence slots, without clones")
    length = sum(len(request.prompt_tokens) for request in requests)
    if any(not request.prompt_tokens for request in requests):
        raise ValueError("Snowball prefill requires nonempty prompts")
    chunk_size = engine.config.max_prefill_size or engine.config.max_seq_len
    capacity = engine.config.max_queued_tokens
    if length > capacity:
        raise ValueError("Prompt exceeds configured admission queue capacity")
    capacity = ((capacity + chunk_size - 1) // chunk_size) * chunk_size
    work = engine._prefill_prompts(requests, buffer_size=capacity)
    if work is None:
        raise RuntimeError("Validated Snowball prompt was not admitted")
    state, prompt_ends = _reserve_prompt(engine.gen_state, work)
    outputs = _DecodeOutputs.init(max_tokens=2 * engine.config.max_seqs, max_seqs=engine.config.max_seqs)
    for start in range(0, length, chunk_size):
        queue = TokenQueue(
            queued_tokens=work.queue.queued_tokens["position", hax.ds(start, chunk_size)],
            queued_slot_ids=work.queue.queued_slot_ids["position", hax.ds(start, chunk_size)],
            queued_pos_ids=work.queue.queued_pos_ids["position", hax.ds(start, chunk_size)],
            num_queued_tokens=jnp.array(min(chunk_size, length - start), jnp.int32),
        )
        if len(requests) == 1 and start + chunk_size < length:
            state = _cache_prefix(state, engine.model, queue)
        else:
            state, new_outputs = _finish_prompt(
                state, engine.model, engine.sampler, queue, engine.config.max_seqs_in_prefill, prompt_ends
            )
            outputs = _append_outputs(outputs, new_outputs)
    engine.gen_state = state
    return outputs
