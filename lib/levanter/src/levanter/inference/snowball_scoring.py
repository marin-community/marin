# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced token scores through Snowball's actual paged decode path."""

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.inference.engine import FIRST_TOKEN_LOGPROB, InferenceEngine, TokenSequenceLogprobs
from levanter.inference.page_table import PageBatchInfo


def score_cached(engine: InferenceEngine, token_ids, top_k: int) -> TokenSequenceLogprobs:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(token_ids) > engine.config.max_seq_len:
        raise ValueError("Token sequence exceeds configured cache length")
    if not token_ids:
        return TokenSequenceLogprobs([], [])
    tokens = np.asarray(token_ids, dtype=np.int32)
    assert engine.config.max_prefill_size is not None
    assert engine.config.max_pages is not None
    chunk_size = min(engine.config.max_prefill_size, 256)
    page_size = engine.config.page_size
    pages = engine.config.max_pages_per_seq
    if pages > engine.config.max_pages:
        raise ValueError("Cache does not have enough pages to score one full sequence")
    top_k = min(top_k, engine.model.Vocab.size)
    cache = engine.gen_state.cache.reset()

    @eqx.filter_jit
    def step(model, cache, ids, targets, positions, info):
        logits, cache = model.decode(ids, cache, info, positions)
        logprobs = jax.nn.log_softmax(logits.array.astype(jnp.float32), axis=-1)
        values, indices = jax.lax.top_k(logprobs, top_k)
        selected = jnp.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
        return cache, selected, values, indices

    scores = [FIRST_TOKEN_LOGPROB]
    tops = [{int(tokens[0]): FIRST_TOKEN_LOGPROB}]
    start = 0
    # Exercise single-query cached decode on both sides of the production window boundary.
    single_query_positions = {2046, 2047, 2048, 2049, 2050}
    while start < len(tokens) - 1:
        length = min(chunk_size, len(tokens) - 1 - start)
        if start in single_query_positions:
            length = 1
        else:
            upcoming = [p for p in single_query_positions if start < p < start + length]
            if upcoming:
                length = min(upcoming) - start
        end = start + length
        buffer_size = 1 if start in single_query_positions else chunk_size
        positions = np.arange(start, start + buffer_size, dtype=np.int32)
        ids = np.zeros(buffer_size, dtype=np.int32)
        targets = np.zeros(buffer_size, dtype=np.int32)
        ids[:length] = tokens[start:end]
        targets[:length] = tokens[start + 1 : end + 1]
        destinations = np.where(np.arange(buffer_size) < length, positions, -1)
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.array([0], jnp.int32), "seq"),
            page_indices=hax.named(jnp.arange(pages, dtype=jnp.int32)[None, :], ("seq", "page")),
            seq_lens=hax.named(jnp.array([end], jnp.int32), "seq"),
            cu_q_lens=hax.named(jnp.array([0, length], jnp.int32), "seq"),
            num_seqs=jnp.array(1, jnp.int32),
            new_token_dests=hax.named(jnp.asarray(destinations), "position"),
            page_size=page_size,
        )
        cache, selected, values, indices = step(
            engine.model,
            cache,
            hax.named(jnp.asarray(ids), "position"),
            jnp.asarray(targets),
            hax.named(jnp.asarray(positions), "position"),
            info,
        )
        scores.extend(np.asarray(selected[:length]).tolist())
        for row_values, row_indices in zip(np.asarray(values[:length]), np.asarray(indices[:length]), strict=True):
            tops.append({int(i): float(v) for i, v in zip(row_indices, row_values, strict=True)})
        start = end
    engine.gen_state = dataclasses.replace(engine.gen_state, cache=cache)
    if np.any(np.asarray(cache.dropped_assignments)):
        raise RuntimeError(f"Cached scoring dropped assignments: {np.asarray(cache.dropped_assignments)}")
    return TokenSequenceLogprobs(scores, tops)
