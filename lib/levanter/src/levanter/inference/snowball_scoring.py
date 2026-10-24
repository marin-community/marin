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
from levanter.models.snowball import SnowballLMHeadModel


def validate_score_request(engine: InferenceEngine, token_ids, top_k: int):
    """Validate deterministic errors before any distributed command is broadcast."""
    if top_k < 1 or len(token_ids) > engine.config.max_seq_len:
        raise ValueError("Scoring requires positive top_k and a sequence within the configured length")
    if any(token < 0 or token >= engine.model.Vocab.size for token in token_ids):
        raise ValueError("Scoring contains a token outside the model vocabulary")


def score_cached(engine: InferenceEngine, token_ids, top_k: int) -> TokenSequenceLogprobs:
    validate_score_request(engine, token_ids, top_k)
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
    assert isinstance(engine.model, SnowballLMHeadModel)
    boundary = engine.model.config.sliding_window
    single_query_positions = set(range(max(0, boundary - 2), boundary + 3))
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


def validate_packed_score_request(engine: InferenceEngine, sequences, top_k: int):
    """Validate the diagnostic's disjoint page reservation before broadcast."""
    if not 1 <= len(sequences) <= engine.config.max_seqs:
        raise ValueError("Packed scoring exceeds configured sequence slots")
    for sequence in sequences:
        validate_score_request(engine, sequence, top_k)
        if len(sequence) < 2:
            raise ValueError("Packed scoring requires at least two tokens per sequence")
    if engine.config.max_prefill_size is None or engine.config.max_prefill_size < len(sequences):
        raise ValueError("Prefill capacity must hold at least one query per sequence")
    required_pages = sum((len(s) - 2 + engine.config.page_size) // engine.config.page_size for s in sequences)
    if required_pages > engine.gen_state.decode_state.page_table.num_pages:
        raise ValueError("Packed scoring exceeds physical page capacity")


@eqx.filter_jit
def _packed_score_step(model, cache, ids, targets, positions, info, top_k):
    logits, cache = model.decode(ids, cache, info, positions)
    logprobs = jax.nn.log_softmax(logits.array.astype(jnp.float32), axis=-1)
    values, indices = jax.lax.top_k(logprobs, top_k)
    selected = jnp.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
    return cache, selected, values, indices


def score_cached_packed(engine: InferenceEngine, sequences, top_k: int):
    """Score independent sequences together through the paged decode kernel.

    Each sequence owns disjoint physical pages and restarts its absolute positions
    at zero. Valid queries are packed contiguously; buffer padding stays invalid.
    Both serial controls and packed checks use this implementation.
    """
    validate_packed_score_request(engine, sequences, top_k)
    tokens = [np.asarray(s, dtype=np.int32) for s in sequences]
    n = len(tokens)
    page_size = engine.config.page_size
    page_width = engine.config.max_pages_per_seq
    page_counts = [(len(s) - 2 + page_size) // page_size for s in tokens]
    page_offsets = np.cumsum([0, *page_counts[:-1]])
    page_indices = np.full((n, page_width), -1, dtype=np.int32)
    for i, (offset, count) in enumerate(zip(page_offsets, page_counts, strict=True)):
        page_indices[i, :count] = offset + np.arange(count)
    cache = engine.gen_state.cache.reset()
    scores = [[FIRST_TOKEN_LOGPROB] for _ in tokens]
    tops = [[{int(s[0]): FIRST_TOKEN_LOGPROB}] for s in tokens]
    starts = np.zeros(n, dtype=np.int32)
    assert engine.config.max_prefill_size is not None
    per_seq_chunk = min(128, engine.config.max_prefill_size // n)
    if per_seq_chunk < 1:
        raise ValueError("Prefill capacity must hold at least one query per sequence")
    boundary = engine.model.config.sliding_window
    singles = set(range(max(0, boundary - 2), boundary + 3))
    top_k = min(top_k, engine.model.Vocab.size)
    layout_steps = []
    while any(start < len(s) - 1 for start, s in zip(starts, tokens, strict=True)):
        lengths = []
        for start, s in zip(starts, tokens, strict=True):
            length = min(per_seq_chunk, len(s) - 1 - start)
            if start in singles:
                length = min(length, 1)
            else:
                upcoming = [p for p in singles if start < p < start + length]
                if upcoming:
                    length = min(upcoming) - start
            lengths.append(int(length))
        # Physical single-row-per-sequence buffers around the window boundary.
        width = n if max(lengths) <= 1 else n * per_seq_chunk
        layout_steps.append(dict(starts=starts.tolist(), valid_lengths=lengths, physical_query_width=width))
        ids = np.zeros(width, np.int32)
        targets = np.zeros(width, np.int32)
        positions = np.zeros(width, np.int32)
        destinations = np.full(width, -1, np.int32)
        cu_q_lens = np.cumsum([0, *lengths], dtype=np.int32)
        ends = starts + np.asarray(lengths, np.int32)
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            dest = slice(cu_q_lens[i], cu_q_lens[i + 1])
            ids[dest] = tokens[i][start:end]
            targets[dest] = tokens[i][start + 1 : end + 1]
            positions[dest] = np.arange(start, end)
            destinations[dest] = page_offsets[i] * page_size + positions[dest]
        # Match the production scheduler: only sequences with queries are active
        # metadata rows. Keep physical page IDs, packed queries and output slices
        # unchanged; removing a zero-length row does not move any real query.
        active = np.flatnonzero(np.asarray(lengths) > 0)
        metadata_pages = np.full_like(page_indices, -1)
        metadata_pages[: len(active)] = page_indices[active]
        metadata_ends = np.full(n, -1, np.int32)
        metadata_ends[: len(active)] = ends[active]
        metadata_slots = np.full(n, -1, np.int32)
        metadata_slots[: len(active)] = active
        metadata_lengths = np.zeros(n, np.int32)
        metadata_lengths[: len(active)] = np.asarray(lengths)[active]
        metadata_cu = np.cumsum([0, *metadata_lengths], dtype=np.int32)
        layout_steps[-1].update(
            active_slots=active.tolist(),
            num_active_sequences=len(active),
            metadata_slot_ids=metadata_slots.tolist(),
            metadata_cu_q_lens=metadata_cu.tolist(),
            metadata_seq_lens=metadata_ends.tolist(),
            metadata_page_indices=metadata_pages.tolist(),
        )
        info = PageBatchInfo(
            slot_ids=hax.named(jnp.asarray(metadata_slots), "seq"),
            page_indices=hax.named(jnp.asarray(metadata_pages), ("seq", "page")),
            seq_lens=hax.named(jnp.asarray(metadata_ends), "seq"),
            cu_q_lens=hax.named(jnp.asarray(metadata_cu), "seq"),
            num_seqs=jnp.asarray(len(active), jnp.int32),
            new_token_dests=hax.named(jnp.asarray(destinations), "position"),
            page_size=page_size,
        )
        cache, selected, values, indices = _packed_score_step(
            engine.model,
            cache,
            hax.named(jnp.asarray(ids), "position"),
            jnp.asarray(targets),
            hax.named(jnp.asarray(positions), "position"),
            info,
            top_k,
        )
        selected, values, indices = map(np.asarray, (selected, values, indices))
        for i in range(n):
            dest = slice(cu_q_lens[i], cu_q_lens[i + 1])
            scores[i].extend(selected[dest].tolist())
            tops[i].extend(
                {int(k): float(v) for k, v in zip(ks, vs, strict=True)}
                for ks, vs in zip(indices[dest], values[dest], strict=True)
            )
        starts = ends
    engine.gen_state = dataclasses.replace(engine.gen_state, cache=cache)
    if np.any(np.asarray(cache.dropped_assignments)):
        raise RuntimeError("Packed cached scoring dropped expert assignments")
    engine.last_packed_score_layout = dict(num_sequences=n, per_sequence_chunk=per_seq_chunk, steps=layout_steps)
    return [TokenSequenceLogprobs(s, t) for s, t in zip(scores, tops, strict=True)]
