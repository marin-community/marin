# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Hosted fixed-slot Snowball generation followed by public training route replay.

Equal-length causal prompts, contiguous positions, no EOS compaction, packed
segments or shared physical pages. Every serving capture is retained, including
the final sampled token's extra unsampled output row. This diagnostic does not
promise numerical parity between different arithmetic backends.
"""

from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug.grug_moe import PreparedMoERouteCapture
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.models.snowball import (
    SnowballDecodeRouteCaptureBatch,
    SnowballLMHeadModel,
    SnowballRouteCaptureBatch,
)


@dataclass(frozen=True)
class SnowballGeneratedRoutes:
    """Generation evidence and the same serving routes in teacher token order.

    Diagnostic arrays are completed host arrays. Serving logit row p predicts
    token p+1. Only predictor_positions produced sampled_token_ids; the final
    row predicts an unsampled token. Teacher replay fixes serving expert IDs
    and qb_alpha, and uses current parameters for combine weights.
    """

    tokens: hax.NamedArray
    sampled_token_ids: np.ndarray
    serving_logits: np.ndarray
    serving_captures: tuple[SnowballDecodeRouteCaptureBatch, ...]
    teacher_capture: SnowballRouteCaptureBatch
    prompt_length: int

    @property
    def predictor_positions(self) -> np.ndarray:
        return np.arange(self.prompt_length - 1, self.tokens.axis_size("position") - 1, dtype=np.int32)

    @property
    def prediction_logits(self) -> np.ndarray:
        return self.serving_logits[:, self.predictor_positions]

    @property
    def unsampled_next_logits(self) -> np.ndarray:
        return self.serving_logits[:, -1]


@eqx.filter_jit
def _decode(model, cache, tokens, info, positions):
    return model.decode_with_route_captures(tokens, cache, info, positions)


@eqx.filter_jit
def _sample(logits: jax.Array, key: jax.Array) -> jax.Array:
    return jax.random.categorical(key, logits.astype(jnp.float32), axis=-1).astype(jnp.int32)


@eqx.filter_jit
def _teacher_logits(model, tokens, capture):
    return model(tokens, route_captures=capture)


def _page_batch(page_map: np.ndarray, page_size: int, *, start: int, count: int) -> PageBatchInfo:
    batch_size = len(page_map)
    positions = np.tile(np.arange(start, start + count, dtype=np.int32), batch_size)
    slots = np.repeat(np.arange(batch_size), count)
    destinations = page_map[slots, positions // page_size] * page_size + positions % page_size
    return PageBatchInfo(
        slot_ids=hax.named(jnp.arange(batch_size, dtype=jnp.int32), "seq"),
        page_indices=hax.named(jnp.asarray(page_map), ("seq", "page")),
        seq_lens=hax.named(jnp.full((batch_size,), start + count, dtype=jnp.int32), "seq"),
        cu_q_lens=hax.named(jnp.arange(batch_size + 1, dtype=jnp.int32) * count, "seq"),
        num_seqs=jnp.asarray(batch_size, dtype=jnp.int32),
        new_token_dests=hax.named(jnp.asarray(destinations), "position"),
        page_size=page_size,
    )


def _teacher_capture(
    tokens: np.ndarray, captures: tuple[SnowballDecodeRouteCaptureBatch, ...], prompt_length: int
) -> SnowballRouteCaptureBatch:
    """Reorder only this fixed-slot generation's retained serving rows on host."""
    batch_size, total_length = tokens.shape
    routes = []
    for layer in range(len(captures[0].route_captures)):
        fields = []
        for field in ("selected_experts", "token_valid", "qb_alpha"):
            chunks = []
            for call, capture in enumerate(captures):
                count = prompt_length if call == 0 else 1
                value = np.asarray(getattr(capture.route_captures[layer], field))
                chunks.append(value.reshape(batch_size, count, *value.shape[1:]))
            # Concatenate time within each slot BEFORE flattening to teacher order.
            joined = np.concatenate(chunks, axis=1)
            fields.append(jnp.asarray(joined.reshape(batch_size * total_length, *joined.shape[2:])))
        routes.append(PreparedMoERouteCapture(*fields))
    # Check provenance against returned metadata, rather than a fresh router pass.
    for call, capture in enumerate(captures):
        start, count = (0, prompt_length) if call == 0 else (prompt_length + call - 1, 1)
        expected_tokens = tokens[:, start : start + count].reshape(-1)
        expected_positions = np.tile(np.arange(start, start + count), batch_size)
        if not np.array_equal(capture.token_ids, expected_tokens):
            raise ValueError("serving capture token binding differs from generated tokens")
        if not np.array_equal(capture.position_ids, expected_positions) or not np.all(capture.token_valid):
            raise ValueError("serving capture is not a contiguous, fully valid fixed-slot batch")
        for route in capture.route_captures:
            if not np.array_equal(route.token_valid, capture.token_valid):
                raise ValueError("per-layer validity differs from serving capture validity")
    return SnowballRouteCaptureBatch(tuple(routes), jnp.asarray(tokens), None)


def generate_with_route_captures(
    model: SnowballLMHeadModel,
    prompts: hax.NamedArray,
    *,
    sampling_key: jax.Array,
    generated_tokens: int,
    page_map: np.ndarray,
    page_size: int,
    cache_dtype: jnp.dtype,
) -> SnowballGeneratedRoutes:
    """Sample from public cached logits and retain their actual per-layer routes.

    The caller owns the explicit mesh, arithmetic precision and model. Sampling
    is categorical at temperature one, using fold_in(sampling_key, step). The
    final sampled token is decoded too, so every teacher input has a serving
    route; its next-logit row is retained but never sampled.
    """
    if prompts.ndim != 2 or tuple(axis.name for axis in prompts.axes) != ("batch", "position"):
        raise ValueError("prompts must have axes (batch, position)")
    prompt_ids = np.asarray(prompts.array)
    batch_size, prompt_length = prompt_ids.shape
    total_length = prompt_length + generated_tokens
    page_map = np.asarray(page_map, dtype=np.int32)
    if generated_tokens <= 0 or prompt_length <= 0 or batch_size <= 0 or page_size <= 0:
        raise ValueError("generation requires nonempty prompts, continuation and pages")
    if total_length > model.config.max_seq_len:
        raise ValueError("generation exceeds the model sequence length")
    if prompt_ids.dtype != np.int32 or np.any(prompt_ids < 0) or np.any(prompt_ids >= model.Vocab.size):
        raise ValueError("prompts must contain int32 vocabulary IDs")
    required_pages = (total_length + page_size - 1) // page_size
    if page_map.shape != (batch_size, required_pages) or np.any(page_map < 0):
        raise ValueError("page map must cover exactly the fixed-slot sequences")
    if len(np.unique(page_map)) != page_map.size:
        raise ValueError("the fixed-slot diagnostic does not share physical pages")
    cache = model.initial_cache(PageTableSpec(int(page_map.max()) + 1, page_size, batch_size), dtype=cache_dtype)
    info = _page_batch(page_map, page_size, start=0, count=prompt_length)
    positions = hax.named(jnp.asarray(np.tile(np.arange(prompt_length, dtype=np.int32), batch_size)), "position")
    logits, cache, capture = _decode(
        model, cache, hax.named(jnp.asarray(prompt_ids.reshape(-1)), "position"), info, positions
    )
    # Complete arrays before slicing: Explicit device indexing can otherwise
    # remove a sharded token dimension from a logically batched value.
    logits_host = np.asarray(logits.array).reshape(batch_size, prompt_length, model.Vocab.size)
    output_rows = [logits_host]
    captures = [jax.device_get(capture)]
    generated = []
    next_logits = logits_host[:, -1]
    for step in range(generated_tokens):
        sampled = np.asarray(_sample(jnp.asarray(next_logits), jax.random.fold_in(sampling_key, step)))
        generated.append(sampled)
        position = prompt_length + step
        info = _page_batch(page_map, page_size, start=position, count=1)
        logits, cache, capture = _decode(
            model,
            cache,
            hax.named(jnp.asarray(sampled), "position"),
            info,
            hax.named(jnp.full((batch_size,), position, dtype=jnp.int32), "position"),
        )
        next_logits = np.asarray(logits.array).reshape(batch_size, model.Vocab.size)
        output_rows.append(next_logits[:, None, :])
        captures.append(jax.device_get(capture))
    sampled_ids = np.stack(generated, axis=1)
    tokens = np.concatenate((prompt_ids, sampled_ids), axis=1)
    retained = tuple(captures)
    return SnowballGeneratedRoutes(
        tokens=hax.named(jnp.asarray(tokens), ("batch", "position")),
        sampled_token_ids=sampled_ids,
        serving_logits=np.concatenate(output_rows, axis=1),
        serving_captures=retained,
        teacher_capture=_teacher_capture(tokens, retained, prompt_length),
        prompt_length=prompt_length,
    )


def replay_generated_routes(model: SnowballLMHeadModel, generated: SnowballGeneratedRoutes) -> np.ndarray:
    """Run public teacher logits using exactly the retained serving route payload."""
    return np.asarray(_teacher_logits(model, generated.tokens, generated.teacher_capture).array)
