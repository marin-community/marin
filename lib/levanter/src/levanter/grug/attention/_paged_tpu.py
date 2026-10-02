# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward-only TPU paged decode with FP32 tile math and accurate softmax."""

from math import log
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

TPU_PAGE_ALIGNMENT = 16
TPU_HEAD_ALIGNMENT = 128
_QUERY_GROUP_ALIGNMENT = 8

_LOG_2_HIGH = 0.693145751953125
_LOG_2_LOW = log(2) - _LOG_2_HIGH
_INVERSE_LOG_2 = 1 / log(2)
_MIN_NORMAL_LOG = -126 * log(2)


class _AttentionState(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    maximum: jax.Array


def _exp_nonpositive(x):
    # Reduce to [-log(2)/2, log(2)/2] with a split log(2), then evaluate
    # the degree-eight Taylor polynomial. Its truncation error is below 3e-10.
    # Power-of-two reconstruction is exact; TPU arithmetic flushes subnormals.
    exponent = jnp.floor(jnp.maximum(x, _MIN_NORMAL_LOG) * _INVERSE_LOG_2 + 0.5)
    remainder = (x - exponent * _LOG_2_HIGH) - exponent * _LOG_2_LOW
    remainder = jnp.where(jnp.isfinite(x), remainder, 0.0)
    polynomial = jnp.full_like(x, 1 / 40320)
    for coefficient in (1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 1 / 2, 1.0, 1.0):
        polynomial = polynomial * remainder + coefficient
    power = jax.lax.bitcast_convert_type((exponent.astype(jnp.int32) + 127) << 23, jnp.float32)
    return jnp.where(x >= _MIN_NORMAL_LOG, polynomial * power, 0.0)


def _decode_kernel(table_ref, bounds_ref, q_ref, cache_ref, scale_ref, output_ref, page_buffer, semaphore):
    token, head = pl.program_id(0), pl.program_id(1)
    lower, upper = bounds_ref[token, 0], bounds_ref[token, 1]
    page_size = page_buffer.shape[0]
    query = q_ref[...].astype(jnp.float32)
    initial = _AttentionState(
        jnp.zeros(query.shape, jnp.float32),
        jnp.zeros((query.shape[0], 1), jnp.float32),
        jnp.full((query.shape[0], 1), -jnp.inf, jnp.float32),
    )

    def attend_page(page, state):
        physical = table_ref[token, page]
        copy = pltpu.make_async_copy(cache_ref.at[physical, :, head], page_buffer, semaphore)
        copy.start()
        copy.wait()
        # Converting first gives Mosaic unpacked FP32 rows for the K/V slice.
        loaded = page_buffer[...].astype(jnp.float32)
        key, value = loaded[:, 0], loaded[:, 1]
        scores = jnp.dot(query, key.T, preferred_element_type=jnp.float32) * scale_ref[0]
        # Construct each mask in its consumer's layout. Mosaic cannot reshape
        # a lane vector into the column vector needed for the value mask.
        position = page * page_size + jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
        scores = jnp.where((position >= lower) & (position < upper), scores, -jnp.inf)
        maximum = jnp.maximum(state.maximum, jnp.max(scores, axis=1, keepdims=True))
        correction = _exp_nonpositive(state.maximum - maximum)
        weights = _exp_nonpositive(scores - maximum)
        value_position = page * page_size + jax.lax.broadcasted_iota(jnp.int32, value.shape, 0)
        value = jnp.where((value_position >= lower) & (value_position < upper), value, 0.0)
        numerator = state.numerator * correction + jnp.dot(weights, value, preferred_element_type=jnp.float32)
        denominator = state.denominator * correction + jnp.sum(weights, axis=1, keepdims=True)
        return _AttentionState(numerator, denominator, maximum)

    first_page = lower // page_size
    last_page = jnp.where(upper > lower, pl.cdiv(upper, page_size), first_page)
    state = jax.lax.fori_loop(first_page, last_page, attend_page, initial)
    output_ref[...] = (state.numerator / jnp.where(state.denominator > 0, state.denominator, 1)).astype(
        output_ref.dtype
    )


def tpu_paged_decode(q, kv_pages, token_pages, bounds, sm_scale, *, interpret=False):
    """Attend to local cache pages inside the caller's KV-head shard_map.

    Queries are [tokens, heads, groups, dim], cache pages are interleaved K/V,
    and bounds are inclusive lower/exclusive upper token positions. Empty
    bounds return zero. This forward-only kernel requires page_size >= 16 and
    head dimensions divisible by 128.
    """
    tokens, heads, groups, dim = q.shape
    page_size = kv_pages.shape[1]
    if page_size < TPU_PAGE_ALIGNMENT or page_size % TPU_PAGE_ALIGNMENT or dim % TPU_HEAD_ALIGNMENT:
        raise ValueError("TPU paged decode requires page_size divisible by 16 and head_dim divisible by 128")
    padded_groups = pl.cdiv(groups, _QUERY_GROUP_ALIGNMENT) * _QUERY_GROUP_ALIGNMENT
    padded = jnp.pad(q, ((0, 0), (0, 0), (0, padded_groups - groups), (0, 0))).astype(jnp.float32)
    cache = kv_pages.reshape(*kv_pages.shape[:2], heads, 2, dim)
    scale = jnp.asarray(sm_scale, jnp.float32).reshape(1)
    output_shape = jax.ShapeDtypeStruct(padded.shape, q.dtype)

    def cost_math(query, key, value):
        scores = query @ key.T
        weights = _exp_nonpositive(scores - scores.max(axis=1, keepdims=True))
        return (weights @ value) / weights.sum(axis=1, keepdims=True)

    cost = pl.estimate_cost(
        cost_math,
        jax.ShapeDtypeStruct((padded_groups, dim), jnp.float32),
        jax.ShapeDtypeStruct((page_size, dim), jnp.float32),
        jax.ShapeDtypeStruct((page_size, dim), jnp.float32),
    )
    work = tokens * heads * token_pages.shape[1]
    cost = pl.CostEstimate(flops=cost.flops * work, transcendentals=cost.transcendentals * work, bytes_accessed=0)
    cost = with_io_bytes_accessed(
        cost,
        kernel_inputs_specs=(token_pages, bounds, padded, cache, scale),
        kernel_outputs_specs=output_shape,
    )
    query_spec = pl.BlockSpec((None, None, padded_groups, dim), lambda token, head, *_: (token, head, 0, 0))
    with jax.default_matmul_precision("highest"):
        result = pl.pallas_call(
            _decode_kernel,
            out_shape=output_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(tokens, heads),
                in_specs=(query_spec, pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pltpu.VMEM)),
                out_specs=query_spec,
                scratch_shapes=(pltpu.VMEM((page_size, 2, dim), kv_pages.dtype), pltpu.SemaphoreType.DMA),
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
            cost_estimate=cost,
            interpret=interpret,
        )(token_pages, bounds, padded, cache, scale)
    return result[:, :, :groups]
