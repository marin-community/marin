# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward-only TPU paged decode with FP32 tile math and accurate softmax."""

from functools import partial
from math import factorial, log
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

TPU_PAGE_ALIGNMENT = 16
TPU_HEAD_ALIGNMENT = 128
_QUERY_GROUP_ALIGNMENT = 8

_INVERSE_LOG_2 = 1 / log(2)
_MIN_NORMAL_LOG = -126 * log(2)
_EXP_DEGREE = 12
_FLOAT32_EXPONENT_BIAS = 127
_FLOAT32_MANTISSA_BITS = 23
_DEKKER_LOW_SIGNIFICAND_BITS = 12
_DEKKER_HIGH_MASK = -(1 << _DEKKER_LOW_SIGNIFICAND_BITS)


class _FloatPair(NamedTuple):
    high: jax.Array
    low: jax.Array


class _AttentionState(NamedTuple):
    numerator: _FloatPair
    denominator: _FloatPair
    maximum: jax.Array


class _PageTerms(NamedTuple):
    numerator: _FloatPair
    denominator: _FloatPair


def _two_sum(a, b):
    high = a + b
    b_virtual = high - a
    return _FloatPair(high, (a - (high - b_virtual)) + (b - b_virtual))


def _two_product(a, b):
    # Dekker splitting recovers FP32 multiplication's rounding residual.
    a_high = jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(a, jnp.int32) & _DEKKER_HIGH_MASK, jnp.float32)
    b_high = jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(b, jnp.int32) & _DEKKER_HIGH_MASK, jnp.float32)
    a_low, b_low = a - a_high, b - b_high
    high = a * b
    low = ((a_high * b_high - high) + a_high * b_low + a_low * b_high) + a_low * b_low
    return _FloatPair(high, low)


def _add(a, b):
    result = _two_sum(a.high, b.high)
    return _two_sum(result.high, result.low + a.low + b.low)


def _scale(a, b):
    result = _two_product(a.high, b)
    return _two_sum(result.high, result.low + a.low * b)


def _multiply(a, b):
    result = _two_product(a.high, b.high)
    return _two_sum(result.high, result.low + a.high * b.low + a.low * b.high + a.low * b.low)


def _divide(a, b):
    quotient = a.high / b.high
    product = _scale(b, quotient)
    remainder = _add(a, _FloatPair(-product.high, -product.low))
    return _two_sum(quotient, (remainder.high + remainder.low) / b.high)


def _constant(value, like):
    high = float(np.float32(value))
    return _FloatPair(jnp.full_like(like, high), jnp.full_like(like, value - high))


def _exp_nonpositive(x):
    # Retain range-reduction and polynomial residuals. A degree-12 polynomial
    # on [-log(2)/2, log(2)/2] keeps truncation below 2e-16 relative error.
    finite = jnp.isfinite(x.high)
    safe = _FloatPair(jnp.where(finite, x.high, 0.0), jnp.where(finite, x.low, 0.0))
    exponent = jnp.floor(jnp.maximum(safe.high, _MIN_NORMAL_LOG) * _INVERSE_LOG_2 + 0.5)
    remainder = _add(safe, _scale(_constant(-log(2), safe.high), exponent))
    polynomial = _constant(1 / factorial(_EXP_DEGREE), safe.high)
    for degree in range(_EXP_DEGREE - 1, -1, -1):
        polynomial = _add(_multiply(polynomial, remainder), _constant(1 / factorial(degree), safe.high))
    power = jax.lax.bitcast_convert_type(
        (exponent.astype(jnp.int32) + _FLOAT32_EXPONENT_BIAS) << _FLOAT32_MANTISSA_BITS, jnp.float32
    )
    supported = finite & (safe.high >= _MIN_NORMAL_LOG)
    return _FloatPair(
        jnp.where(supported, polynomial.high * power, 0.0),
        jnp.where(supported, polynomial.low * power, 0.0),
    )


def _fixed_high(value):
    # The leading component shares a binary quantum along the dot's reduction
    # axis. Its integers fit in eight bits, keeping 128-term leading products
    # exactly representable in an FP32 accumulator.
    magnitude = jnp.max(jnp.abs(value), axis=1, keepdims=True)
    bits = jax.lax.bitcast_convert_type(magnitude, jnp.int32)
    exponent = jnp.maximum(((bits >> _FLOAT32_MANTISSA_BITS) & 255) - _FLOAT32_EXPONENT_BIAS + 1 - 8, -126)
    quantum = jax.lax.bitcast_convert_type((exponent + _FLOAT32_EXPONENT_BIAS) << _FLOAT32_MANTISSA_BITS, jnp.float32)
    # Both scales are normal powers of two throughout the finite BF16 range.
    # Explicit multiplication avoids Mosaic's general FP32 division lowering.
    inverse = jax.lax.bitcast_convert_type((_FLOAT32_EXPONENT_BIAS - exponent) << _FLOAT32_MANTISSA_BITS, jnp.float32)
    return jnp.round(value * inverse) * quantum


def _query_key_components(value):
    high = _fixed_high(value).astype(jnp.bfloat16)
    low = (value - high.astype(jnp.float32)).astype(jnp.bfloat16)
    return high, low


def _query_key_dot(query, key, query_dtype, key_dtype):
    if query_dtype != jnp.bfloat16 or key_dtype != jnp.bfloat16:
        high = jnp.dot(query, key.T, preferred_element_type=jnp.float32)
        return _FloatPair(high, jnp.zeros_like(high))
    with jax.default_matmul_precision("default"):
        products = [
            jnp.dot(q, k.T, preferred_element_type=jnp.float32)
            for q in _query_key_components(query)
            for k in _query_key_components(key)
        ]
    return _add(_two_sum(products[0], products[1]), _two_sum(products[2], products[3]))


def _page_terms(weights, value):
    if value.dtype == jnp.bfloat16:
        high = weights.high.astype(jnp.bfloat16)
        residual = weights.high - high.astype(jnp.float32)
        middle = residual.astype(jnp.bfloat16)
        low = (residual - middle.astype(jnp.float32)).astype(jnp.bfloat16)
        with jax.default_matmul_precision("default"):
            products = [jnp.dot(w, value, preferred_element_type=jnp.float32) for w in (high, middle, low)]
        numerator = _add(_two_sum(products[0], products[1]), _FloatPair(products[2], jnp.zeros_like(products[2])))
        sums = [jnp.sum(w.astype(jnp.float32), axis=1, keepdims=True) for w in (high, middle, low)]
        denominator = _add(_two_sum(sums[0], sums[1]), _FloatPair(sums[2], jnp.zeros_like(sums[2])))
    else:
        # Highest-precision FP32 dots retain FP32 cache values without a
        # redundant conversion through the BF16 component implementation.
        numerator_high = jnp.dot(weights.high, value, preferred_element_type=jnp.float32)
        denominator_high = jnp.sum(weights.high, axis=1, keepdims=True)
        numerator = _FloatPair(numerator_high, jnp.zeros_like(numerator_high))
        denominator = _FloatPair(denominator_high, jnp.zeros_like(denominator_high))
    low_num = jnp.dot(weights.low, value.astype(jnp.float32), preferred_element_type=jnp.float32)
    low_den = jnp.sum(weights.low, axis=1, keepdims=True)
    return _PageTerms(
        _add(numerator, _FloatPair(low_num, jnp.zeros_like(low_num))),
        _add(denominator, _FloatPair(low_den, jnp.zeros_like(low_den))),
    )


def _round_output(result, dtype):
    if dtype != jnp.bfloat16:
        return (result.high + result.low).astype(dtype)
    bits = jax.lax.bitcast_convert_type(result.high, jnp.int32)
    midpoint = (bits & 65535) == 32768
    direction = jnp.where((result.low > 0) == (result.high > 0), 1, -1)
    adjusted = jax.lax.bitcast_convert_type(bits + direction, jnp.float32)
    # A low word resolves an exact BF16 tie before the final conversion.
    return jnp.where(midpoint & (result.low != 0), adjusted, result.high).astype(dtype)


def _decode_kernel(
    table_ref,
    bounds_ref,
    q_ref,
    cache_ref,
    scale_ref,
    output_ref,
    page_buffers,
    semaphores,
    *,
    dma_buffers,
    query_dtype,
):
    token, head = pl.program_id(0), pl.program_id(1)
    lower, upper = bounds_ref[token, 0], bounds_ref[token, 1]
    page_size = page_buffers.shape[1]
    query = q_ref[...].astype(jnp.float32)
    initial = _AttentionState(
        _constant(0, query),
        _constant(0, query[:, :1]),
        jnp.full((query.shape[0], 1), -jnp.inf, jnp.float32),
    )

    def page_copy(page):
        physical = table_ref[token, page]
        buffer = page % dma_buffers
        return pltpu.make_async_copy(cache_ref.at[physical, :, head], page_buffers.at[buffer], semaphores.at[buffer])

    first_page = lower // page_size
    last_page = jnp.where(upper > lower, pl.cdiv(upper, page_size), first_page)
    if dma_buffers == 2:

        @pl.when(first_page < last_page)
        def prefetch_first():
            page_copy(first_page).start()

    def attend_page(page, state):
        copy = page_copy(page)
        if dma_buffers == 1:
            copy.start()
        copy.wait()
        if dma_buffers == 2:

            @pl.when(page + 1 < last_page)
            def prefetch_next():
                page_copy(page + 1).start()

        # Converting first gives Mosaic unpacked FP32 rows for the K/V slice.
        loaded = page_buffers[page % dma_buffers].astype(jnp.float32)
        key, value = loaded[:, 0], loaded[:, 1]
        scores = _scale(_query_key_dot(query, key, query_dtype, page_buffers.dtype), scale_ref[0])
        # Construct masks in each consumer's layout; Mosaic cannot reshape a
        # lane vector into the column vector required to mask cached values.
        position = page * page_size + jax.lax.broadcasted_iota(jnp.int32, scores.high.shape, 1)
        valid = (position >= lower) & (position < upper)
        scores = _FloatPair(jnp.where(valid, scores.high, -jnp.inf), jnp.where(valid, scores.low, 0.0))
        maximum = jnp.maximum(state.maximum, jnp.max(scores.high, axis=1, keepdims=True))
        correction = _exp_nonpositive(_two_sum(state.maximum, -maximum))
        weights = _exp_nonpositive(_add(scores, _FloatPair(-maximum, jnp.zeros_like(maximum))))
        value_position = page * page_size + jax.lax.broadcasted_iota(jnp.int32, value.shape, 0)
        value = jnp.where((value_position >= lower) & (value_position < upper), value, 0.0).astype(page_buffers.dtype)
        terms = _page_terms(weights, value)
        numerator = _add(_multiply(state.numerator, correction), terms.numerator)
        denominator = _add(_multiply(state.denominator, correction), terms.denominator)
        return _AttentionState(numerator, denominator, maximum)

    state = jax.lax.fori_loop(first_page, last_page, attend_page, initial)
    denominator = _FloatPair(jnp.where(state.denominator.high > 0, state.denominator.high, 1), state.denominator.low)
    output_ref[...] = _round_output(_divide(state.numerator, denominator), output_ref.dtype)


def tpu_paged_decode(q, kv_pages, token_pages, bounds, sm_scale, *, dma_buffers=1, interpret=False):
    """Attend to local cache pages inside the caller's KV-head shard_map.

    Queries are [tokens, heads, groups, dim], cache pages are interleaved K/V,
    and bounds are inclusive lower/exclusive upper token positions. Empty bounds
    return zero. Outputs preserve the query dtype. This forward-only kernel requires positive page sizes
    divisible by 16 and head dimensions divisible by 128. Two DMA buffers
    overlap the next page load with current-page math; one preserves serial DMA.
    """
    if dma_buffers not in (1, 2):
        raise ValueError("TPU decode supports one or two DMA buffers")
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
        scores = _scale(_query_key_dot(query, key, q.dtype, kv_pages.dtype), jnp.asarray(sm_scale, jnp.float32))
        maximum = jnp.max(scores.high, axis=1, keepdims=True)
        weights = _exp_nonpositive(_add(scores, _FloatPair(-maximum, jnp.zeros_like(maximum))))
        terms = _page_terms(weights, value.astype(kv_pages.dtype))
        return _round_output(_divide(terms.numerator, terms.denominator), q.dtype)

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
            partial(_decode_kernel, dma_buffers=dma_buffers, query_dtype=q.dtype),
            out_shape=output_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(tokens, heads),
                in_specs=(query_spec, pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pltpu.VMEM)),
                out_specs=query_spec,
                scratch_shapes=(
                    pltpu.VMEM((dma_buffers, page_size, 2, dim), kv_pages.dtype),
                    pltpu.SemaphoreType.DMA((dma_buffers,)),
                ),
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
            cost_estimate=cost,
            interpret=interpret,
        )(token_pages, bounds, padded, cache, scale)
    return result[:, :, :groups]
