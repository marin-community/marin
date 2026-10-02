# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Split-K paged decode following JAX's Pallas GPU paged-attention algorithm.

The implementation reads the interleaved cache directly and adds a lower
attention bound for sliding windows. This backend is forward-only.
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

GpuPagedAvPrecision = Literal["ieee", "bf16_3x"]


def _attention_values(probabilities, values, precision):
    if precision == "bf16_3x" and values.dtype == jnp.bfloat16:
        # Reconstruct FP32 weights from three BF16 components. Values already have
        # BF16 precision, so three tensor-core dots replace the FP32 IEEE dot.
        high = probabilities.astype(jnp.bfloat16)
        residual = probabilities - high.astype(jnp.float32)
        middle = residual.astype(jnp.bfloat16)
        low = (residual - middle.astype(jnp.float32)).astype(jnp.bfloat16)
        return (plgpu.dot(low, values) + plgpu.dot(middle, values)) + plgpu.dot(high, values)
    return plgpu.dot(probabilities, values.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)


def _page_kernel(
    q_ref,
    cache_ref,
    table_ref,
    bounds_ref,
    scale_ref,
    out_ref,
    sum_ref,
    max_ref,
    *,
    page_size,
    pages_per_split,
    soft_cap,
    av_precision,
    padded_groups,
):
    split = pl.program_id(0)
    head = pl.program_id(2)
    group = jnp.arange(padded_groups)
    dims = jnp.arange(q_ref.shape[-1])
    q = plgpu.load(
        q_ref.at[pl.program_id(1), head, group[:, None], dims[None, :]],
        mask=group[:, None] < q_ref.shape[2],
        other=0,
    )
    lower, upper = bounds_ref[0], bounds_ref[1]
    first_page = jnp.maximum(split * pages_per_split, lower // page_size)
    last_page = jnp.minimum((split + 1) * pages_per_split, pl.cdiv(upper, page_size))
    last_page = jnp.where(upper > lower, last_page, first_page)
    slots = jnp.arange(page_size)
    initial = (
        jnp.zeros(q.shape, jnp.float32),
        jnp.zeros(q.shape[0], jnp.float32),
        jnp.full((q.shape[0],), -jnp.inf, jnp.float32),
    )

    def body(page, carry):
        output, denominator, maximum = carry
        physical = table_ref[page]
        k = cache_ref[physical, slots[:, None], 2 * head, dims[None, :]]
        v = cache_ref[physical, slots[:, None], 2 * head + 1, dims[None, :]]
        logits = plgpu.dot(q, k.T, precision=jax.lax.Precision.HIGHEST) * scale_ref[()]
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)
        position = page * page_size + slots
        allowed = (position >= lower) & (position < upper)
        logits = jnp.where(allowed[None, :], logits, -jnp.inf)
        next_maximum = jnp.maximum(maximum, jnp.max(logits, axis=-1))
        correction = jnp.exp(maximum - next_maximum)
        probabilities = jnp.exp(logits - next_maximum[:, None])
        v = jnp.where(allowed[:, None], v, 0)
        output = correction[:, None] * output + _attention_values(probabilities, v, av_precision)
        return output, correction * denominator + probabilities.sum(axis=-1), next_maximum

    output, denominator, maximum = jax.lax.fori_loop(first_page, last_page, body, initial)
    out_ref[...] = output
    sum_ref[...] = denominator
    max_ref[...] = maximum


def _combine_splits(output_ref, denominator_ref, maximum_ref, result_ref):
    output, denominator, maximum = output_ref[...], denominator_ref[...], maximum_ref[...]
    maximum = jnp.where(denominator > 0, maximum, -jnp.inf)
    global_maximum = maximum.max()
    global_maximum = jnp.where(jnp.isfinite(global_maximum), global_maximum, 0)
    correction = jnp.exp(maximum - global_maximum)
    total = (denominator * correction).sum()
    result = (output * correction[:, None]).sum(axis=0) / jnp.where(total > 0, total, 1)
    result_ref[...] = result.astype(result_ref.dtype)


def gpu_paged_attention(
    q,
    kv_pages,
    token_pages,
    bounds,
    sm_scale,
    *,
    soft_cap=None,
    kv_splits=8,
    av_precision: GpuPagedAvPrecision = "ieee",
    interpret=False,
):
    """Compute local paged attention; call within a KV-head shard_map.

    q is [tokens, kv_heads, groups, head_dim], token_pages is [tokens, pages],
    and bounds contains inclusive lower/exclusive upper token positions.
    Empty ranges produce zero. CPU interpret mode exercises the same kernel.
    """
    tokens, heads, groups, dim = q.shape
    page_size = kv_pages.shape[1]
    if page_size < 16 or page_size & (page_size - 1) or dim < 16 or dim & (dim - 1):
        raise ValueError("GPU paged attention needs power-of-two page_size and head_dim, both at least 16")
    if not interpret and jax.default_backend() != "gpu":
        raise ValueError("GPU paged attention requires a GPU")
    if av_precision not in ("ieee", "bf16_3x"):
        raise ValueError(f"Unknown AV precision: {av_precision}")
    padded_groups = max(16, pl.next_power_of_2(groups))
    if kv_splits not in (8, 16):
        raise ValueError("GPU paged attention supports 8 or 16 KV splits")
    num_splits = min(kv_splits, pl.next_power_of_2(token_pages.shape[1]))
    pages_per_split = pl.cdiv(token_pages.shape[1], num_splits)
    padded_pages = num_splits * pages_per_split
    token_pages = jnp.pad(token_pages, ((0, 0), (0, padded_pages - token_pages.shape[1])))
    sm_scale = jnp.asarray(sm_scale, jnp.float32)
    out_shape = (
        jax.ShapeDtypeStruct((num_splits, tokens, heads, padded_groups, dim), jnp.float32),
        jax.ShapeDtypeStruct((num_splits, tokens, heads, padded_groups), jnp.float32),
        jax.ShapeDtypeStruct((num_splits, tokens, heads, padded_groups), jnp.float32),
    )

    def cost_math(query, key, value):
        logits = query @ key.T
        weights = jax.nn.softmax(logits, axis=-1)
        return weights @ value

    body_cost = pl.estimate_cost(
        cost_math,
        jax.ShapeDtypeStruct((padded_groups, dim), q.dtype),
        jax.ShapeDtypeStruct((page_size, dim), q.dtype),
        jax.ShapeDtypeStruct((page_size, dim), q.dtype),
    )
    page_work = tokens * heads * token_pages.shape[1]
    body_cost = pl.CostEstimate(
        flops=body_cost.flops * page_work,
        transcendentals=body_cost.transcendentals * page_work,
        bytes_accessed=0,
    )
    cost = with_io_bytes_accessed(
        body_cost, kernel_inputs_specs=(q, kv_pages, token_pages, bounds, sm_scale), kernel_outputs_specs=out_shape
    )
    output, denominator, maximum = pl.pallas_call(
        partial(
            _page_kernel,
            page_size=page_size,
            pages_per_split=pages_per_split,
            soft_cap=soft_cap,
            av_precision=av_precision,
            padded_groups=padded_groups,
        ),
        grid=(num_splits, tokens, heads),
        in_specs=(
            pl.BlockSpec(q.shape, lambda s, t, h: (0, 0, 0, 0)),
            pl.BlockSpec(kv_pages.shape, lambda s, t, h: (0, 0, 0, 0)),
            pl.BlockSpec((None, padded_pages), lambda s, t, h: (t, 0)),
            pl.BlockSpec((None, 2), lambda s, t, h: (t, 0)),
            pl.BlockSpec(),
        ),
        out_specs=(
            pl.BlockSpec((None, None, None, padded_groups, dim), lambda s, t, h: (s, t, h, 0, 0)),
            pl.BlockSpec((None, None, None, padded_groups), lambda s, t, h: (s, t, h, 0)),
            pl.BlockSpec((None, None, None, padded_groups), lambda s, t, h: (s, t, h, 0)),
        ),
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        interpret=interpret,
        cost_estimate=cost,
        name="grug_paged_decode",
    )(q, kv_pages, token_pages, bounds, sm_scale)
    result_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    combine_cost = with_io_bytes_accessed(
        pl.CostEstimate(
            flops=tokens * heads * groups * num_splits * (2 * dim + 3),
            transcendentals=tokens * heads * groups * num_splits,
            bytes_accessed=0,
        ),
        kernel_inputs_specs=(output, denominator, maximum),
        kernel_outputs_specs=(result_shape,),
    )
    return pl.pallas_call(
        _combine_splits,
        grid=(tokens, heads, groups),
        in_specs=(
            pl.BlockSpec((num_splits, None, None, None, dim), lambda t, h, g: (0, t, h, g, 0)),
            pl.BlockSpec((num_splits, None, None, None), lambda t, h, g: (0, t, h, g)),
            pl.BlockSpec((num_splits, None, None, None), lambda t, h, g: (0, t, h, g)),
        ),
        out_specs=pl.BlockSpec((None, None, None, dim), lambda t, h, g: (t, h, g, 0)),
        out_shape=result_shape,
        compiler_params=plgpu.CompilerParams(num_warps=4),
        interpret=interpret,
        cost_estimate=combine_cost,
        name="grug_paged_combine_splits",
    )(output, denominator, maximum)
