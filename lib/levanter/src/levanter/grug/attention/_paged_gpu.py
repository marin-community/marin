# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU attention for Levanter's interleaved KV pages.

Decode follows JAX's Pallas GPU split-K algorithm. Prefill gathers the
sequence's logical pages for the segmented FA4 forward kernel.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention._fa4_cute_backend import segmented_flash_attention_forward
from levanter.grug.attention._fa4_cute_config import (
    SM100_GQA_RATIOS,
    SM100_HEAD_DIM,
    flash4_cute_kernel_config,
    sm100_flash4_cute_kernel_config,
)
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

_MIN_DOT_TILE = 16


def gpu_paged_prefill(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_len: jax.Array,
    page_ids: jax.Array,
    q_len: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None,
) -> jax.Array:
    """Run causal FA4 on an active sequence suffix and its reordered cached prefix."""
    query_capacity, heads, groups, head_dim = q.shape
    page_size = kv_pages.shape[1]
    kv_capacity = page_ids.size * page_size
    pages = kv_pages[jnp.maximum(page_ids, 0)]
    positions = jnp.arange(kv_capacity).reshape(page_ids.size, page_size)
    valid_pages = (page_ids[:, None] >= 0) & (positions < kv_len[0])
    pages = jnp.where(valid_pages[:, :, None, None], pages, 0)
    cache = pages.reshape(1, kv_capacity, 2 * heads, head_dim)

    offset = jnp.maximum(0, kv_len - q_len)
    tokens = jnp.arange(query_capacity)[None, :]
    valid = tokens < q_len
    lower_bounds = (
        jnp.zeros_like(tokens) if sliding_window is None else jnp.maximum(0, offset + tokens + 1 - sliding_window)
    )
    # Skip FA4 key tiles for padded query blocks.
    lower_bounds = jnp.where(valid, lower_bounds, kv_capacity)
    arch = gpu_compute_capability()
    kernel_config = (
        sm100_flash4_cute_kernel_config()
        if arch // 10 == 10 and head_dim == SM100_HEAD_DIM and groups in SM100_GQA_RATIOS
        else flash4_cute_kernel_config(head_dim, arch=arch)
    )
    output, _ = segmented_flash_attention_forward(
        q.reshape(1, query_capacity, heads * groups, head_dim),
        cache[:, :, 0::2],
        cache[:, :, 1::2],
        lower_bounds,
        valid,
        softmax_scale=sm_scale,
        kernel_config=kernel_config,
        q_offset=offset,
    )
    return jnp.where(valid.reshape(-1, 1, 1, 1), output.reshape(q.shape), 0)


class _AttentionState(NamedTuple):
    output: jax.Array
    denominator: jax.Array
    maximum: jax.Array


def _attention_values(probabilities, values):
    # Preserve FP32 softmax precision in BF16 tensor-core matmuls.
    high = probabilities.astype(jnp.bfloat16)
    residual = probabilities - high.astype(jnp.float32)
    middle = residual.astype(jnp.bfloat16)
    low = (residual - middle.astype(jnp.float32)).astype(jnp.bfloat16)
    return (plgpu.dot(low, values) + plgpu.dot(middle, values)) + plgpu.dot(high, values)


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
    split_count,
    soft_cap,
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
    visible_start = lower // page_size
    visible_end = jnp.where(upper > lower, pl.cdiv(upper, page_size), visible_start)
    # Spread the pages in the live window across splits, independent of its offset.
    pages_per_split = pl.cdiv(visible_end - visible_start, split_count)
    first_page = visible_start + split * pages_per_split
    last_page = jnp.minimum(first_page + pages_per_split, visible_end)
    slots = jnp.arange(page_size)
    initial = _AttentionState(
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
        output = correction[:, None] * output + _attention_values(probabilities, v)
        return _AttentionState(output, correction * denominator + probabilities.sum(axis=-1), next_maximum)

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
    q: jax.Array,
    kv_pages: jax.Array,
    token_pages: jax.Array,
    bounds: jax.Array,
    sm_scale: float,
    *,
    max_splits: int,
    soft_cap: float | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Compute local paged attention; call within a KV-head shard_map.

    q is [tokens, kv_heads, groups, head_dim], token_pages is [tokens, pages],
    and bounds contains inclusive lower/exclusive upper token positions.
    Empty ranges produce zero. CPU interpret mode exercises the same kernel.
    """
    tokens, heads, groups, dim = q.shape
    page_size = kv_pages.shape[1]
    if page_size < _MIN_DOT_TILE or page_size & (page_size - 1) or dim < _MIN_DOT_TILE or dim & (dim - 1):
        raise ValueError("GPU paged attention needs power-of-two page_size and head_dim, both at least 16")
    if not interpret and jax.default_backend() != "gpu":
        raise ValueError("GPU paged attention requires a GPU")
    padded_groups = max(_MIN_DOT_TILE, pl.next_power_of_2(groups))
    num_splits = min(max_splits, pl.next_power_of_2(token_pages.shape[1]))
    scale = jnp.asarray(sm_scale, jnp.float32)
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
        body_cost, kernel_inputs_specs=(q, kv_pages, token_pages, bounds, scale), kernel_outputs_specs=out_shape
    )
    output, denominator, maximum = pl.pallas_call(
        partial(
            _page_kernel,
            page_size=page_size,
            split_count=num_splits,
            soft_cap=soft_cap,
            padded_groups=padded_groups,
        ),
        grid=(num_splits, tokens, heads),
        in_specs=(
            pl.BlockSpec(q.shape, lambda s, t, h: (0, 0, 0, 0)),
            pl.BlockSpec(kv_pages.shape, lambda s, t, h: (0, 0, 0, 0)),
            pl.BlockSpec((None, token_pages.shape[1]), lambda s, t, h: (t, 0)),
            pl.BlockSpec((None, 2), lambda s, t, h: (t, 0)),
            pl.BlockSpec(),
        ),
        out_specs=(
            pl.BlockSpec((None, None, None, padded_groups, dim), lambda s, t, h: (s, t, h, 0, 0)),
            pl.BlockSpec((None, None, None, padded_groups), lambda s, t, h: (s, t, h, 0)),
            pl.BlockSpec((None, None, None, padded_groups), lambda s, t, h: (s, t, h, 0)),
        ),
        out_shape=out_shape,
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        cost_estimate=cost,
        name="grug_paged_decode",
    )(q, kv_pages, token_pages, bounds, scale)
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
