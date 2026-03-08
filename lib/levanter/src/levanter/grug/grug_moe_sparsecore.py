# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""SparseCore-assisted dispatch helpers for Grug MoE benchmarks.

This module intentionally keeps SparseCore use narrow and benchmark-oriented.
Routing metadata stays on the existing XLA path while SparseCore performs the
activation pack step via indexed gathers, matching the supported
`plsc.BlockSpec(indexed_by=...)` pattern from the JAX SparseCore tutorial.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

_MAX_GATHER_BLOCK_SIZE = 128
_VMEM_WORD_BUDGET = 120_000


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _sparsecore_lane_width() -> int:
    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore dispatch requires TPU backend")
    device_kind = jax.devices()[0].device_kind
    if device_kind in ("TPU v5", "TPU v5p", "TPU v6", "TPU v6 lite"):
        return 8
    if device_kind == "TPU7x":
        return 16
    raise NotImplementedError(f"SparseCore dispatch is unsupported on device kind {device_kind!r}")


def _bitcast_rows_for_sparsecore(
    x: jax.Array,
) -> tuple[jax.Array, Callable[[jax.Array], jax.Array]]:
    if x.ndim != 2:
        raise ValueError(f"SparseCore row gather expects rank-2 input, got {x.shape}")

    if x.dtype in (jnp.float32, jnp.int32):
        return x, lambda y: y

    if x.dtype == jnp.bfloat16:
        if x.shape[1] % 2 != 0:
            raise ValueError(
                "SparseCore bf16 gather requires an even hidden size so the "
                f"minor dimension can be bitcast to int32; got shape={x.shape}"
            )
        return x.view(jnp.int32), lambda y: y.view(jnp.bfloat16)

    raise NotImplementedError(f"SparseCore dispatch does not support dtype {x.dtype}")


def _gather_block_size(row_width: int) -> int:
    explicit_block_size = _env_int("GRUG_SPARSECORE_GATHER_BLOCK_SIZE")
    explicit_word_budget = _env_int("GRUG_SPARSECORE_GATHER_VMEM_WORD_BUDGET")
    word_budget = explicit_word_budget if explicit_word_budget is not None else _VMEM_WORD_BUDGET

    if explicit_block_size is not None:
        if explicit_block_size <= 0:
            raise ValueError("GRUG_SPARSECORE_GATHER_BLOCK_SIZE must be positive, " f"got {explicit_block_size}")
        if explicit_block_size > _MAX_GATHER_BLOCK_SIZE:
            raise ValueError(
                "GRUG_SPARSECORE_GATHER_BLOCK_SIZE exceeds the hard maximum; "
                f"got {explicit_block_size}, max={_MAX_GATHER_BLOCK_SIZE}"
            )
        if 4 * explicit_block_size * row_width > word_budget:
            raise ValueError(
                "GRUG_SPARSECORE_GATHER_BLOCK_SIZE exceeds the configured VMEM word budget; "
                f"got block_size={explicit_block_size}, row_width={row_width}, budget={word_budget}"
            )
        return explicit_block_size

    # The indexed `plsc.BlockSpec` path is internally double-buffered. Keep the
    # tile small enough to fit per-subcore VMEM on v5p/v6 while still using a
    # reasonably large gather window.
    for block_size in (128, 64, 32, 16, 8, 4, 2, 1):
        if block_size <= _MAX_GATHER_BLOCK_SIZE and 4 * block_size * row_width <= word_budget:
            return block_size
    raise ValueError(f"SparseCore gather row width {row_width} exceeds the VMEM budget {word_budget}")


def _sparsecore_row_gather_impl(x: jax.Array, row_indices: jax.Array) -> jax.Array:
    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore dispatch requires TPU backend")
    if row_indices.ndim != 1:
        raise ValueError(f"row_indices must be rank-1, got {row_indices.shape}")
    if row_indices.dtype != jnp.int32:
        row_indices = row_indices.astype(jnp.int32)

    x_sc, restore = _bitcast_rows_for_sparsecore(x)
    row_width = int(x_sc.shape[1])
    lane_width = _sparsecore_lane_width()
    block_size = _gather_block_size(row_width)
    if row_width % lane_width != 0:
        raise ValueError(
            "SparseCore gather requires the gathered row width to be divisible "
            f"by the SparseCore lane width {lane_width}; got width={row_width}"
        )

    num_indices = int(row_indices.shape[0])
    if num_indices == 0:
        return x[:0]

    pad = (-num_indices) % block_size
    if pad:
        row_indices = jnp.pad(row_indices, ((0, pad),), constant_values=0)
    padded_num_indices = num_indices + pad

    @partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((padded_num_indices, row_width), x_sc.dtype),
        grid=(padded_num_indices // block_size,),
        in_specs=(
            plsc.BlockSpec(
                (block_size, row_width),
                indexed_by=1,
                indexed_dim=0,
            ),
            pl.BlockSpec((block_size,), lambda i: i),
        ),
        out_specs=pl.BlockSpec((block_size, row_width), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
            dimension_semantics=(pltpu.PARALLEL,),
        ),
    )
    def kernel(gathered_ref, _indices_ref, out_ref):
        @pl.loop(0, block_size)
        def _(row):
            @pl.loop(0, out_ref.shape[1], step=lane_width)
            def _(col):
                slc = (pl.ds(row, 1), pl.ds(col, lane_width))
                out_ref.at[*slc][...] = gathered_ref.at[*slc][...]

    gathered = kernel(x_sc, row_indices)
    gathered = gathered[:num_indices]
    return restore(gathered)


@jax.custom_vjp
def sparsecore_row_gather(x: jax.Array, row_indices: jax.Array) -> jax.Array:
    """Gathers rows from ``x`` using SparseCore in the forward pass."""

    return _sparsecore_row_gather_impl(x, row_indices)


def _sparsecore_row_gather_fwd(x: jax.Array, row_indices: jax.Array):
    gathered = _sparsecore_row_gather_impl(x, row_indices)
    return gathered, (row_indices, x.shape[0])


def _sparsecore_row_gather_bwd(res, g):
    row_indices, num_rows = res
    dx = jnp.zeros((num_rows, g.shape[1]), dtype=g.dtype).at[row_indices].add(g, mode="drop")
    return dx, None


sparsecore_row_gather.defvjp(_sparsecore_row_gather_fwd, _sparsecore_row_gather_bwd)


@jax.custom_vjp
def sparsecore_row_gather_bf16_prebitcast(
    x_bf16: jax.Array,
    x_i32: jax.Array,
    row_indices: jax.Array,
) -> jax.Array:
    """Benchmark-only bf16 gather that reuses a pre-bitcast int32 activation view.

    The forward path gathers from ``x_i32`` so callers can bitcast once outside
    a chunk loop. The backward path is defined with respect to ``x_bf16`` so we
    preserve the activation gradient even though the forward uses the aliased
    int32 view.
    """

    if x_bf16.dtype != jnp.bfloat16:
        raise ValueError(f"x_bf16 must be bf16, got {x_bf16.dtype}")
    if x_i32.dtype != jnp.int32:
        raise ValueError(f"x_i32 must be int32, got {x_i32.dtype}")
    if x_bf16.shape[0] != x_i32.shape[0] or x_bf16.shape[1] != x_i32.shape[1] * 2:
        raise ValueError(
            "x_bf16 and x_i32 must be matching bf16/int32 views; " f"got {x_bf16.shape=} and {x_i32.shape=}"
        )
    gathered_i32 = _sparsecore_row_gather_impl(x_i32, row_indices)
    return gathered_i32.view(jnp.bfloat16)


def _sparsecore_row_gather_bf16_prebitcast_fwd(
    x_bf16: jax.Array,
    x_i32: jax.Array,
    row_indices: jax.Array,
):
    gathered = _sparsecore_row_gather_impl(x_i32, row_indices).view(jnp.bfloat16)
    return gathered, (row_indices, x_bf16.shape[0])


def _sparsecore_row_gather_bf16_prebitcast_bwd(res, g):
    row_indices, num_rows = res
    dx = jnp.zeros((num_rows, g.shape[1]), dtype=g.dtype).at[row_indices].add(g, mode="drop")
    return dx, None, None


sparsecore_row_gather_bf16_prebitcast.defvjp(
    _sparsecore_row_gather_bf16_prebitcast_fwd,
    _sparsecore_row_gather_bf16_prebitcast_bwd,
)


def sparsecore_row_scatter_add_transpose(updates: jax.Array, row_indices: jax.Array, *, num_rows: int) -> jax.Array:
    """Scatter-add rows by transposing the underlying SparseCore gather lowering.

    This bypasses the custom VJP on ``sparsecore_row_gather`` so we can directly
    test whether JAX's transpose of the SC pallas call yields a better
    scatter-add implementation than the default XLA lowering.
    """

    if updates.ndim != 2:
        raise ValueError(f"updates must be rank-2, got {updates.shape}")
    if row_indices.ndim != 1:
        raise ValueError(f"row_indices must be rank-1, got {row_indices.shape}")
    if updates.shape[0] != row_indices.shape[0]:
        raise ValueError(
            f"updates and row_indices must agree on the leading dimension; got {updates.shape=} and {row_indices.shape=}"
        )

    if updates.dtype in (jnp.float32, jnp.int32):
        input_aval = jax.ShapeDtypeStruct((num_rows, updates.shape[1]), updates.dtype)
        try:
            transpose = jax.linear_transpose(lambda x: _sparsecore_row_gather_impl(x, row_indices), input_aval)
            (dx,) = transpose(updates)
        except (AssertionError, NotImplementedError) as exc:
            raise RuntimeError(
                "SparseCore scatter-add via transpose is unsupported on this JAX stack; "
                "the underlying pallas/SparseCore gather lowering does not expose a usable transpose rule."
            ) from exc
        return dx

    if updates.dtype == jnp.bfloat16:
        if updates.shape[1] % 2 != 0:
            raise ValueError(
                "SparseCore bf16 scatter-add requires an even hidden size so the "
                f"minor dimension can be bitcast to int32; got shape={updates.shape}"
            )
        updates_sc = updates.view(jnp.int32)
        input_aval_sc = jax.ShapeDtypeStruct((num_rows, updates_sc.shape[1]), jnp.int32)
        try:
            transpose = jax.linear_transpose(
                lambda x_sc: _sparsecore_row_gather_impl(x_sc, row_indices),
                input_aval_sc,
            )
            (dx_sc,) = transpose(updates_sc)
        except (AssertionError, NotImplementedError) as exc:
            raise RuntimeError(
                "SparseCore bf16 scatter-add via transpose is unsupported on this JAX stack; "
                "bf16 payloads can be bitcast into SC-friendly int32 lanes, but the transposed pallas "
                "gather path still lacks the required transpose support."
            ) from exc
        return dx_sc.view(jnp.bfloat16)

    raise NotImplementedError(f"SparseCore scatter-add transpose does not support dtype {updates.dtype}")


def sparsecore_row_scatter_set_unique(updates: jax.Array, row_indices: jax.Array, *, num_rows: int) -> jax.Array:
    """Scatter rows with unique indices using the documented SC indexed-copy path.

    This is benchmark-only and intentionally narrow:
    - overwrite semantics only
    - assumes `row_indices` contains no duplicates
    - uses a single SC kernel instance over the whole buffer so we can validate
      whether SC scatter itself looks promising on this stack
    """

    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore scatter requires TPU backend")
    if updates.ndim != 2:
        raise ValueError(f"updates must be rank-2, got {updates.shape}")
    if row_indices.ndim != 1:
        raise ValueError(f"row_indices must be rank-1, got {row_indices.shape}")
    if updates.shape[0] != row_indices.shape[0]:
        raise ValueError(
            f"updates and row_indices must agree on the leading dimension; got {updates.shape=} and {row_indices.shape=}"
        )
    if row_indices.dtype != jnp.int32:
        row_indices = row_indices.astype(jnp.int32)
    row_indices_2d = jnp.broadcast_to(row_indices[:, None], (row_indices.shape[0], _sparsecore_lane_width()))

    num_updates, row_width = updates.shape
    if num_updates == 0:
        return jnp.zeros((num_rows, row_width), dtype=updates.dtype)

    @partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((num_rows, row_width), updates.dtype),
        grid=(),
        in_specs=(
            pl.BlockSpec((num_updates, row_width), lambda: (0, 0)),
            pl.BlockSpec((num_updates, _sparsecore_lane_width()), lambda: (0, 0)),
        ),
        out_specs=pl.BlockSpec((num_rows, row_width), lambda: (0, 0), memory_space=pl.MemorySpace.ANY),
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
            dimension_semantics=(),
        ),
    )
    def kernel(updates_hbm, indices_hbm, out_hbm):
        @pl.loop(0, out_hbm.shape[0])
        def _(row):
            @pl.loop(0, out_hbm.shape[1])
            def _(col):
                out_hbm.at[row, col][...] = jnp.zeros((), dtype=out_hbm.dtype)

        def body(update_vmem, index_vmem):
            row_index = index_vmem.at[0, 0][...]
            pltpu.sync_copy(update_vmem.at[0], out_hbm.at[row_index])

        pltpu.emit_pipeline(
            body,
            grid=(num_updates,),
            in_specs=(
                pl.BlockSpec((1, row_width), lambda row: (row, 0)),
                pl.BlockSpec((1, _sparsecore_lane_width()), lambda row: (row, 0)),
            ),
            out_specs=(),
            dimension_semantics=(pltpu.PARALLEL,),
        )(updates_hbm, indices_hbm)

    try:
        return kernel(updates, row_indices_2d)
    except Exception as exc:
        raise RuntimeError(
            "SparseCore unique-index scatter could not be lowered on this JAX 0.8 stack. "
            "The documented indexed HBM scatter pattern still compiled into a local-to-local DMA "
            "on v5p for this benchmark helper."
        ) from exc


def sparsecore_prepare_dispatch(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    *,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare grouped per-expert dispatch buffers with SC-assisted activation packing."""

    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore dispatch requires TPU backend")

    tokens, topk = selected_experts.shape
    expert_ids = selected_experts.reshape(tokens * topk)
    dispatch_weights = combine_weights.reshape(tokens * topk)

    sort_idx = jnp.argsort(expert_ids, axis=0)
    token_ids_sort = (jnp.arange(tokens * topk, dtype=jnp.int32) // topk)[sort_idx]
    x_sort = sparsecore_row_gather(x, token_ids_sort)
    w_sort = dispatch_weights[sort_idx].astype(x.dtype)
    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return x_sort, w_sort, token_ids_sort, group_sizes


__all__ = [
    "sparsecore_prepare_dispatch",
    "sparsecore_row_gather_bf16_prebitcast",
    "sparsecore_row_gather",
    "sparsecore_row_scatter_add_transpose",
    "sparsecore_row_scatter_set_unique",
]
