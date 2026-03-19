# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..cost_estimate_utils import with_io_bytes_accessed
from .config import BlockSizes


def _local_output_tile_reference(
    a_q: jax.Array,
    a_k: jax.Array,
    src_k: jax.Array,
    b_k: jax.Array,
    c_q: jax.Array,
    x_k: jax.Array,
) -> jax.Array:
    acc_dtype = jnp.float32
    q_block_size = a_q.shape[0]
    k_block_size = a_k.shape[0]
    q_index = jnp.arange(q_block_size, dtype=jnp.int32)
    k_index = jnp.arange(k_block_size, dtype=jnp.int32)
    mask = q_index[:, None] >= k_index[None, :]
    decay = jnp.exp(jnp.where(mask, a_q.astype(acc_dtype)[:, None] - a_k.astype(acc_dtype)[None, :], -jnp.inf))
    cb = jnp.dot(c_q.astype(acc_dtype), b_k.astype(acc_dtype).T, preferred_element_type=acc_dtype)
    x_scaled = x_k.astype(acc_dtype) * src_k.astype(acc_dtype)[:, None]
    return jnp.dot(cb * decay, x_scaled, preferred_element_type=acc_dtype)


def _ssd_intra_chunk_cost_estimate(
    *,
    a_q_spec,
    a_k_spec,
    src_k_spec,
    b_k_spec,
    c_q_spec,
    x_k_spec,
    out_spec,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(
        _local_output_tile_reference,
        jax.ShapeDtypeStruct(a_q_spec.shape[1:], a_q_spec.dtype),
        jax.ShapeDtypeStruct(a_k_spec.shape[1:], a_k_spec.dtype),
        jax.ShapeDtypeStruct(src_k_spec.shape[1:], src_k_spec.dtype),
        jax.ShapeDtypeStruct(b_k_spec.shape[1:], b_k_spec.dtype),
        jax.ShapeDtypeStruct(c_q_spec.shape[1:], c_q_spec.dtype),
        jax.ShapeDtypeStruct(x_k_spec.shape[1:], x_k_spec.dtype),
    )
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=(a_q_spec, a_k_spec, src_k_spec, b_k_spec, c_q_spec, x_k_spec),
        kernel_outputs_specs=(out_spec,),
    )


def _ssd_intra_chunk_kernel(
    a_q_ref,
    a_k_ref,
    src_k_ref,
    b_k_ref,
    c_q_ref,
    x_k_ref,
    out_ref,
    acc_ref,
    *,
    q_block_size: int,
    k_block_size: int,
    num_k_blocks: int,
):
    k_block_index = pl.program_id(3)
    q_start = pl.program_id(1) * q_block_size
    k_start = k_block_index * k_block_size
    acc_dtype = acc_ref.dtype

    @pl.when(k_block_index == 0)
    def init() -> None:
        acc_ref[...] = jnp.zeros_like(acc_ref)

    a_q = a_q_ref[0].astype(acc_dtype)
    a_k = a_k_ref[0].astype(acc_dtype)
    src_k = src_k_ref[0].astype(acc_dtype)
    b_k = b_k_ref[0].astype(acc_dtype)
    c_q = c_q_ref[0].astype(acc_dtype)
    x_k = x_k_ref[0].astype(acc_dtype)

    q_index = q_start + jnp.arange(q_block_size, dtype=jnp.int32)
    k_index = k_start + jnp.arange(k_block_size, dtype=jnp.int32)
    mask = q_index[:, None] >= k_index[None, :]
    decay = jnp.exp(jnp.where(mask, a_q[:, None] - a_k[None, :], -jnp.inf))
    cb = jnp.dot(c_q, b_k.T, preferred_element_type=acc_dtype)
    x_scaled = x_k * src_k[:, None]
    acc_ref[...] = acc_ref[...] + jnp.dot(cb * decay, x_scaled, preferred_element_type=acc_dtype)

    @pl.when(k_block_index == num_k_blocks - 1)
    def store() -> None:
        out_ref[0, ...] = acc_ref[...].astype(out_ref.dtype)


def _validate_block_sizes(
    *,
    a_log_cumsum: jax.Array,
    x: jax.Array,
    block_sizes: BlockSizes,
    interpret: bool,
) -> tuple[int, int, int]:
    if a_log_cumsum.ndim != 2 or x.ndim != 3:
        raise ValueError("Pallas SSD local block expects flattened inputs [G, C], [G, C, P].")
    if block_sizes.group_block_size != 1:
        raise NotImplementedError(
            f"group_block_size must be 1 for this prototype, got {block_sizes.group_block_size}."
        )
    if (
        block_sizes.query_block_size is None
        or block_sizes.key_block_size is None
        or block_sizes.value_block_size is None
    ):
        raise ValueError("query/key/value block sizes must be set for the TPU Pallas prototype.")

    chunk_size = a_log_cumsum.shape[-1]
    value_dim = x.shape[-1]
    query_block_size = block_sizes.query_block_size
    key_block_size = block_sizes.key_block_size
    value_block_size = block_sizes.value_block_size

    if chunk_size % query_block_size != 0:
        raise NotImplementedError(f"chunk_size={chunk_size} must be divisible by query_block_size={query_block_size}.")
    if chunk_size % key_block_size != 0:
        raise NotImplementedError(f"chunk_size={chunk_size} must be divisible by key_block_size={key_block_size}.")
    if value_dim % value_block_size != 0:
        raise NotImplementedError(f"value_dim={value_dim} must be divisible by value_block_size={value_block_size}.")

    if not interpret:
        for name, size in (
            ("query_block_size", query_block_size),
            ("key_block_size", key_block_size),
            ("value_block_size", value_block_size),
        ):
            if size % 128 != 0:
                raise NotImplementedError(f"{name} must be a multiple of 128 on TPU, got {size}.")

    return query_block_size, key_block_size, value_block_size


@partial(jax.jit, static_argnames=("block_sizes", "interpret", "backend"))
def ssd_intra_chunk_pallas_tpu(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    c: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
    *,
    block_sizes: BlockSizes,
    interpret: bool = False,
    backend: str | None = None,
) -> Float[Array, "groups chunk value"]:
    """TPU Pallas prototype for the SSD intra-chunk local block."""

    if backend is not None and backend != "tpu":
        raise NotImplementedError(f"Unsupported backend for TPU Pallas kernel: {backend}.")
    if not interpret and jax.default_backend() != "tpu":
        raise NotImplementedError("TPU Pallas kernel requires a TPU backend unless interpret=True.")
    if a_log_cumsum.shape != src_scale.shape:
        raise ValueError("`src_scale` must match `a_log_cumsum`.")
    if b.shape[:2] != a_log_cumsum.shape or c.shape[:2] != a_log_cumsum.shape or x.shape[:2] != a_log_cumsum.shape:
        raise ValueError("All Pallas SSD inputs must share `[G, C]` leading dimensions.")
    if b.shape != c.shape:
        raise ValueError("`b` and `c` must match for the Pallas SSD local block.")

    query_block_size, key_block_size, value_block_size = _validate_block_sizes(
        a_log_cumsum=a_log_cumsum,
        x=x,
        block_sizes=block_sizes,
        interpret=interpret,
    )

    groups, chunk_size = a_log_cumsum.shape
    state_dim = b.shape[-1]
    value_dim = x.shape[-1]
    num_q_blocks = chunk_size // query_block_size
    num_k_blocks = chunk_size // key_block_size
    num_v_blocks = value_dim // value_block_size
    out_shape = jax.ShapeDtypeStruct((groups, chunk_size, value_dim), x.dtype)
    acc_dtype = jnp.float32

    a_q_spec = jax.ShapeDtypeStruct((1, query_block_size), a_log_cumsum.dtype)
    a_k_spec = jax.ShapeDtypeStruct((1, key_block_size), a_log_cumsum.dtype)
    src_k_spec = jax.ShapeDtypeStruct((1, key_block_size), src_scale.dtype)
    b_k_spec = jax.ShapeDtypeStruct((1, key_block_size, state_dim), b.dtype)
    c_q_spec = jax.ShapeDtypeStruct((1, query_block_size, state_dim), c.dtype)
    x_k_spec = jax.ShapeDtypeStruct((1, key_block_size, value_block_size), x.dtype)
    out_block_spec = jax.ShapeDtypeStruct((1, query_block_size, value_block_size), x.dtype)

    return pl.pallas_call(
        partial(
            _ssd_intra_chunk_kernel,
            q_block_size=query_block_size,
            k_block_size=key_block_size,
            num_k_blocks=num_k_blocks,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(groups, num_q_blocks, num_v_blocks, num_k_blocks),
            in_specs=[
                pl.BlockSpec((1, query_block_size), lambda g, q, v, k: (g, q)),
                pl.BlockSpec((1, key_block_size), lambda g, q, v, k: (g, k)),
                pl.BlockSpec((1, key_block_size), lambda g, q, v, k: (g, k)),
                pl.BlockSpec((1, key_block_size, state_dim), lambda g, q, v, k: (g, k, 0)),
                pl.BlockSpec((1, query_block_size, state_dim), lambda g, q, v, k: (g, q, 0)),
                pl.BlockSpec((1, key_block_size, value_block_size), lambda g, q, v, k: (g, k, v)),
            ],
            out_specs=pl.BlockSpec((1, query_block_size, value_block_size), lambda g, q, v, k: (g, q, v)),
            scratch_shapes=[pltpu.VMEM((query_block_size, value_block_size), acc_dtype)],
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
        interpret=interpret,
        cost_estimate=_ssd_intra_chunk_cost_estimate(
            a_q_spec=a_q_spec,
            a_k_spec=a_k_spec,
            src_k_spec=src_k_spec,
            b_k_spec=b_k_spec,
            c_q_spec=c_q_spec,
            x_k_spec=x_k_spec,
            out_spec=out_block_spec,
        ),
    )(a_log_cumsum, a_log_cumsum, src_scale, b, c, x)


__all__ = ["ssd_intra_chunk_pallas_tpu"]
