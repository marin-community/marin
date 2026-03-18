# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..ssd.reference import (
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    ssd_chunked_from_local_blocks_reference_batched,
)
from ..ssd.xla import ssd_chunk_state_xla_batched, ssd_intra_chunk_xla_batched
from .reference import (
    prepare_mamba3_chunked_scales,
)


def mamba3_intra_chunk_xla_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    out_correction: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    c: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups chunk value"]:
    """Plain-JAX Mamba-3 local block on the transformed `g` recurrence."""

    acc_dtype = jnp.float32
    y = ssd_intra_chunk_xla_batched(a_log_cumsum, src_scale, b, c, x).astype(acc_dtype)
    diag_cb = jnp.sum(c.astype(acc_dtype) * b.astype(acc_dtype), axis=-1)
    correction = (out_correction.astype(acc_dtype) * diag_cb)[:, :, None] * x.astype(acc_dtype)
    return (y - correction).astype(x.dtype)


def mamba3_chunk_state_xla_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups value state"]:
    """Plain-JAX chunk-state accumulation."""

    return ssd_chunk_state_xla_batched(a_log_cumsum, src_scale, b, x)


def mamba3_chunked_forward_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Chunked Mamba-3 forward pass in plain JAX/XLA."""

    local_output = jax.vmap(
        mamba3_intra_chunk_xla_batched,
        in_axes=(1, 1, 1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, out_correction, b, c, x)
    chunk_state = jax.vmap(mamba3_chunk_state_xla_batched, in_axes=(1, 1, 1, 1), out_axes=1)(
        a_log_cumsum,
        src_scale,
        b,
        x,
    )
    return ssd_chunked_from_local_blocks_reference_batched(a_log_cumsum, c, local_output, chunk_state)


def mamba3_chunked_forward_native_xla_batched(
    dt: Float[Array, "groups chunks chunk"],
    lam: Float[Array, "groups chunks chunk"],
    a: Float[Array, "groups chunks chunk"] | Float[Array, "groups chunks"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """XLA fast path on native Mamba-3 parameters."""

    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    return mamba3_chunked_forward_xla_batched(a_log_cumsum, src_scale, out_correction, b, c, x)
