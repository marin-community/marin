# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .reference import (
    ssd_chunk_state_reference_batched,
    ssd_intra_chunk_reference_batched,
    ssd_scan_chunk_states_reference_batched,
)


def ssd_intra_chunk_xla_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    c: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups chunk value"]:
    """Plain-JAX SSD intra-chunk implementation used as the default backend."""

    with jax.named_scope("ssd_intra_chunk"):
        return ssd_intra_chunk_reference_batched(a_log_cumsum, src_scale, b, c, x)


def ssd_chunk_state_xla_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups value state"]:
    """Plain-JAX chunk-state accumulation."""

    with jax.named_scope("ssd_chunk_state"):
        return ssd_chunk_state_reference_batched(a_log_cumsum, src_scale, b, x)


def _causal_decay_matrix_chunked(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
) -> Float[Array, "groups chunks chunk chunk"]:
    chunk_size = a_log_cumsum.shape[-1]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))[None, None, :, :]
    diff = a_log_cumsum[..., :, None] - a_log_cumsum[..., None, :]
    return jnp.exp(jnp.where(mask, diff, -jnp.inf))


def ssd_intra_chunk_xla_chunked_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> Float[Array, "groups chunks chunk value"]:
    acc_dtype = jnp.float32
    cb = jnp.einsum(
        "gktn,gksn->gkts",
        c.astype(acc_dtype),
        b.astype(acc_dtype),
        preferred_element_type=acc_dtype,
    )
    decay = _causal_decay_matrix_chunked(a_log_cumsum.astype(acc_dtype))
    x_scaled = x.astype(acc_dtype) * src_scale.astype(acc_dtype)[..., None]
    return jnp.einsum(
        "gkts,gksp->gktp",
        cb * decay,
        x_scaled,
        preferred_element_type=acc_dtype,
    ).astype(x.dtype)


def ssd_chunk_state_xla_chunked_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> Float[Array, "groups chunks value state"]:
    acc_dtype = jnp.float32
    decay_to_end = jnp.exp(a_log_cumsum.astype(acc_dtype)[..., -1:] - a_log_cumsum.astype(acc_dtype))
    return jnp.einsum(
        "gkcn,gkc,gkcp->gkpn",
        b.astype(acc_dtype),
        decay_to_end * src_scale.astype(acc_dtype),
        x.astype(acc_dtype),
        preferred_element_type=acc_dtype,
    ).astype(x.dtype)


def ssd_chunked_from_local_blocks_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    c: Float[Array, "groups chunks chunk state"],
    local_output: Float[Array, "groups chunks chunk value"],
    chunk_state: Float[Array, "groups chunks value state"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    acc_dtype = jnp.float32
    incoming_state, final_state = ssd_scan_chunk_states_reference_batched(
        jnp.exp(a_log_cumsum[..., -1]),
        chunk_state,
    )
    prefix_output = jnp.einsum(
        "gkcn,gkpn,gkc->gkcp",
        c.astype(acc_dtype),
        incoming_state.astype(acc_dtype),
        jnp.exp(a_log_cumsum.astype(acc_dtype)),
        preferred_element_type=acc_dtype,
    ).astype(c.dtype)
    return (local_output + prefix_output).astype(local_output.dtype), final_state


def ssd_chunked_forward_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Chunked SSD forward pass in plain JAX/XLA."""

    with jax.named_scope("ssd_chunked_forward"):
        with jax.named_scope("local_output"):
            local_output = ssd_intra_chunk_xla_chunked_batched(a_log_cumsum, src_scale, b, c, x)
        with jax.named_scope("chunk_state"):
            chunk_state = ssd_chunk_state_xla_chunked_batched(a_log_cumsum, src_scale, b, x)
        with jax.named_scope("prefix_emit"):
            return ssd_chunked_from_local_blocks_xla_batched(a_log_cumsum, c, local_output, chunk_state)
