# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
from jaxtyping import Array, Float

from .reference import (
    ssd_chunk_state_reference_batched,
    ssd_chunked_from_local_blocks_reference_batched,
    ssd_intra_chunk_reference_batched,
)


def ssd_intra_chunk_xla_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    c: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups chunk value"]:
    """Plain-JAX SSD intra-chunk implementation used as the default backend."""

    return ssd_intra_chunk_reference_batched(a_log_cumsum, src_scale, b, c, x)


def ssd_chunk_state_xla_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups value state"]:
    """Plain-JAX chunk-state accumulation."""

    return ssd_chunk_state_reference_batched(a_log_cumsum, src_scale, b, x)


def ssd_chunked_forward_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Chunked SSD forward pass in plain JAX/XLA."""

    local_output = jax.vmap(
        ssd_intra_chunk_xla_batched,
        in_axes=(1, 1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, b, c, x)
    chunk_state = jax.vmap(
        ssd_chunk_state_xla_batched,
        in_axes=(1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, b, x)
    return ssd_chunked_from_local_blocks_reference_batched(a_log_cumsum, c, local_output, chunk_state)
