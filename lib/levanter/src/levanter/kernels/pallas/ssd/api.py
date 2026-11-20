# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Literal, TypeAlias

from jaxtyping import Array, Float

from .reference import (
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    ssd_chunk_state_reference_batched,
    ssd_chunked_forward_reference_batched,
    ssd_chunked_sequential_reference_batched,
    ssd_intra_chunk_reference_batched,
)
from .xla import ssd_chunk_state_xla_batched, ssd_chunked_forward_xla_batched, ssd_intra_chunk_xla_batched

Implementation: TypeAlias = Literal["xla", "reference"]


IMPLEMENTATIONS = {
    "reference": ssd_intra_chunk_reference_batched,
    "xla": ssd_intra_chunk_xla_batched,
}
_DEFAULT_IMPLEMENTATION: Implementation = "xla"


def _flatten_intra_chunk_inputs(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    c: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
) -> tuple[
    tuple[
        Float[Array, "groups chunk"],
        Float[Array, "groups chunk"],
        Float[Array, "groups chunk state"],
        Float[Array, "groups chunk state"],
        Float[Array, "groups chunk value"],
    ],
    tuple[int, ...],
]:
    if a_log_cumsum.ndim < 1:
        raise ValueError(f"`a_log_cumsum` must be at least rank-1, got {a_log_cumsum.shape}.")
    if src_scale.shape != a_log_cumsum.shape:
        raise ValueError("`src_scale` must match `a_log_cumsum`.")
    if b.ndim < 2 or c.ndim < 2 or x.ndim < 2:
        raise ValueError("`b`, `c`, and `x` must be at least rank-2.")

    leading_shape = a_log_cumsum.shape[:-1]
    chunk_size = a_log_cumsum.shape[-1]
    if b.shape[:-2] != leading_shape or c.shape[:-2] != leading_shape or x.shape[:-2] != leading_shape:
        raise ValueError("All inputs must share the same leading batch/group axes.")
    if b.shape[-2] != chunk_size or c.shape[-2] != chunk_size or x.shape[-2] != chunk_size:
        raise ValueError("All inputs must share the same chunk axis.")

    groups = math.prod(leading_shape) if leading_shape else 1
    flat_inputs = (
        a_log_cumsum.reshape(groups, chunk_size),
        src_scale.reshape(groups, chunk_size),
        b.reshape(groups, chunk_size, b.shape[-1]),
        c.reshape(groups, chunk_size, c.shape[-1]),
        x.reshape(groups, chunk_size, x.shape[-1]),
    )
    return flat_inputs, leading_shape


def ssd_intra_chunk(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    c: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
    *,
    implementation: Implementation | None = None,
) -> Float[Array, "... chunk value"]:
    """Dispatch the SSD intra-chunk block to the requested backend."""

    flat_inputs, leading_shape = _flatten_intra_chunk_inputs(a_log_cumsum, src_scale, b, c, x)
    impl = implementation if implementation is not None else _DEFAULT_IMPLEMENTATION
    fn = IMPLEMENTATIONS.get(impl)
    if fn is None:
        raise ValueError(f"Unsupported SSD implementation: {impl}.")
    y = fn(*flat_inputs)
    return y.reshape(leading_shape + y.shape[-2:])


def ssd_chunk_state(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
) -> Float[Array, "... value state"]:
    """Compute chunk-end SSD state accumulation."""

    if a_log_cumsum.ndim < 1:
        raise ValueError(f"`a_log_cumsum` must be at least rank-1, got {a_log_cumsum.shape}.")
    leading_shape = a_log_cumsum.shape[:-1]
    chunk_size = a_log_cumsum.shape[-1]
    if src_scale.shape != a_log_cumsum.shape:
        raise ValueError("`src_scale` must match `a_log_cumsum`.")
    if b.shape[:-2] != leading_shape or x.shape[:-2] != leading_shape:
        raise ValueError("`b` and `x` must share the same leading axes as `a_log_cumsum`.")
    if b.shape[-2] != chunk_size or x.shape[-2] != chunk_size:
        raise ValueError("`b` and `x` must share the same chunk axis as `a_log_cumsum`.")

    groups = math.prod(leading_shape) if leading_shape else 1
    y = ssd_chunk_state_xla_batched(
        a_log_cumsum.reshape(groups, chunk_size),
        src_scale.reshape(groups, chunk_size),
        b.reshape(groups, chunk_size, b.shape[-1]),
        x.reshape(groups, chunk_size, x.shape[-1]),
    )
    return y.reshape(leading_shape + y.shape[-2:])


def ssd_chunked_forward(
    a_log_cumsum: Float[Array, "... chunks chunk"],
    src_scale: Float[Array, "... chunks chunk"],
    b: Float[Array, "... chunks chunk state"],
    c: Float[Array, "... chunks chunk state"],
    x: Float[Array, "... chunks chunk value"],
    *,
    implementation: Implementation | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Chunked SSD forward pass with an XLA-first local block dispatch."""

    if a_log_cumsum.ndim < 2 or src_scale.shape != a_log_cumsum.shape:
        raise ValueError("`a_log_cumsum` and `src_scale` must have shape `[..., chunks, chunk]`.")
    leading_shape = a_log_cumsum.shape[:-2]
    groups = math.prod(leading_shape) if leading_shape else 1
    num_chunks, chunk_size = a_log_cumsum.shape[-2:]
    flat_a_log_cumsum = a_log_cumsum.reshape(groups, num_chunks, chunk_size)
    flat_src_scale = src_scale.reshape(groups, num_chunks, chunk_size)
    flat_b = b.reshape(groups, num_chunks, chunk_size, b.shape[-1])
    flat_c = c.reshape(groups, num_chunks, chunk_size, c.shape[-1])
    flat_x = x.reshape(groups, num_chunks, chunk_size, x.shape[-1])

    impl = implementation if implementation is not None else _DEFAULT_IMPLEMENTATION
    if impl == "reference":
        y, final_state = ssd_chunked_forward_reference_batched(
            flat_a_log_cumsum, flat_src_scale, flat_b, flat_c, flat_x
        )
    elif impl == "xla":
        y, final_state = ssd_chunked_forward_xla_batched(flat_a_log_cumsum, flat_src_scale, flat_b, flat_c, flat_x)
    else:
        raise ValueError(f"Unsupported SSD implementation: {impl}.")

    return y.reshape(leading_shape + y.shape[-3:]), final_state.reshape(leading_shape + final_state.shape[-2:])


__all__ = [
    "IMPLEMENTATIONS",
    "Implementation",
    "intra_chunk_log_alpha_cumsum",
    "local_log_alpha",
    "ssd_chunked_forward",
    "ssd_chunked_forward_reference_batched",
    "ssd_chunked_sequential_reference_batched",
    "ssd_chunk_state",
    "ssd_chunk_state_reference_batched",
    "ssd_intra_chunk",
    "ssd_intra_chunk_reference_batched",
]
