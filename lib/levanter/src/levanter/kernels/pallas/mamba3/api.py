# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal, TypeAlias, cast
import warnings

import jax
from jaxtyping import Array, Float

from .config import BlockSizes
from .reference import (
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    mamba3_chunk_state_reference_batched,
    mamba3_chunked_forward_reference_batched,
    mamba3_chunked_sequential_reference_batched,
    mamba3_direct_recurrence_reference_batched,
    mamba3_intra_chunk_reference_batched,
    prepare_mamba3_chunked_scales,
    prepare_mamba3_scales,
)
from .xla import mamba3_chunk_state_xla_batched, mamba3_chunked_forward_xla_batched, mamba3_intra_chunk_xla_batched


Implementation: TypeAlias = Literal["pallas_tpu", "xla", "reference"]


class PallasUnsupportedError(NotImplementedError):
    """Raised when the optional Pallas Mamba-3 backend is explicitly requested but unavailable."""


def mamba3_intra_chunk_pallas(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    out_correction: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    c: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
    *,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> Float[Array, "groups chunk value"]:
    del a_log_cumsum, src_scale, out_correction, b, c, x, block_sizes, interpret, backend
    raise PallasUnsupportedError("Mamba-3 TPU Pallas kernel is intentionally absent; use the XLA path.")


IMPLEMENTATIONS = {
    "reference": mamba3_intra_chunk_reference_batched,
    "xla": mamba3_intra_chunk_xla_batched,
    "pallas_tpu": mamba3_intra_chunk_pallas,
}
_DEFAULT_IMPLEMENTATIONS: tuple[Implementation, ...] = ("xla",)
_PALLAS_FALLBACK_WARNINGS_EMITTED: set[str] = set()


def _warn_pallas_fallback_once(exc: Exception) -> None:
    message = str(exc)
    if message in _PALLAS_FALLBACK_WARNINGS_EMITTED:
        return
    _PALLAS_FALLBACK_WARNINGS_EMITTED.add(message)
    warnings.warn(f"Mamba-3 Pallas kernel unavailable, falling back to XLA: {message}", RuntimeWarning)


def _flatten_intra_chunk_inputs(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    out_correction: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    c: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[int, ...]]:
    if a_log_cumsum.ndim < 1:
        raise ValueError(f"`a_log_cumsum` must be at least rank-1, got {a_log_cumsum.shape}.")
    if src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("`src_scale` and `out_correction` must match `a_log_cumsum`.")
    if b.ndim < 2 or c.ndim < 2 or x.ndim < 2:
        raise ValueError("`b`, `c`, and `x` must be at least rank-2.")

    leading_shape = a_log_cumsum.shape[:-1]
    chunk_size = a_log_cumsum.shape[-1]
    if b.shape[:-2] != leading_shape or c.shape[:-2] != leading_shape or x.shape[:-2] != leading_shape:
        raise ValueError("All inputs must share the same leading batch/group axes.")
    if b.shape[-2] != chunk_size or c.shape[-2] != chunk_size or x.shape[-2] != chunk_size:
        raise ValueError("All inputs must share the same chunk axis.")

    groups = math.prod(leading_shape) if leading_shape else 1
    return (
        a_log_cumsum.reshape(groups, chunk_size),
        src_scale.reshape(groups, chunk_size),
        out_correction.reshape(groups, chunk_size),
        b.reshape(groups, chunk_size, b.shape[-1]),
        c.reshape(groups, chunk_size, c.shape[-1]),
        x.reshape(groups, chunk_size, x.shape[-1]),
    ), leading_shape


def mamba3_intra_chunk(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    out_correction: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    c: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> Float[Array, "... chunk value"]:
    """Dispatch the quadratic intra-chunk Mamba-3 block to the requested backend."""

    flat_inputs, leading_shape = _flatten_intra_chunk_inputs(a_log_cumsum, src_scale, out_correction, b, c, x)
    if implementation is None:
        impls: Sequence[Implementation] = _DEFAULT_IMPLEMENTATIONS
    elif isinstance(implementation, Sequence) and not isinstance(implementation, (str, bytes)):
        impls = cast(Sequence[Implementation], implementation)
    else:
        impls = (cast(Implementation, implementation),)

    errors: list[Exception] = []
    for impl in impls:
        fn = IMPLEMENTATIONS[impl]
        try:
            if impl == "pallas_tpu":
                y = fn(*flat_inputs, block_sizes=block_sizes, interpret=interpret, backend=backend)
            else:
                y = fn(*flat_inputs)
            return y.reshape(leading_shape + y.shape[-2:])
        except PallasUnsupportedError as exc:
            errors.append(exc)
            if len(impls) == 1:
                raise
            _warn_pallas_fallback_once(exc)

    raise ExceptionGroup("all Mamba-3 intra-chunk implementations failed", errors)


def mamba3_chunk_state(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
) -> Float[Array, "... value state"]:
    """Compute chunk-end transformed state accumulation."""

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
    y = mamba3_chunk_state_xla_batched(
        a_log_cumsum.reshape(groups, chunk_size),
        src_scale.reshape(groups, chunk_size),
        b.reshape(groups, chunk_size, b.shape[-1]),
        x.reshape(groups, chunk_size, x.shape[-1]),
    )
    return y.reshape(leading_shape + y.shape[-2:])


def mamba3_chunked_forward_from_transformed(
    a_log_cumsum: Float[Array, "... chunks chunk"],
    src_scale: Float[Array, "... chunks chunk"],
    out_correction: Float[Array, "... chunks chunk"],
    b: Float[Array, "... chunks chunk state"],
    c: Float[Array, "... chunks chunk state"],
    x: Float[Array, "... chunks chunk value"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Chunked Mamba-3 forward pass on transformed inputs."""

    if a_log_cumsum.ndim < 2 or src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("Expected transformed inputs with shape `[..., chunks, chunk]`.")
    leading_shape = a_log_cumsum.shape[:-2]
    groups = math.prod(leading_shape) if leading_shape else 1
    num_chunks, chunk_size = a_log_cumsum.shape[-2:]
    flat_a_log_cumsum = a_log_cumsum.reshape(groups, num_chunks, chunk_size)
    flat_src_scale = src_scale.reshape(groups, num_chunks, chunk_size)
    flat_out_correction = out_correction.reshape(groups, num_chunks, chunk_size)
    flat_b = b.reshape(groups, num_chunks, chunk_size, b.shape[-1])
    flat_c = c.reshape(groups, num_chunks, chunk_size, c.shape[-1])
    flat_x = x.reshape(groups, num_chunks, chunk_size, x.shape[-1])

    if implementation == "reference":
        y, final_state = mamba3_chunked_forward_reference_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x,
        )
    elif implementation is None or implementation == "xla":
        y, final_state = mamba3_chunked_forward_xla_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x,
        )
    else:
        local_output = jax.vmap(
            lambda a_i, s_i, q_i, b_i, c_i, x_i: mamba3_intra_chunk(
                a_i,
                s_i,
                q_i,
                b_i,
                c_i,
                x_i,
                implementation=implementation,
                block_sizes=block_sizes,
                interpret=interpret,
                backend=backend,
            ),
            in_axes=(1, 1, 1, 1, 1, 1),
            out_axes=1,
        )(flat_a_log_cumsum, flat_src_scale, flat_out_correction, flat_b, flat_c, flat_x)
        chunk_state = jax.vmap(mamba3_chunk_state_xla_batched, in_axes=(1, 1, 1, 1), out_axes=1)(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_b,
            flat_x,
        )
        from ..ssd.reference import ssd_chunked_from_local_blocks_reference_batched

        y, final_state = ssd_chunked_from_local_blocks_reference_batched(
            flat_a_log_cumsum, flat_c, local_output, chunk_state
        )

    return y.reshape(leading_shape + y.shape[-3:]), final_state.reshape(leading_shape + final_state.shape[-2:])


def mamba3_chunked_forward(
    dt: Float[Array, "... chunks chunk"],
    lam: Float[Array, "... chunks chunk"],
    a: Float[Array, "... chunks chunk"] | Float[Array, "... chunks"],
    b: Float[Array, "... chunks chunk state"],
    c: Float[Array, "... chunks chunk state"],
    x: Float[Array, "... chunks chunk value"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Stable Mamba-3 entrypoint on native chunked inputs."""

    if implementation == "reference":
        leading_shape = dt.shape[:-2]
        groups = math.prod(leading_shape) if leading_shape else 1
        y, final_state = mamba3_direct_recurrence_reference_batched(
            dt.reshape(groups, dt.shape[-2], dt.shape[-1]),
            lam.reshape(groups, lam.shape[-2], lam.shape[-1]),
            a.reshape(groups, a.shape[-2], a.shape[-1]) if a.shape == dt.shape else a.reshape(groups, a.shape[-1]),
            b.reshape(groups, b.shape[-3], b.shape[-2], b.shape[-1]),
            c.reshape(groups, c.shape[-3], c.shape[-2], c.shape[-1]),
            x.reshape(groups, x.shape[-3], x.shape[-2], x.shape[-1]),
        )
        return y.reshape(leading_shape + y.shape[-3:]), final_state.reshape(leading_shape + final_state.shape[-2:])

    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    return mamba3_chunked_forward_from_transformed(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        implementation=implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        backend=backend,
    )


__all__ = [
    "BlockSizes",
    "IMPLEMENTATIONS",
    "Implementation",
    "PallasUnsupportedError",
    "intra_chunk_log_alpha_cumsum",
    "local_log_alpha",
    "mamba3_chunk_state",
    "mamba3_chunk_state_reference_batched",
    "mamba3_chunked_forward",
    "mamba3_chunked_forward_from_transformed",
    "mamba3_chunked_forward_reference_batched",
    "mamba3_chunked_sequential_reference_batched",
    "mamba3_direct_recurrence_reference_batched",
    "mamba3_intra_chunk",
    "mamba3_intra_chunk_pallas",
    "mamba3_intra_chunk_reference_batched",
    "prepare_mamba3_chunked_scales",
    "prepare_mamba3_scales",
]
