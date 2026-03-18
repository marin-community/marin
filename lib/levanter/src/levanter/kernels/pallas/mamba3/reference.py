# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..ssd.reference import (
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    ssd_chunk_state_reference_batched,
    ssd_chunked_from_local_blocks_reference_batched,
    ssd_chunked_sequential_reference_batched,
    ssd_intra_chunk_reference_batched,
)


def prepare_mamba3_scales(
    dt: Float[Array, "... chunk"],
    lam: Float[Array, "... chunk"],
) -> tuple[Float[Array, "... chunk"], Float[Array, "... chunk"]]:
    """Build transformed source scale and correction for a single chunk."""

    if dt.shape != lam.shape:
        raise ValueError(f"`dt` and `lam` must have the same shape, got {dt.shape} and {lam.shape}.")
    q = (1.0 - lam) * dt
    q_next = jnp.concatenate([q[..., 1:], jnp.zeros_like(q[..., :1])], axis=-1)
    src_scale = lam * dt + q_next
    return src_scale, q_next


def prepare_mamba3_chunked_scales(
    dt: Float[Array, "... chunks chunk"],
    lam: Float[Array, "... chunks chunk"],
) -> tuple[Float[Array, "... chunks chunk"], Float[Array, "... chunks chunk"]]:
    """Build transformed source scale and correction across chunk boundaries."""

    if dt.shape != lam.shape:
        raise ValueError(f"`dt` and `lam` must have the same shape, got {dt.shape} and {lam.shape}.")
    if dt.ndim < 2:
        raise ValueError(f"Chunked scales require at least rank-2 `[..., chunks, chunk]`, got {dt.shape}.")

    q = (1.0 - lam) * dt
    flat_q = q.reshape(q.shape[:-2] + (math.prod(q.shape[-2:]),))
    flat_q_next = jnp.concatenate([flat_q[..., 1:], jnp.zeros_like(flat_q[..., :1])], axis=-1)
    q_next = flat_q_next.reshape(q.shape)
    src_scale = lam * dt + q_next
    return src_scale, q_next


def mamba3_intra_chunk_reference_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    out_correction: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    c: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups chunk value"]:
    """Reference intra-chunk contraction on the transformed Mamba-3 `g` state."""

    acc_dtype = jnp.float32
    y = ssd_intra_chunk_reference_batched(a_log_cumsum, src_scale, b, c, x).astype(acc_dtype)
    diag_cb = jnp.sum(c.astype(acc_dtype) * b.astype(acc_dtype), axis=-1)
    correction = (out_correction.astype(acc_dtype) * diag_cb)[:, :, None] * x.astype(acc_dtype)
    return (y - correction).astype(x.dtype)


def mamba3_chunk_state_reference_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state"],
    x: Float[Array, "groups chunk value"],
) -> Float[Array, "groups value state"]:
    """Reference chunk-end accumulation for the carried transformed `g` state."""

    return ssd_chunk_state_reference_batched(a_log_cumsum, src_scale, b, x)


def mamba3_chunked_forward_reference_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Reference chunked Mamba-3 forward pass using the SSD chunk scaffolding."""

    local_output = jax.vmap(
        mamba3_intra_chunk_reference_batched,
        in_axes=(1, 1, 1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, out_correction, b, c, x)
    chunk_state = jax.vmap(
        mamba3_chunk_state_reference_batched,
        in_axes=(1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, b, x)
    return ssd_chunked_from_local_blocks_reference_batched(a_log_cumsum, c, local_output, chunk_state)


def mamba3_chunked_sequential_reference_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Sequential transformed-state oracle used to validate the `g_t` rewrite."""

    y, final_state = ssd_chunked_sequential_reference_batched(a_log_cumsum, src_scale, b, c, x)
    acc_dtype = jnp.float32
    diag_cb = jnp.sum(c.astype(acc_dtype) * b.astype(acc_dtype), axis=-1)
    correction = (out_correction.astype(acc_dtype) * diag_cb)[:, :, :, None] * x.astype(acc_dtype)
    return (y.astype(acc_dtype) - correction).astype(x.dtype), final_state


def mamba3_direct_recurrence_reference_batched(
    dt: Float[Array, "groups chunks chunk"],
    lam: Float[Array, "groups chunks chunk"],
    a: Float[Array, "groups chunks chunk"] | Float[Array, "groups chunks"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Direct paper recurrence oracle on the native Mamba-3 parameters."""

    if dt.shape != lam.shape:
        raise ValueError("`dt` and `lam` must have identical shape.")
    if dt.ndim != 3 or b.ndim != 4 or c.ndim != 4 or x.ndim != 4:
        raise ValueError("Expected chunked inputs with shapes [G, K, C] and [G, K, C, *].")
    log_alpha = local_log_alpha(dt, a)
    acc_dtype = jnp.float32
    alpha_tm = jnp.swapaxes(jnp.exp(log_alpha.astype(acc_dtype)).reshape(dt.shape[0], math.prod(dt.shape[1:])), 0, 1)
    beta_tm = jnp.swapaxes(
        ((1.0 - lam.astype(acc_dtype)) * dt.astype(acc_dtype) * jnp.exp(log_alpha.astype(acc_dtype))).reshape(
            dt.shape[0], math.prod(dt.shape[1:])
        ),
        0,
        1,
    )
    gamma_tm = jnp.swapaxes(
        (lam.astype(acc_dtype) * dt.astype(acc_dtype)).reshape(dt.shape[0], math.prod(dt.shape[1:])), 0, 1
    )
    b_tm = jnp.swapaxes(b.astype(acc_dtype).reshape(dt.shape[0], math.prod(dt.shape[1:]), b.shape[-1]), 0, 1)
    c_tm = jnp.swapaxes(c.astype(acc_dtype).reshape(dt.shape[0], math.prod(dt.shape[1:]), c.shape[-1]), 0, 1)
    x_tm = jnp.swapaxes(x.astype(acc_dtype).reshape(dt.shape[0], math.prod(dt.shape[1:]), x.shape[-1]), 0, 1)

    init_h = jnp.zeros((dt.shape[0], x.shape[-1], b.shape[-1]), dtype=acc_dtype)
    init_v = jnp.zeros_like(init_h)

    def step(
        carry: tuple[Float[Array, "groups value state"], Float[Array, "groups value state"]],
        inputs: tuple[
            Float[Array, "groups"],
            Float[Array, "groups"],
            Float[Array, "groups"],
            Float[Array, "groups state"],
            Float[Array, "groups state"],
            Float[Array, "groups value"],
        ],
    ) -> tuple[
        tuple[Float[Array, "groups value state"], Float[Array, "groups value state"]], Float[Array, "groups value"]
    ]:
        h_prev, v_prev = carry
        alpha_t, beta_t, gamma_t, b_t, c_t, x_t = inputs
        v_t = x_t[:, :, None] * b_t[:, None, :]
        h_t = alpha_t[:, None, None] * h_prev + beta_t[:, None, None] * v_prev + gamma_t[:, None, None] * v_t
        y_t = jnp.sum(c_t[:, None, :] * h_t, axis=-1)
        return (h_t, v_t), y_t

    (final_h, _), y_tm = jax.lax.scan(step, (init_h, init_v), (alpha_tm, beta_tm, gamma_tm, b_tm, c_tm, x_tm))
    y = jnp.swapaxes(y_tm, 0, 1).reshape(dt.shape[0], dt.shape[1], dt.shape[2], x.shape[-1])
    return y.astype(x.dtype), final_h.astype(x.dtype)


__all__ = [
    "intra_chunk_log_alpha_cumsum",
    "local_log_alpha",
    "mamba3_chunk_state_reference_batched",
    "mamba3_chunked_forward_reference_batched",
    "mamba3_chunked_sequential_reference_batched",
    "mamba3_direct_recurrence_reference_batched",
    "mamba3_intra_chunk_reference_batched",
    "prepare_mamba3_chunked_scales",
    "prepare_mamba3_scales",
]
