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
    ssd_scan_chunk_states_reference_batched,
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


def mamba3_mimo_rank_expand(
    base: Float[Array, "... value"],
    rank_weights: Float[Array, "... value rank"],
) -> Float[Array, "... value rank"]:
    """Expand a per-value tensor into rank columns using lightweight rank scales."""

    if base.ndim < 1 or rank_weights.ndim < 2:
        raise ValueError("Expected shapes `[..., P]` and `[..., P, R]` for rank expansion.")
    if rank_weights.shape[-2] != base.shape[-1]:
        raise ValueError(f"Rank weights must match the value dim, got {rank_weights.shape} for {base.shape}.")
    expand_singletons = (1,) * (base.ndim - rank_weights.ndim + 1)
    reshaped_weights = rank_weights.reshape(rank_weights.shape[:-2] + expand_singletons + rank_weights.shape[-2:])
    return base[..., :, None] * reshaped_weights


def mamba3_mimo_rank_expand_chunked(
    base: Float[Array, "... chunk value"],
    rank_weights: Float[Array, "... value rank"],
) -> Float[Array, "... rank chunk value"]:
    """Expand chunked tensors into rank-major form so `[chunk, value]` remains the dense inner tile."""

    return jnp.moveaxis(mamba3_mimo_rank_expand(base, rank_weights), -1, -3)


def mamba3_mimo_rank_collapse(
    ranked: Float[Array, "... value rank"],
    rank_weights: Float[Array, "... value rank"],
) -> Float[Array, "... value"]:
    """Collapse rank columns back to the base value width with lightweight rank scales."""

    if ranked.ndim < 2 or rank_weights.ndim < 2:
        raise ValueError("Expected shapes `[..., P, R]` for rank collapse.")
    if rank_weights.shape[-2:] != ranked.shape[-2:]:
        raise ValueError(
            f"Rank weights must match the ranked tensor shape, got {rank_weights.shape} for {ranked.shape}."
        )
    collapse_singletons = (1,) * (ranked.ndim - rank_weights.ndim)
    reshaped_weights = rank_weights.reshape(rank_weights.shape[:-2] + collapse_singletons + rank_weights.shape[-2:])
    return jnp.sum(ranked * reshaped_weights, axis=-1)


def mamba3_mimo_rank_collapse_chunked(
    ranked: Float[Array, "... rank chunk value"],
    rank_weights: Float[Array, "... value rank"],
) -> Float[Array, "... chunk value"]:
    """Collapse rank-major chunked tensors back to base width."""

    return mamba3_mimo_rank_collapse(jnp.moveaxis(ranked, -3, -1), rank_weights)


def mamba3_mimo_apply_gate_and_collapse(
    y_ranked: Float[Array, "... value rank"],
    z_ranked: Float[Array, "... value rank"],
    out_rank_weights: Float[Array, "... value rank"],
) -> Float[Array, "... value"]:
    """Apply the paper's SiLU gating and lightweight rank collapse."""

    gated = y_ranked.astype(jnp.float32) * jax.nn.silu(z_ranked.astype(jnp.float32))
    return mamba3_mimo_rank_collapse(gated, out_rank_weights.astype(jnp.float32)).astype(y_ranked.dtype)


def mamba3_mimo_apply_gate_and_collapse_chunked(
    y_ranked: Float[Array, "... rank chunk value"],
    z_ranked: Float[Array, "... rank chunk value"],
    out_rank_weights: Float[Array, "... value rank"],
) -> Float[Array, "... chunk value"]:
    """Apply the paper's SiLU gating and collapse for rank-major chunked tensors."""

    gated = y_ranked.astype(jnp.float32) * jax.nn.silu(z_ranked.astype(jnp.float32))
    return mamba3_mimo_rank_collapse_chunked(gated, out_rank_weights.astype(jnp.float32)).astype(y_ranked.dtype)


def mamba3_mimo_intra_chunk_reference_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    out_correction: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state rank"],
    c: Float[Array, "groups chunk state rank"],
    x: Float[Array, "groups rank chunk value"],
) -> Float[Array, "groups rank chunk value"]:
    """Reference intra-chunk MIMO contraction on the transformed Mamba-3 `g` state."""

    if a_log_cumsum.ndim != 2 or src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("Expected `[G, C]` schedules for chunked MIMO inputs.")
    if b.ndim != 4 or c.ndim != 4 or x.ndim != 4:
        raise ValueError("Expected rank-major X tensors with shape `[G, R, C, P]`.")
    if b.shape[:2] != a_log_cumsum.shape or c.shape[:2] != a_log_cumsum.shape or x.shape[0] != a_log_cumsum.shape[0]:
        raise ValueError("All MIMO intra-chunk inputs must share the same group leading dimensions.")
    if x.shape[-2] != a_log_cumsum.shape[-1]:
        raise ValueError("Rank-major X tensors must place the chunk axis in the second-most-minor position.")
    if b.shape[-1] != c.shape[-1] or b.shape[-1] != x.shape[-3]:
        raise ValueError("B, C, and X must share the same rank dimension.")

    acc_dtype = jnp.float32
    bc = jnp.einsum(
        "gcnv,gsnu->gcsuv",
        c.astype(acc_dtype),
        b.astype(acc_dtype),
        preferred_element_type=acc_dtype,
    )
    decay = jnp.exp(
        jnp.where(
            jnp.tril(jnp.ones((a_log_cumsum.shape[-1], a_log_cumsum.shape[-1]), dtype=jnp.bool_))[None, :, :],
            a_log_cumsum.astype(acc_dtype)[:, :, None] - a_log_cumsum.astype(acc_dtype)[:, None, :],
            -jnp.inf,
        )
    )
    x_scaled = x.astype(acc_dtype) * src_scale.astype(acc_dtype)[:, None, :, None]
    y = jnp.einsum(
        "gcs,gcsuv,gusp->gvcp",
        decay,
        bc,
        x_scaled,
        preferred_element_type=acc_dtype,
    )
    bc_diag = jnp.einsum(
        "gcnu,gcnv->gcuv",
        b.astype(acc_dtype),
        c.astype(acc_dtype),
        preferred_element_type=acc_dtype,
    )
    correction = out_correction.astype(acc_dtype)[:, None, :, None] * jnp.einsum(
        "gucp,gcuv->gvcp",
        x.astype(acc_dtype),
        bc_diag,
        preferred_element_type=acc_dtype,
    )
    return (y - correction).astype(x.dtype)


def mamba3_mimo_chunk_state_reference_batched(
    a_log_cumsum: Float[Array, "groups chunk"],
    src_scale: Float[Array, "groups chunk"],
    b: Float[Array, "groups chunk state rank"],
    x: Float[Array, "groups rank chunk value"],
) -> Float[Array, "groups value state"]:
    """Reference chunk-end accumulation for the carried transformed MIMO `g` state."""

    if a_log_cumsum.ndim != 2 or src_scale.shape != a_log_cumsum.shape:
        raise ValueError("Expected `[G, C]` schedules for chunk-state accumulation.")
    if b.ndim != 4 or x.ndim != 4:
        raise ValueError("Expected rank-major `x` with shape `[G, R, C, P]`.")
    if (
        b.shape[:2] != a_log_cumsum.shape
        or x.shape[0] != a_log_cumsum.shape[0]
        or x.shape[-2] != a_log_cumsum.shape[1]
    ):
        raise ValueError("Chunk-state inputs must share the same group and chunk dimensions.")
    if b.shape[-1] != x.shape[-3]:
        raise ValueError("B and X must share the same rank dimension.")

    acc_dtype = jnp.float32
    decay_to_end = jnp.exp(a_log_cumsum.astype(acc_dtype)[:, -1:] - a_log_cumsum.astype(acc_dtype))
    return jnp.einsum(
        "gcnu,gucp,gc->gpn",
        b.astype(acc_dtype),
        x.astype(acc_dtype),
        decay_to_end * src_scale.astype(acc_dtype),
        preferred_element_type=acc_dtype,
    ).astype(x.dtype)


def mamba3_mimo_chunked_from_local_blocks_reference_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    c: Float[Array, "groups chunks chunk state rank"],
    local_output: Float[Array, "groups chunks rank chunk value"],
    chunk_state: Float[Array, "groups chunks value state"],
) -> tuple[Float[Array, "groups chunks rank chunk value"], Float[Array, "groups value state"]]:
    """Combine local MIMO outputs with scanned cross-chunk prefix states."""

    if a_log_cumsum.ndim != 3 or c.ndim != 5 or local_output.ndim != 5 or chunk_state.ndim != 4:
        raise ValueError(
            "Expected chunked MIMO inputs with rank-major local outputs `[G, K, R, C, P]` and chunk state `[G, K, P, N]`."
        )
    incoming_state, final_state = ssd_scan_chunk_states_reference_batched(jnp.exp(a_log_cumsum[..., -1]), chunk_state)
    prefix_output = jnp.einsum(
        "gkpn,gkcnv,gkc->gkvcp",
        incoming_state.astype(jnp.float32),
        c.astype(jnp.float32),
        jnp.exp(a_log_cumsum.astype(jnp.float32)),
        preferred_element_type=jnp.float32,
    )
    return (local_output.astype(jnp.float32) + prefix_output).astype(local_output.dtype), final_state


def mamba3_mimo_chunked_forward_ranked_reference_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state rank"],
    c: Float[Array, "groups chunks chunk state rank"],
    x: Float[Array, "groups chunks rank chunk value"],
) -> tuple[Float[Array, "groups chunks rank chunk value"], Float[Array, "groups value state"]]:
    """Reference chunked MIMO forward pass on rank-expanded tensors before gating/collapse."""

    local_output = jax.vmap(
        mamba3_mimo_intra_chunk_reference_batched,
        in_axes=(1, 1, 1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, out_correction, b, c, x)
    chunk_state = jax.vmap(
        mamba3_mimo_chunk_state_reference_batched,
        in_axes=(1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, b, x)
    return mamba3_mimo_chunked_from_local_blocks_reference_batched(a_log_cumsum, c, local_output, chunk_state)


def mamba3_mimo_chunked_sequential_ranked_reference_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state rank"],
    c: Float[Array, "groups chunks chunk state rank"],
    x: Float[Array, "groups chunks rank chunk value"],
) -> tuple[Float[Array, "groups chunks rank chunk value"], Float[Array, "groups value state"]]:
    """Sequential transformed-state MIMO oracle used to validate the chunked decomposition."""

    if a_log_cumsum.ndim != 3 or src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("Expected transformed schedules with shape `[G, K, C]`.")
    if b.ndim != 5 or c.ndim != 5 or x.ndim != 5:
        raise ValueError("Expected rank-major MIMO inputs with shape `[G, K, R, C, P]`.")

    groups, num_chunks, chunk_size = a_log_cumsum.shape
    acc_dtype = jnp.float32
    local_log_alpha_chunked = jnp.concatenate(
        [a_log_cumsum[..., :1], a_log_cumsum[..., 1:] - a_log_cumsum[..., :-1]],
        axis=-1,
    ).astype(acc_dtype)
    alpha_tm = jnp.swapaxes(jnp.exp(local_log_alpha_chunked).reshape(groups, num_chunks * chunk_size), 0, 1)
    src_scale_tm = jnp.swapaxes(src_scale.astype(acc_dtype).reshape(groups, num_chunks * chunk_size), 0, 1)
    out_tm = jnp.swapaxes(out_correction.astype(acc_dtype).reshape(groups, num_chunks * chunk_size), 0, 1)
    b_tm = jnp.swapaxes(b.astype(acc_dtype).reshape(groups, num_chunks * chunk_size, b.shape[-2], b.shape[-1]), 0, 1)
    c_tm = jnp.swapaxes(c.astype(acc_dtype).reshape(groups, num_chunks * chunk_size, c.shape[-2], c.shape[-1]), 0, 1)
    x_tm = jnp.swapaxes(
        jnp.moveaxis(x.astype(acc_dtype), -3, -2).reshape(groups, num_chunks * chunk_size, x.shape[-3], x.shape[-1]),
        0,
        1,
    )

    init_g = jnp.zeros((groups, x.shape[-1], b.shape[-2]), dtype=acc_dtype)

    def step(
        g_prev: Float[Array, "groups value state"],
        inputs: tuple[
            Float[Array, "groups"],
            Float[Array, "groups"],
            Float[Array, "groups"],
            Float[Array, "groups state rank"],
            Float[Array, "groups state rank"],
            Float[Array, "groups rank value"],
        ],
    ) -> tuple[Float[Array, "groups value state"], Float[Array, "groups rank value"]]:
        alpha_t, src_scale_t, out_t, b_t, c_t, x_t = inputs
        v_t = jnp.einsum("gnu,gup->gpn", b_t, x_t, preferred_element_type=acc_dtype)
        g_t = alpha_t[:, None, None] * g_prev + src_scale_t[:, None, None] * v_t
        prefix = jnp.einsum("gpn,gnv->gvp", g_t, c_t, preferred_element_type=acc_dtype)
        bc_t = jnp.einsum("gnu,gnv->guv", b_t, c_t, preferred_element_type=acc_dtype)
        correction = out_t[:, None, None] * jnp.einsum("gup,guv->gvp", x_t, bc_t, preferred_element_type=acc_dtype)
        return g_t, prefix - correction

    final_g, y_tm = jax.lax.scan(step, init_g, (alpha_tm, src_scale_tm, out_tm, b_tm, c_tm, x_tm))
    y = jnp.moveaxis(
        jnp.swapaxes(y_tm, 0, 1).reshape(groups, num_chunks, chunk_size, x.shape[-3], x.shape[-1]),
        -2,
        -3,
    )
    return y.astype(x.dtype), final_g.astype(x.dtype)


def mamba3_mimo_direct_recurrence_ranked_reference_batched(
    dt: Float[Array, "groups chunks chunk"],
    lam: Float[Array, "groups chunks chunk"],
    a: Float[Array, "groups chunks chunk"] | Float[Array, "groups chunks"],
    b: Float[Array, "groups chunks chunk state rank"],
    c: Float[Array, "groups chunks chunk state rank"],
    x: Float[Array, "groups chunks rank chunk value"],
) -> tuple[Float[Array, "groups chunks rank chunk value"], Float[Array, "groups value state"]]:
    """Direct paper recurrence oracle for rank-expanded MIMO inputs."""

    if dt.shape != lam.shape:
        raise ValueError("`dt` and `lam` must have identical shape.")
    if dt.ndim != 3 or b.ndim != 5 or c.ndim != 5 or x.ndim != 5:
        raise ValueError("Expected MIMO inputs with shapes `[G, K, C]` and rank-major `[G, K, R, C, P]`.")

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
    b_tm = jnp.swapaxes(
        b.astype(acc_dtype).reshape(dt.shape[0], math.prod(dt.shape[1:]), b.shape[-2], b.shape[-1]), 0, 1
    )
    c_tm = jnp.swapaxes(
        c.astype(acc_dtype).reshape(dt.shape[0], math.prod(dt.shape[1:]), c.shape[-2], c.shape[-1]), 0, 1
    )
    x_tm = jnp.swapaxes(
        jnp.moveaxis(x.astype(acc_dtype), -3, -2).reshape(
            dt.shape[0], math.prod(dt.shape[1:]), x.shape[-3], x.shape[-1]
        ),
        0,
        1,
    )

    init_h = jnp.zeros((dt.shape[0], x.shape[-1], b.shape[-2]), dtype=acc_dtype)
    init_v = jnp.zeros_like(init_h)

    def step(
        carry: tuple[Float[Array, "groups value state"], Float[Array, "groups value state"]],
        inputs: tuple[
            Float[Array, "groups"],
            Float[Array, "groups"],
            Float[Array, "groups"],
            Float[Array, "groups state rank"],
            Float[Array, "groups state rank"],
            Float[Array, "groups rank value"],
        ],
    ) -> tuple[
        tuple[Float[Array, "groups value state"], Float[Array, "groups value state"]],
        Float[Array, "groups rank value"],
    ]:
        h_prev, v_prev = carry
        alpha_t, beta_t, gamma_t, b_t, c_t, x_t = inputs
        v_t = jnp.einsum("gnu,gup->gpn", b_t, x_t, preferred_element_type=acc_dtype)
        h_t = alpha_t[:, None, None] * h_prev + beta_t[:, None, None] * v_prev + gamma_t[:, None, None] * v_t
        y_t = jnp.einsum("gpn,gnv->gvp", h_t, c_t, preferred_element_type=acc_dtype)
        return (h_t, v_t), y_t

    (final_h, _), y_tm = jax.lax.scan(step, (init_h, init_v), (alpha_tm, beta_tm, gamma_tm, b_tm, c_tm, x_tm))
    y = jnp.moveaxis(
        jnp.swapaxes(y_tm, 0, 1).reshape(dt.shape[0], dt.shape[1], dt.shape[2], x.shape[-3], x.shape[-1]),
        -2,
        -3,
    )
    return y.astype(x.dtype), final_h.astype(x.dtype)


def mamba3_mimo_chunked_forward_reference_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state rank"],
    c: Float[Array, "groups chunks chunk state rank"],
    x_base: Float[Array, "groups chunks chunk value"],
    z_base: Float[Array, "groups chunks chunk value"],
    w_x: Float[Array, "groups value rank"] | Float[Array, "value rank"],
    w_z: Float[Array, "groups value rank"] | Float[Array, "value rank"],
    w_o: Float[Array, "groups value rank"] | Float[Array, "value rank"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Reference chunked MIMO forward pass with lightweight rank expand/gate/collapse."""

    x_ranked = mamba3_mimo_rank_expand_chunked(x_base, w_x)
    z_ranked = mamba3_mimo_rank_expand_chunked(z_base, w_z)
    y_ranked, final_state = mamba3_mimo_chunked_forward_ranked_reference_batched(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x_ranked,
    )
    return mamba3_mimo_apply_gate_and_collapse_chunked(y_ranked, z_ranked, w_o), final_state


def mamba3_mimo_direct_recurrence_reference_batched(
    dt: Float[Array, "groups chunks chunk"],
    lam: Float[Array, "groups chunks chunk"],
    a: Float[Array, "groups chunks chunk"] | Float[Array, "groups chunks"],
    b: Float[Array, "groups chunks chunk state rank"],
    c: Float[Array, "groups chunks chunk state rank"],
    x_base: Float[Array, "groups chunks chunk value"],
    z_base: Float[Array, "groups chunks chunk value"],
    w_x: Float[Array, "groups value rank"] | Float[Array, "value rank"],
    w_z: Float[Array, "groups value rank"] | Float[Array, "value rank"],
    w_o: Float[Array, "groups value rank"] | Float[Array, "value rank"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Direct paper recurrence oracle for the full lightweight-factorized MIMO path."""

    x_ranked = mamba3_mimo_rank_expand_chunked(x_base, w_x)
    z_ranked = mamba3_mimo_rank_expand_chunked(z_base, w_z)
    y_ranked, final_state = mamba3_mimo_direct_recurrence_ranked_reference_batched(dt, lam, a, b, c, x_ranked)
    return mamba3_mimo_apply_gate_and_collapse_chunked(y_ranked, z_ranked, w_o), final_state


__all__ = [
    "intra_chunk_log_alpha_cumsum",
    "local_log_alpha",
    "mamba3_mimo_apply_gate_and_collapse",
    "mamba3_mimo_apply_gate_and_collapse_chunked",
    "mamba3_mimo_chunk_state_reference_batched",
    "mamba3_mimo_chunked_forward_ranked_reference_batched",
    "mamba3_mimo_chunked_forward_reference_batched",
    "mamba3_mimo_chunked_from_local_blocks_reference_batched",
    "mamba3_mimo_chunked_sequential_ranked_reference_batched",
    "mamba3_mimo_direct_recurrence_ranked_reference_batched",
    "mamba3_mimo_direct_recurrence_reference_batched",
    "mamba3_mimo_intra_chunk_reference_batched",
    "mamba3_mimo_rank_collapse",
    "mamba3_mimo_rank_collapse_chunked",
    "mamba3_mimo_rank_expand",
    "mamba3_mimo_rank_expand_chunked",
    "mamba3_chunk_state_reference_batched",
    "mamba3_chunked_forward_reference_batched",
    "mamba3_chunked_sequential_reference_batched",
    "mamba3_direct_recurrence_reference_batched",
    "mamba3_intra_chunk_reference_batched",
    "prepare_mamba3_chunked_scales",
    "prepare_mamba3_scales",
]
