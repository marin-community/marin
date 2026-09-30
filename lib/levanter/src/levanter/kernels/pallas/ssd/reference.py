# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from shape_extensions import IntTuple, IntVar


def local_log_alpha[Batch: IntTuple, C: IntVar](
    dt: jax.Array[[*Batch, C]],
    a: jax.Array[[*Batch, C]] | jax.Array[[*Batch]],
) -> jax.Array[[*Batch, C]]:
    """Compute per-token log decay for a scalar-transition SSD block."""

    if dt.ndim < 1:
        raise ValueError(f"`dt` must be at least rank-1, got {dt.shape}.")
    if a.shape == dt.shape:
        return dt * a
    if a.shape == dt.shape[:-1]:
        # pyrefly: ignore[bad-return]  # the runtime shape check does not narrow `a`'s shape union.
        return dt * a[..., None]
    raise ValueError(f"`a` must match `dt` or `dt` without the token axis, got {a.shape} for {dt.shape}.")


def intra_chunk_log_alpha_cumsum[Batch: IntTuple, C: IntVar](
    log_alpha: jax.Array[[*Batch, C]],
) -> jax.Array[[*Batch, C]]:
    """Compute the inclusive cumulative log decay within each chunk."""

    return jnp.cumsum(log_alpha, axis=-1)


def _validate_ssd_batched_inputs[G: IntVar, C: IntVar, N: IntVar, V: IntVar](
    a_log_cumsum: jax.Array[[G, C]],
    src_scale: jax.Array[[G, C]],
    b: jax.Array[[G, C, N]],
    x: jax.Array[[G, C, V]],
    c: jax.Array[[G, C, N]] | None = None,
) -> tuple[int, int, int, int]:
    if a_log_cumsum.ndim != 2:
        raise ValueError(f"`a_log_cumsum` must be rank-2 [G, C], got {a_log_cumsum.shape}.")
    if src_scale.shape != a_log_cumsum.shape:
        raise ValueError(f"`src_scale` must match `a_log_cumsum`, got {src_scale.shape} and {a_log_cumsum.shape}.")
    if b.ndim != 3 or x.ndim != 3:
        raise ValueError(f"`b` and `x` must be rank-3 [G, C, *], got {b.shape} and {x.shape}.")

    groups, chunk_size = a_log_cumsum.shape
    b_groups, b_chunk, state_dim = b.shape
    x_groups, x_chunk, value_dim = x.shape

    if (b_groups, x_groups) != (groups, groups):
        raise ValueError("All inputs must have the same flattened group dimension.")
    if (b_chunk, x_chunk) != (chunk_size, chunk_size):
        raise ValueError("All inputs must have the same chunk size.")

    if c is not None:
        if c.ndim != 3:
            raise ValueError(f"`c` must be rank-3 [G, C, N], got {c.shape}.")
        c_groups, c_chunk, c_state_dim = c.shape
        if (c_groups, c_chunk) != (groups, chunk_size):
            raise ValueError("`c` must share the same `[G, C]` leading dimensions.")
        if c_state_dim != state_dim:
            raise ValueError(f"`b` and `c` must share the same state dim, got {state_dim} and {c_state_dim}.")

    return groups, chunk_size, state_dim, value_dim


def _causal_decay_matrix[G: IntVar, C: IntVar](a_log_cumsum: jax.Array[[G, C]]) -> jax.Array[[G, C, C]]:
    chunk_size = a_log_cumsum.shape[-1]
    diff = a_log_cumsum[:, :, None] - a_log_cumsum[:, None, :]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))[None, :, :]
    return jnp.exp(jnp.where(mask, diff, -jnp.inf))


def ssd_intra_chunk_reference_batched[G: IntVar, C: IntVar, N: IntVar, V: IntVar](
    a_log_cumsum: jax.Array[[G, C]],
    src_scale: jax.Array[[G, C]],
    b: jax.Array[[G, C, N]],
    c: jax.Array[[G, C, N]],
    x: jax.Array[[G, C, V]],
) -> jax.Array[[G, C, V]]:
    """Reference SSD intra-chunk contraction on a transformed source term."""

    _validate_ssd_batched_inputs(a_log_cumsum, src_scale, b, x, c)
    acc_dtype = jnp.float32

    cb = jnp.einsum(
        "gtn,gsn->gts",
        c.astype(acc_dtype),
        b.astype(acc_dtype),
        preferred_element_type=acc_dtype,
    )
    decay = _causal_decay_matrix(a_log_cumsum.astype(acc_dtype))
    x_scaled = x.astype(acc_dtype) * src_scale.astype(acc_dtype)[:, :, None]
    y: jax.Array[[G, C, V]] = jnp.einsum("gts,gsp->gtp", cb * decay, x_scaled, preferred_element_type=acc_dtype)
    return y.astype(x.dtype)


def ssd_chunk_state_reference_batched[G: IntVar, C: IntVar, N: IntVar, V: IntVar](
    a_log_cumsum: jax.Array[[G, C]],
    src_scale: jax.Array[[G, C]],
    b: jax.Array[[G, C, N]],
    x: jax.Array[[G, C, V]],
) -> jax.Array[[G, V, N]]:
    """Reference chunk-end accumulation for the carried SSD state."""

    _validate_ssd_batched_inputs(a_log_cumsum, src_scale, b, x)
    acc_dtype = jnp.float32
    decay_to_end = jnp.exp(a_log_cumsum.astype(acc_dtype)[:, -1:] - a_log_cumsum.astype(acc_dtype))
    chunk_state: jax.Array[[G, V, N]] = jnp.einsum(
        "gcn,gc,gcp->gpn",
        b.astype(acc_dtype),
        decay_to_end * src_scale.astype(acc_dtype),
        x.astype(acc_dtype),
        preferred_element_type=acc_dtype,
    )
    return chunk_state.astype(x.dtype)


def ssd_emit_from_prefix_reference_batched[G: IntVar, C: IntVar, N: IntVar, V: IntVar](
    a_log_cumsum: jax.Array[[G, C]],
    c: jax.Array[[G, C, N]],
    prefix_state: jax.Array[[G, V, N]],
) -> jax.Array[[G, C, V]]:
    """Emit within-chunk outputs from an incoming chunk-prefix state."""

    if a_log_cumsum.ndim != 2 or c.ndim != 3 or prefix_state.ndim != 3:
        raise ValueError("Prefix emit inputs must have shapes [G, C], [G, C, N], and [G, P, N].")
    if c.shape[:2] != a_log_cumsum.shape:
        raise ValueError("`c` must share the same `[G, C]` leading dimensions as `a_log_cumsum`.")
    if prefix_state.shape[0] != a_log_cumsum.shape[0] or prefix_state.shape[-1] != c.shape[-1]:
        raise ValueError("`prefix_state` must share the same group and state dims as `c`.")

    acc_dtype = jnp.float32
    decay = jnp.exp(a_log_cumsum.astype(acc_dtype))
    out: jax.Array[[G, C, V]] = jnp.einsum(
        "gcn,gpn,gc->gcp",
        c.astype(acc_dtype),
        prefix_state.astype(acc_dtype),
        decay,
        preferred_element_type=acc_dtype,
    )
    return out.astype(c.dtype)


def ssd_scan_chunk_states_reference_batched[G: IntVar, K: IntVar, V: IntVar, N: IntVar](
    chunk_decay: jax.Array[[G, K]],
    chunk_state: jax.Array[[G, K, V, N]],
) -> tuple[jax.Array[[G, K, V, N]], jax.Array[[G, V, N]]]:
    """Scan chunk summaries into incoming prefix states for each chunk."""

    if chunk_decay.ndim != 2:
        raise ValueError(f"`chunk_decay` must be rank-2 [G, K], got {chunk_decay.shape}.")
    if chunk_state.ndim != 4:
        raise ValueError(f"`chunk_state` must be rank-4 [G, K, P, N], got {chunk_state.shape}.")
    groups, num_chunks = chunk_decay.shape
    if chunk_state.shape[:2] != (groups, num_chunks):
        raise ValueError("`chunk_decay` and `chunk_state` must share the same `[G, K]` leading dimensions.")

    acc_dtype = jnp.float32
    carry_init: jax.Array[[G, V, N]] = jnp.zeros((groups, chunk_state.shape[2], chunk_state.shape[3]), dtype=acc_dtype)
    decay_tm: jax.Array[[K, G]] = jnp.swapaxes(chunk_decay.astype(acc_dtype), 0, 1)
    chunk_state_tm: jax.Array[[K, G, V, N]] = jnp.swapaxes(chunk_state.astype(acc_dtype), 0, 1)

    def step(
        carry: jax.Array[[G, V, N]],
        inputs: tuple[jax.Array[[G]], jax.Array[[G, V, N]]],
    ) -> tuple[jax.Array[[G, V, N]], jax.Array[[G, V, N]]]:
        decay_i, chunk_state_i = inputs
        next_carry = carry * decay_i[:, None, None] + chunk_state_i
        return next_carry, carry

    final_state, incoming_tm = jax.lax.scan(step, carry_init, (decay_tm, chunk_state_tm))
    return jnp.swapaxes(incoming_tm, 0, 1).astype(chunk_state.dtype), final_state.astype(chunk_state.dtype)


def ssd_chunked_from_local_blocks_reference_batched[G: IntVar, K: IntVar, C: IntVar, N: IntVar, V: IntVar](
    a_log_cumsum: jax.Array[[G, K, C]],
    c: jax.Array[[G, K, C, N]],
    local_output: jax.Array[[G, K, C, V]],
    chunk_state: jax.Array[[G, K, V, N]],
) -> tuple[jax.Array[[G, K, C, V]], jax.Array[[G, V, N]]]:
    """Combine local chunk outputs with scanned cross-chunk prefix states."""

    if a_log_cumsum.ndim != 3 or c.ndim != 4 or local_output.ndim != 4 or chunk_state.ndim != 4:
        raise ValueError("Chunked SSD inputs must have shapes [G, K, C, *].")
    if c.shape[:3] != a_log_cumsum.shape or local_output.shape[:3] != a_log_cumsum.shape:
        raise ValueError("`a_log_cumsum`, `c`, and `local_output` must share `[G, K, C]` leading dimensions.")
    if chunk_state.shape[:2] != a_log_cumsum.shape[:2]:
        raise ValueError("`chunk_state` must share `[G, K]` leading dimensions with `a_log_cumsum`.")

    incoming_state, final_state = ssd_scan_chunk_states_reference_batched(
        jnp.exp(a_log_cumsum[..., -1]),
        chunk_state,
    )
    prefix_output: jax.Array[[G, K, C, V]] = jax.vmap(
        ssd_emit_from_prefix_reference_batched,
        in_axes=(1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, c, incoming_state)
    return (local_output + prefix_output).astype(local_output.dtype), final_state


def ssd_chunked_forward_reference_batched[G: IntVar, K: IntVar, C: IntVar, N: IntVar, V: IntVar](
    a_log_cumsum: jax.Array[[G, K, C]],
    src_scale: jax.Array[[G, K, C]],
    b: jax.Array[[G, K, C, N]],
    c: jax.Array[[G, K, C, N]],
    x: jax.Array[[G, K, C, V]],
) -> tuple[jax.Array[[G, K, C, V]], jax.Array[[G, V, N]]]:
    """Reference chunked SSD forward pass: local block + chunk scan + prefix emit."""

    local_output: jax.Array[[G, K, C, V]] = jax.vmap(
        ssd_intra_chunk_reference_batched,
        in_axes=(1, 1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, b, c, x)
    chunk_state: jax.Array[[G, K, V, N]] = jax.vmap(
        ssd_chunk_state_reference_batched,
        in_axes=(1, 1, 1, 1),
        out_axes=1,
    )(a_log_cumsum, src_scale, b, x)
    return ssd_chunked_from_local_blocks_reference_batched(a_log_cumsum, c, local_output, chunk_state)


def ssd_chunked_sequential_reference_batched[G: IntVar, K: IntVar, C: IntVar, N: IntVar, V: IntVar](
    a_log_cumsum: jax.Array[[G, K, C]],
    src_scale: jax.Array[[G, K, C]],
    b: jax.Array[[G, K, C, N]],
    c: jax.Array[[G, K, C, N]],
    x: jax.Array[[G, K, C, V]],
) -> tuple[jax.Array[[G, K, C, V]], jax.Array[[G, V, N]]]:
    """Direct transformed-state recurrence oracle used to validate the SSD shim."""

    if a_log_cumsum.ndim != 3 or src_scale.shape != a_log_cumsum.shape:
        raise ValueError("`a_log_cumsum` and `src_scale` must have shape [G, K, C].")
    if b.ndim != 4 or c.ndim != 4 or x.ndim != 4:
        raise ValueError("`b`, `c`, and `x` must have shape [G, K, C, *].")

    acc_dtype = jnp.float32
    groups, num_chunks, chunk_size = a_log_cumsum.shape
    value_dim = x.shape[-1]
    state_dim = b.shape[-1]
    local_log_alpha_chunked = jnp.concatenate(
        [a_log_cumsum[..., :1], a_log_cumsum[..., 1:] - a_log_cumsum[..., :-1]],
        axis=-1,
    ).astype(acc_dtype)

    # `num_chunks * chunk_size` has no type parameter of its own, so these flattened views stay bare.
    alpha_tm = jnp.swapaxes(jnp.exp(local_log_alpha_chunked).reshape(groups, num_chunks * chunk_size), 0, 1)
    src_scale_tm = jnp.swapaxes(src_scale.astype(acc_dtype).reshape(groups, num_chunks * chunk_size), 0, 1)
    b_tm = jnp.swapaxes(b.astype(acc_dtype).reshape(groups, num_chunks * chunk_size, state_dim), 0, 1)
    c_tm = jnp.swapaxes(c.astype(acc_dtype).reshape(groups, num_chunks * chunk_size, state_dim), 0, 1)
    x_tm = jnp.swapaxes(x.astype(acc_dtype).reshape(groups, num_chunks * chunk_size, value_dim), 0, 1)

    def step(
        state: jax.Array[[G, V, N]],
        inputs: tuple[
            jax.Array[[G]],
            jax.Array[[G]],
            jax.Array[[G, N]],
            jax.Array[[G, N]],
            jax.Array[[G, V]],
        ],
    ) -> tuple[jax.Array[[G, V, N]], jax.Array[[G, V]]]:
        alpha_t, src_scale_t, b_t, c_t, x_t = inputs
        state = alpha_t[:, None, None] * state + x_t[:, :, None] * (src_scale_t[:, None, None] * b_t[:, None, :])
        y_t = jnp.sum(c_t[:, None, :] * state, axis=-1)
        return state, y_t

    init_state: jax.Array[[G, V, N]] = jnp.zeros((groups, value_dim, state_dim), dtype=acc_dtype)
    final_state, y_tm = jax.lax.scan(step, init_state, (alpha_tm, src_scale_tm, b_tm, c_tm, x_tm))
    y = jnp.swapaxes(y_tm, 0, 1).reshape(groups, num_chunks, chunk_size, value_dim)
    return y.astype(x.dtype), final_state.astype(x.dtype)
