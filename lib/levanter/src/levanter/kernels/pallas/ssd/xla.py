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

PREFIX_EMIT_EINSUM3 = "einsum3"
PREFIX_EMIT_SCAN_FUSED = "scan_fused"
PREFIX_EMIT_AUTO = "auto"
LOCAL_OUTPUT_EINSUM = "einsum"
LOCAL_OUTPUT_MATMUL = "matmul"
LOCAL_OUTPUT_TRIL_SCALED_EINSUM = "tril_scaled_einsum"
LOCAL_OUTPUT_TRIL_SCALED_MATMUL = "tril_scaled_matmul"
LOCAL_OUTPUT_AUTO = "auto"

PrefixEmitVariant = str
LocalOutputVariant = str


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
    *,
    local_output_variant: LocalOutputVariant = LOCAL_OUTPUT_AUTO,
) -> Float[Array, "groups chunks chunk value"]:
    acc_dtype = jnp.float32
    c_f32 = c.astype(acc_dtype)
    b_f32 = b.astype(acc_dtype)
    x_f32 = x.astype(acc_dtype)
    src_scale_f32 = src_scale.astype(acc_dtype)
    if local_output_variant == LOCAL_OUTPUT_AUTO:
        local_output_variant = (
            LOCAL_OUTPUT_TRIL_SCALED_MATMUL if b.shape[-1] >= 2 * x.shape[-1] else LOCAL_OUTPUT_EINSUM
        )
    if local_output_variant == LOCAL_OUTPUT_EINSUM:
        cb = jnp.einsum(
            "gktn,gksn->gkts",
            c_f32,
            b_f32,
            preferred_element_type=acc_dtype,
        )
        decay = _causal_decay_matrix_chunked(a_log_cumsum.astype(acc_dtype))
        x_scaled = x_f32 * src_scale_f32[..., None]
        return jnp.einsum(
            "gkts,gksp->gktp",
            cb * decay,
            x_scaled,
            preferred_element_type=acc_dtype,
        ).astype(x.dtype)
    if local_output_variant == LOCAL_OUTPUT_MATMUL:
        with jax.named_scope("cb_matmul"):
            cb = jnp.matmul(
                c_f32,
                jnp.swapaxes(b_f32, -1, -2),
                preferred_element_type=acc_dtype,
            )
        with jax.named_scope("apply_decay"):
            weighted_cb = cb * _causal_decay_matrix_chunked(a_log_cumsum.astype(acc_dtype))
        with jax.named_scope("scale_x"):
            x_scaled = x_f32 * src_scale_f32[..., None]
        with jax.named_scope("emit_matmul"):
            return jnp.matmul(weighted_cb, x_scaled, preferred_element_type=acc_dtype).astype(x.dtype)
    if local_output_variant in (LOCAL_OUTPUT_TRIL_SCALED_EINSUM, LOCAL_OUTPUT_TRIL_SCALED_MATMUL):
        chunk_size = a_log_cumsum.shape[-1]
        lower_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))[None, None, :, :]
        with jax.named_scope("cb_matmul"):
            cb = jnp.matmul(
                c_f32,
                jnp.swapaxes(b_f32, -1, -2),
                preferred_element_type=acc_dtype,
            )
        with jax.named_scope("apply_tril"):
            masked_cb = jnp.where(lower_mask, cb, 0.0)
        with jax.named_scope("scale_x"):
            x_scaled = x_f32 * src_scale_f32[..., None] * jnp.exp(-a_log_cumsum.astype(acc_dtype))[..., None]
        with jax.named_scope("emit_row_scaled"):
            if local_output_variant == LOCAL_OUTPUT_TRIL_SCALED_EINSUM:
                y = jnp.einsum(
                    "gkts,gksp->gktp",
                    masked_cb,
                    x_scaled,
                    preferred_element_type=acc_dtype,
                )
            else:
                y = jnp.matmul(masked_cb, x_scaled, preferred_element_type=acc_dtype)
            y = y * jnp.exp(a_log_cumsum.astype(acc_dtype))[..., None]
        return y.astype(x.dtype)
    raise ValueError(f"Unsupported local_output_variant: {local_output_variant}.")


def ssd_chunk_state_xla_chunked_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> Float[Array, "groups chunks value state"]:
    acc_dtype = jnp.float32
    with jax.named_scope("prepare_decay_to_end"):
        decay_to_end = jnp.exp(a_log_cumsum.astype(acc_dtype)[..., -1:] - a_log_cumsum.astype(acc_dtype))
    with jax.named_scope("prepare_scaled_src"):
        scaled_src = decay_to_end * src_scale.astype(acc_dtype)
    with jax.named_scope("emit_chunk_state"):
        return jnp.einsum(
            "gkcn,gkc,gkcp->gkpn",
            b.astype(acc_dtype),
            scaled_src,
            x.astype(acc_dtype),
            preferred_element_type=acc_dtype,
        ).astype(x.dtype)


def ssd_chunked_from_local_blocks_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    c: Float[Array, "groups chunks chunk state"],
    local_output: Float[Array, "groups chunks chunk value"],
    chunk_state: Float[Array, "groups chunks value state"],
    *,
    prefix_emit_variant: PrefixEmitVariant = PREFIX_EMIT_AUTO,
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    acc_dtype = jnp.float32
    if prefix_emit_variant == PREFIX_EMIT_AUTO:
        prefix_emit_variant = PREFIX_EMIT_SCAN_FUSED if chunk_state.shape[2] > c.shape[-1] else PREFIX_EMIT_EINSUM3

    with jax.named_scope("prepare_chunk_decay"):
        chunk_decay = jnp.exp(a_log_cumsum[..., -1])
    with jax.named_scope("prepare_prefix_decay"):
        decay = jnp.exp(a_log_cumsum.astype(acc_dtype))

    if prefix_emit_variant == PREFIX_EMIT_SCAN_FUSED:
        with jax.named_scope("scan_fused"):
            carry_init = jnp.zeros((chunk_state.shape[0], chunk_state.shape[2], chunk_state.shape[3]), dtype=acc_dtype)
            decay_tm = jnp.swapaxes(chunk_decay.astype(acc_dtype), 0, 1)
            chunk_state_tm = jnp.swapaxes(chunk_state.astype(acc_dtype), 0, 1)
            with jax.named_scope("prepare_c_scaled"):
                c_scaled_tm = jnp.swapaxes(c.astype(acc_dtype) * decay[..., None], 0, 1)

            def step(
                carry: Float[Array, "groups value state"],
                inputs: tuple[
                    Float[Array, "groups"], Float[Array, "groups value state"], Float[Array, "groups chunk state"]
                ],
            ) -> tuple[Float[Array, "groups value state"], Float[Array, "groups chunk value"]]:
                decay_i, chunk_state_i, c_scaled_i = inputs
                prefix_output_i = jnp.einsum(
                    "gcn,gpn->gcp",
                    c_scaled_i,
                    carry,
                    preferred_element_type=acc_dtype,
                )
                next_carry = carry * decay_i[:, None, None] + chunk_state_i
                return next_carry, prefix_output_i

            final_state, prefix_output_tm = jax.lax.scan(step, carry_init, (decay_tm, chunk_state_tm, c_scaled_tm))
            final_state = final_state.astype(chunk_state.dtype)
            prefix_output = jnp.swapaxes(prefix_output_tm, 0, 1).astype(c.dtype)
    else:
        with jax.named_scope("incoming_state_scan"):
            incoming_state, final_state = ssd_scan_chunk_states_reference_batched(chunk_decay, chunk_state)
        incoming_state_f32 = incoming_state.astype(acc_dtype)
        if prefix_emit_variant == PREFIX_EMIT_EINSUM3:
            with jax.named_scope("einsum3"):
                prefix_output = jnp.einsum(
                    "gkcn,gkpn,gkc->gkcp",
                    c.astype(acc_dtype),
                    incoming_state_f32,
                    decay,
                    preferred_element_type=acc_dtype,
                ).astype(c.dtype)
        else:
            raise ValueError(f"Unsupported prefix_emit_variant: {prefix_emit_variant}.")

    return (local_output + prefix_output).astype(local_output.dtype), final_state


def ssd_chunked_forward_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
    *,
    prefix_emit_variant: PrefixEmitVariant = PREFIX_EMIT_AUTO,
    local_output_variant: LocalOutputVariant = LOCAL_OUTPUT_AUTO,
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Chunked SSD forward pass in plain JAX/XLA."""

    with jax.named_scope("ssd_chunked_forward"):
        with jax.named_scope("local_output"):
            local_output = ssd_intra_chunk_xla_chunked_batched(
                a_log_cumsum,
                src_scale,
                b,
                c,
                x,
                local_output_variant=local_output_variant,
            )
        with jax.named_scope("chunk_state"):
            chunk_state = ssd_chunk_state_xla_chunked_batched(a_log_cumsum, src_scale, b, x)
        with jax.named_scope("prefix_emit"):
            return ssd_chunked_from_local_blocks_xla_batched(
                a_log_cumsum,
                c,
                local_output,
                chunk_state,
                prefix_emit_variant=prefix_emit_variant,
            )
