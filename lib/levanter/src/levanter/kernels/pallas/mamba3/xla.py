# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..ssd.reference import (
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    ssd_scan_chunk_states_reference_batched,
)
from ..ssd.xla import (
    LOCAL_OUTPUT_AUTO,
    LocalOutputVariant,
    PREFIX_EMIT_AUTO,
    PrefixEmitVariant,
    ssd_chunk_state_xla_batched,
    ssd_chunk_state_xla_chunked_batched,
    ssd_chunked_from_local_blocks_xla_batched,
    ssd_intra_chunk_xla_batched,
    ssd_intra_chunk_xla_chunked_batched,
)
from .reference import (
    mamba3_mimo_apply_gate_and_collapse_chunked,
    mamba3_mimo_rank_expand_chunked,
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

    with jax.named_scope("mamba3_intra_chunk"):
        acc_dtype = jnp.float32
        with jax.named_scope("ssd_core"):
            y = ssd_intra_chunk_xla_batched(a_log_cumsum, src_scale, b, c, x).astype(acc_dtype)
        with jax.named_scope("diagonal_correction"):
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

    with jax.named_scope("mamba3_chunk_state"):
        return ssd_chunk_state_xla_batched(a_log_cumsum, src_scale, b, x)


def mamba3_chunked_forward_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
    *,
    prefix_emit_variant: PrefixEmitVariant = PREFIX_EMIT_AUTO,
    local_output_variant: LocalOutputVariant = LOCAL_OUTPUT_AUTO,
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Chunked Mamba-3 forward pass in plain JAX/XLA."""

    if prefix_emit_variant == PREFIX_EMIT_AUTO and local_output_variant == LOCAL_OUTPUT_AUTO:
        return _mamba3_chunked_forward_xla_batched_default_custom_vjp(a_log_cumsum, src_scale, out_correction, b, c, x)
    return _mamba3_chunked_forward_xla_batched_impl(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        prefix_emit_variant=prefix_emit_variant,
        local_output_variant=local_output_variant,
    )


def _mamba3_chunked_forward_xla_batched_impl(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
    *,
    prefix_emit_variant: PrefixEmitVariant = PREFIX_EMIT_AUTO,
    local_output_variant: LocalOutputVariant = LOCAL_OUTPUT_AUTO,
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """Chunked Mamba-3 forward pass in plain JAX/XLA."""

    with jax.named_scope("mamba3_chunked_forward"):
        with jax.named_scope("local_output"):
            acc_dtype = jnp.float32
            y = ssd_intra_chunk_xla_chunked_batched(
                a_log_cumsum,
                src_scale,
                b,
                c,
                x,
                local_output_variant=local_output_variant,
            ).astype(acc_dtype)
            diag_cb = jnp.sum(c.astype(acc_dtype) * b.astype(acc_dtype), axis=-1)
            correction = (out_correction.astype(acc_dtype) * diag_cb)[..., None] * x.astype(acc_dtype)
            local_output = (y - correction).astype(x.dtype)
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


def _materialize_cotangent(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
    primal: jax.Array,
) -> jax.Array:
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(primal)
    return cotangent


@jax.custom_vjp
def _mamba3_chunked_forward_xla_batched_default_custom_vjp(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    return _mamba3_chunked_forward_xla_batched_impl(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        prefix_emit_variant=PREFIX_EMIT_AUTO,
        local_output_variant=LOCAL_OUTPUT_AUTO,
    )


def _mamba3_chunked_forward_xla_batched_default_custom_vjp_fwd(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[
    tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    outputs = _mamba3_chunked_forward_xla_batched_impl(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        prefix_emit_variant=PREFIX_EMIT_AUTO,
        local_output_variant=LOCAL_OUTPUT_AUTO,
    )
    residuals = (a_log_cumsum, src_scale, out_correction, b, c, x)
    return outputs, residuals


def _mamba3_chunked_forward_xla_batched_default_custom_vjp_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangents: tuple[
        jax.Array | jax.custom_derivatives.SymbolicZero,
        jax.Array | jax.custom_derivatives.SymbolicZero,
    ],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    a_log_cumsum, src_scale, out_correction, b, c, x = residuals
    y_bar, final_state_bar = cotangents
    primals = (a_log_cumsum, src_scale, out_correction, b, c, x)

    def forward_impl(
        a_log_cumsum_in: jax.Array,
        src_scale_in: jax.Array,
        out_correction_in: jax.Array,
        b_in: jax.Array,
        c_in: jax.Array,
        x_in: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _mamba3_chunked_forward_xla_batched_impl(
            a_log_cumsum_in,
            src_scale_in,
            out_correction_in,
            b_in,
            c_in,
            x_in,
            prefix_emit_variant=PREFIX_EMIT_AUTO,
            local_output_variant=LOCAL_OUTPUT_AUTO,
        )

    # This VJP is intentionally recompute-heavy: we re-run the forward under
    # autodiff rather than storing large intermediates from the chunked kernel.
    # primals_out is only used to materialize SymbolicZero cotangents with the
    # correct shapes/dtypes before feeding them into the pullback.
    primals_out, pullback = jax.vjp(forward_impl, *primals)
    y_ct = _materialize_cotangent(y_bar, primals_out[0])
    final_state_ct = _materialize_cotangent(final_state_bar, primals_out[1])
    return pullback((y_ct, final_state_ct))


_mamba3_chunked_forward_xla_batched_default_custom_vjp.defvjp(
    _mamba3_chunked_forward_xla_batched_default_custom_vjp_fwd,
    _mamba3_chunked_forward_xla_batched_default_custom_vjp_bwd,
)


def mamba3_chunked_forward_native_xla_batched(
    dt: Float[Array, "groups chunks chunk"],
    lam: Float[Array, "groups chunks chunk"],
    a: Float[Array, "groups chunks chunk"] | Float[Array, "groups chunks"],
    b: Float[Array, "groups chunks chunk state"],
    c: Float[Array, "groups chunks chunk state"],
    x: Float[Array, "groups chunks chunk value"],
) -> tuple[Float[Array, "groups chunks chunk value"], Float[Array, "groups value state"]]:
    """XLA fast path on native Mamba-3 parameters."""

    with jax.named_scope("mamba3_native_xla"):
        with jax.named_scope("prepare_scales"):
            src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
            a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
        return mamba3_chunked_forward_xla_batched(a_log_cumsum, src_scale, out_correction, b, c, x)


def mamba3_mimo_chunk_state_xla_chunked_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state rank"],
    x: Float[Array, "groups chunks rank chunk value"],
) -> Float[Array, "groups chunks value state"]:
    """Chunk-state accumulation for rank-expanded MIMO inputs with the same `[P, N]` carry as SISO."""

    acc_dtype = jnp.float32
    with jax.named_scope("prepare_decay_to_end"):
        decay_to_end = jnp.exp(a_log_cumsum.astype(acc_dtype)[..., -1:] - a_log_cumsum.astype(acc_dtype))
    with jax.named_scope("prepare_scaled_src"):
        scaled_src = decay_to_end * src_scale.astype(acc_dtype)
    with jax.named_scope("emit_chunk_state"):
        return jnp.einsum(
            "gkcnu,gkucp,gkc->gkpn",
            b.astype(acc_dtype),
            x.astype(acc_dtype),
            scaled_src,
            preferred_element_type=acc_dtype,
        ).astype(x.dtype)


def mamba3_mimo_chunked_forward_ranked_xla_batched(
    a_log_cumsum: Float[Array, "groups chunks chunk"],
    src_scale: Float[Array, "groups chunks chunk"],
    out_correction: Float[Array, "groups chunks chunk"],
    b: Float[Array, "groups chunks chunk state rank"],
    c: Float[Array, "groups chunks chunk state rank"],
    x: Float[Array, "groups chunks rank chunk value"],
) -> tuple[Float[Array, "groups chunks rank chunk value"], Float[Array, "groups value state"]]:
    """Chunked MIMO forward pass on rank-expanded tensors before gating/collapse."""

    acc_dtype = jnp.float32
    with jax.named_scope("mamba3_mimo_chunked_forward"):
        with jax.named_scope("local_output"):
            with jax.named_scope("bc_contraction"):
                bc = jnp.einsum(
                    "gktnv,gksnu->gktsuv",
                    c.astype(acc_dtype),
                    b.astype(acc_dtype),
                    preferred_element_type=acc_dtype,
                )
            with jax.named_scope("causal_decay"):
                chunk_size = a_log_cumsum.shape[-1]
                mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))[None, None, :, :]
                diff = a_log_cumsum.astype(acc_dtype)[..., :, None] - a_log_cumsum.astype(acc_dtype)[..., None, :]
                decay = jnp.exp(jnp.where(mask, diff, -jnp.inf))
            with jax.named_scope("scale_x"):
                x_scaled = x.astype(acc_dtype) * src_scale.astype(acc_dtype)[:, :, None, :, None]
            with jax.named_scope("emit_local_output"):
                local_output = jnp.einsum(
                    "gkts,gktsuv,gkusp->gkvtp",
                    decay,
                    bc,
                    x_scaled,
                    preferred_element_type=acc_dtype,
                )
            with jax.named_scope("diagonal_correction"):
                bc_diag = jnp.einsum(
                    "gktnu,gktnv->gktuv",
                    b.astype(acc_dtype),
                    c.astype(acc_dtype),
                    preferred_element_type=acc_dtype,
                )
                correction = out_correction.astype(acc_dtype)[:, :, None, :, None] * jnp.einsum(
                    "gkucp,gkcuv->gkvcp",
                    x.astype(acc_dtype),
                    bc_diag,
                    preferred_element_type=acc_dtype,
                )
                local_output = (local_output - correction).astype(x.dtype)
        with jax.named_scope("chunk_state"):
            chunk_state = mamba3_mimo_chunk_state_xla_chunked_batched(a_log_cumsum, src_scale, b, x)
        with jax.named_scope("prefix_emit"):
            with jax.named_scope("prepare_chunk_decay"):
                chunk_decay = jnp.exp(a_log_cumsum[..., -1])
            with jax.named_scope("incoming_state_scan"):
                incoming_state, final_state = ssd_scan_chunk_states_reference_batched(chunk_decay, chunk_state)
            with jax.named_scope("prepare_prefix_decay"):
                prefix_decay = jnp.exp(a_log_cumsum.astype(acc_dtype))
            with jax.named_scope("emit_prefix"):
                prefix_output = jnp.einsum(
                    "gkpn,gkcnv,gkc->gkvcp",
                    incoming_state.astype(acc_dtype),
                    c.astype(acc_dtype),
                    prefix_decay,
                    preferred_element_type=acc_dtype,
                )
        return (local_output.astype(acc_dtype) + prefix_output).astype(x.dtype), final_state


def mamba3_mimo_chunked_forward_xla_batched(
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
    """Chunked MIMO forward pass in plain JAX/XLA with lightweight rank expand/gate/collapse."""

    with jax.named_scope("mamba3_mimo_output_xla"):
        with jax.named_scope("rank_expand"):
            x_ranked = mamba3_mimo_rank_expand_chunked(x_base, w_x)
            z_ranked = mamba3_mimo_rank_expand_chunked(z_base, w_z)
        y_ranked, final_state = mamba3_mimo_chunked_forward_ranked_xla_batched(
            a_log_cumsum,
            src_scale,
            out_correction,
            b,
            c,
            x_ranked,
        )
        with jax.named_scope("gate_and_collapse"):
            y = mamba3_mimo_apply_gate_and_collapse_chunked(y_ranked, z_ranked, w_o)
        return y, final_state
