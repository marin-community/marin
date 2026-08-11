# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Reverse-mode algebra shared by CODA-style RMS-GatedNorm implementations."""

import jax
import jax.numpy as jnp


def coda_rms_backward_producer_reference(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    direct_cotangent: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reference for the CODA RMS-backward producer epilogue."""
    gate_cotangent_accumulator = jnp.einsum(
        "tr,dr->td",
        gate_preactivation_cotangent,
        w_down,
        preferred_element_type=jnp.float32,
    )
    unweighted_cotangent = gate_cotangent_accumulator + direct_cotangent.astype(jnp.float32)
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    weighted_cotangent_accumulator = unweighted_cotangent * norm_weight
    row_dot = jnp.sum(weighted_cotangent_accumulator * normalized_x, axis=-1)
    norm_weight_cotangent = jnp.sum(unweighted_cotangent * normalized_x, axis=0)
    return weighted_cotangent_accumulator.astype(x.dtype), row_dot, norm_weight_cotangent


def exact_rms_backward_producer_reference(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    direct_cotangent: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reference for the BF16-exact CODA producer used by the training path."""
    gate_cotangent = jnp.einsum("tr,dr->td", gate_preactivation_cotangent, w_down)
    unweighted_cotangent = gate_cotangent + direct_cotangent
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    weighted_cotangent = unweighted_cotangent.astype(jnp.float32) * norm_weight
    row_dot = jnp.sum(weighted_cotangent * normalized_x, axis=-1)
    norm_weight_cotangent = jnp.sum(unweighted_cotangent.astype(jnp.float32) * normalized_x, axis=0)
    return weighted_cotangent, row_dot, norm_weight_cotangent


def coda_rms_backward_consumer_reference(
    weighted_cotangent: jax.Array,
    row_dot: jax.Array,
    x: jax.Array,
    inverse_rms: jax.Array,
) -> jax.Array:
    """Apply the reduced RMS row scalar while emitting the input cotangent."""
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    row_mean = row_dot / x.shape[-1]
    return ((weighted_cotangent.astype(jnp.float32) - normalized_x * row_mean[:, None]) * inverse_rms[:, None]).astype(
        x.dtype
    )


def exact_rms_backward_consumer_reference(
    unweighted_cotangent: jax.Array,
    row_dot: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> jax.Array:
    """Apply the norm gain and reduced row scalar while emitting ``dx``."""
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    weighted_cotangent = unweighted_cotangent.astype(jnp.float32) * norm_weight
    row_mean = row_dot / x.shape[-1]
    return ((weighted_cotangent - normalized_x * row_mean[:, None]) * inverse_rms[:, None]).astype(x.dtype)


def exact_gated_norm_up_reverse(
    output_cotangent: jax.Array,
    residuals: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reverse the stock BF16 output gate without changing rounding sites."""
    _, _, _, _, _, normalized, _, _, gate_hidden, gate = residuals
    output_cotangent = output_cotangent.reshape(normalized.shape)
    direct_cotangent = output_cotangent * gate
    gate_cotangent = output_cotangent * normalized
    gate_accumulator_cotangent = gate_cotangent * (gate * (1 - gate))
    w_up_cotangent = jnp.einsum("tr,td->rd", gate_hidden, gate_accumulator_cotangent)
    return direct_cotangent, gate_accumulator_cotangent, w_up_cotangent


def exact_gated_norm_reverse_inputs(
    output_cotangent: jax.Array,
    residuals: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reverse the stock BF16 GatedNorm arithmetic without changing rounding sites."""
    _, _, _, w_up, _, normalized, gate_preactivation, silu_sigmoid, _, _ = residuals
    direct_cotangent, gate_accumulator_cotangent, w_up_cotangent = exact_gated_norm_up_reverse(
        output_cotangent, residuals
    )
    gate_hidden_cotangent = jnp.einsum("td,rd->tr", gate_accumulator_cotangent, w_up)
    silu_sigmoid_derivative = silu_sigmoid * (1 - silu_sigmoid)
    gate_preactivation_cotangent = (
        gate_hidden_cotangent * silu_sigmoid + (gate_preactivation * gate_hidden_cotangent) * silu_sigmoid_derivative
    )
    w_down_cotangent = jnp.einsum("td,tr->dr", normalized, gate_preactivation_cotangent)
    return direct_cotangent, gate_preactivation_cotangent, w_down_cotangent, w_up_cotangent


def coda_gated_norm_up_reverse(
    output_cotangent: jax.Array,
    residuals: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reverse the output gate through sigmoid, returning the SiLU-output cotangent."""
    x, norm_weight, _, w_up, inverse_rms, _, gate_hidden, gate = residuals
    x_flat = x.reshape((-1, x.shape[-1]))
    output_cotangent = output_cotangent.reshape(x_flat.shape)
    normalized = (x_flat.astype(jnp.float32) * norm_weight * inverse_rms[:, None]).astype(x.dtype)
    direct_cotangent = (output_cotangent * gate).astype(x.dtype)
    gate_cotangent = output_cotangent * normalized
    gate_accumulator_cotangent = (
        gate_cotangent.astype(jnp.float32) * gate.astype(jnp.float32) * (1.0 - gate.astype(jnp.float32))
    ).astype(x.dtype)
    w_up_cotangent = jnp.einsum(
        "tr,td->rd",
        gate_hidden,
        gate_accumulator_cotangent,
        preferred_element_type=jnp.float32,
    ).astype(w_up.dtype)
    return normalized, direct_cotangent, gate_accumulator_cotangent, w_up_cotangent


def coda_gated_norm_reverse_inputs(
    output_cotangent: jax.Array,
    residuals: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the direct RMS cotangent, down-projection cotangent, and weight gradients."""
    x, _, w_down, w_up, _, gate_preactivation, _, _ = residuals
    normalized, direct_cotangent, gate_accumulator_cotangent, w_up_cotangent = coda_gated_norm_up_reverse(
        output_cotangent, residuals
    )
    gate_hidden_cotangent = jnp.einsum(
        "td,rd->tr",
        gate_accumulator_cotangent,
        w_up,
        preferred_element_type=jnp.float32,
    )
    gate_sigmoid = jax.nn.sigmoid(gate_preactivation)
    silu_derivative = gate_sigmoid * (1.0 + gate_preactivation * (1.0 - gate_sigmoid))
    gate_preactivation_cotangent = (gate_hidden_cotangent * silu_derivative).astype(x.dtype)
    w_down_cotangent = jnp.einsum(
        "td,tr->dr",
        normalized,
        gate_preactivation_cotangent,
        preferred_element_type=jnp.float32,
    ).astype(w_down.dtype)
    return direct_cotangent, gate_preactivation_cotangent, w_down_cotangent, w_up_cotangent


def coda_rms_gated_norm_analytic_backward(
    output_cotangent: jax.Array,
    residuals: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Differentiate delayed-RMS GatedNorm without replaying its forward GEMMs."""
    x, norm_weight, w_down, _, inverse_rms, _, _, _ = residuals
    x_flat = x.reshape((-1, x.shape[-1]))
    direct_cotangent, gate_preactivation_cotangent, w_down_cotangent, w_up_cotangent = coda_gated_norm_reverse_inputs(
        output_cotangent, residuals
    )
    weighted_cotangent, row_dot, norm_weight_cotangent = coda_rms_backward_producer_reference(
        gate_preactivation_cotangent,
        w_down,
        direct_cotangent,
        x_flat,
        norm_weight,
        inverse_rms,
    )
    x_cotangent = coda_rms_backward_consumer_reference(weighted_cotangent, row_dot, x_flat, inverse_rms).reshape(
        x.shape
    )
    return x_cotangent, norm_weight_cotangent.astype(norm_weight.dtype), w_down_cotangent, w_up_cotangent
