# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Reverse-mode algebra for the fused RMS-GatedNorm boundary."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RmsGatedNormResiduals(NamedTuple):
    """Stock BF16 forward values retained by the fused reverse."""

    x: jax.Array
    norm_weight: jax.Array
    w_down: jax.Array
    w_up: jax.Array
    inverse_rms: jax.Array
    normalized: jax.Array
    gate_preactivation: jax.Array
    gate_hidden: jax.Array
    gate: jax.Array


class GatedNormUpCotangents(NamedTuple):
    """Cotangents emitted by the output-gate reverse."""

    direct: jax.Array
    gate_accumulator: jax.Array
    w_up: jax.Array


def exact_gated_norm_up_reverse(
    output_cotangent: jax.Array,
    residuals: RmsGatedNormResiduals,
) -> GatedNormUpCotangents:
    """Reverse the stock BF16 output gate without changing rounding sites."""
    output_cotangent = output_cotangent.reshape(residuals.normalized.shape)
    direct_cotangent = output_cotangent * residuals.gate
    gate_cotangent = output_cotangent * residuals.normalized
    gate_accumulator_cotangent = gate_cotangent * (residuals.gate * (1 - residuals.gate))
    w_up_cotangent = jnp.einsum("tr,td->rd", residuals.gate_hidden, gate_accumulator_cotangent)
    return GatedNormUpCotangents(direct_cotangent, gate_accumulator_cotangent, w_up_cotangent)


def exact_silu_backward_reference(
    output_cotangent: jax.Array,
    w_up: jax.Array,
    preactivation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reference for the fused SiLU-input cotangent GEMM.

    Mirrors ``quack_silu_backward_gemm``: returns the cotangent of the SiLU input and the
    recomputed SiLU output.
    """
    postactivation, silu_pullback = jax.vjp(jax.nn.silu, preactivation)
    preactivation_cotangent = silu_pullback(jnp.einsum("td,rd->tr", output_cotangent, w_up))[0]
    return preactivation_cotangent, postactivation


def exact_rms_backward_producer_reference(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    direct_cotangent: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reference for the BF16 producer GEMM and its RMS row partials.

    Matches ``quack_coda_rms_backward_producer``, including its output contract: the row dot
    arrives as ``[M, tiles]`` partials for the caller to reduce. The kernel emits one column per
    N tile of the GEMM; this reference reduces in one pass and so emits a single column.
    """
    gate_cotangent = jnp.einsum("tr,dr->td", gate_preactivation_cotangent, w_down)
    unweighted_cotangent = gate_cotangent + direct_cotangent
    normalized_x = x.astype(jnp.float32) * inverse_rms[:, None]
    weighted_cotangent = unweighted_cotangent.astype(jnp.float32) * norm_weight
    row_dot_partial = jnp.sum(weighted_cotangent * normalized_x, axis=-1, keepdims=True)
    return unweighted_cotangent, row_dot_partial


def exact_rms_backward_consumer(
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
