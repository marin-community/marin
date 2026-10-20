# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference ``GatedNorm(RMSNorm(x))``: the Grug MoE hero's two norms, composed."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def gated_rms_norm_reference(
    x: Float[Array, "T D"],
    rms_weight: Float[Array, " D"],
    w_down: Float[Array, "D R"],
    w_up: Float[Array, "R D"],
    eps: float,
) -> Float[Array, "T D"]:
    """``n * sigmoid(silu(n @ w_down) @ w_up)`` with ``n = RMSNorm(x)``.

    Matches ``RMSNorm.__call__`` followed by ``GatedNorm.__call__`` in
    ``experiments/grug/moe_hero_ep/model.py``: the RMS statistics are fp32 and ``n`` is rounded to
    ``x``'s dtype before the bottleneck.
    """
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normed = ((x32 * jax.lax.rsqrt(variance + eps)) * rms_weight).astype(dtype)
    gate_hidden = jax.nn.silu(jnp.einsum("...d,dr->...r", normed, w_down))
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, w_up))
    return normed * gate.astype(dtype)
