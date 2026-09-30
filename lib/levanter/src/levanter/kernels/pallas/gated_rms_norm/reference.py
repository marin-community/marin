# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Readable vanilla-JAX oracle for RMSNorm followed by a low-rank sigmoid gate (GatedNorm).

This mirrors ``RMSNorm.__call__`` followed by ``GatedNorm.__call__`` in
``experiments/grug/moe_hero_ep/model.py``, op for op, so its rounding points (the normalized
activation, the rank projection and the gate all rounded to the activation dtype) are the
ones the fused kernel is compared against.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def gated_rms_norm_reference(
    x: Float[Array, "T D"],
    norm_weight: Float[Array, " D"],
    w_down: Float[Array, "D R"],
    w_up: Float[Array, "R D"],
    *,
    eps: float,
) -> Float[Array, "T D"]:
    """``y * sigmoid(silu(y @ w_down) @ w_up)`` with ``y = rms_norm(x) * norm_weight``."""
    dtype = x.dtype
    xf = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    y = (xf * jax.lax.rsqrt(variance + eps) * norm_weight).astype(dtype)
    gate_hidden = jax.nn.silu(jnp.einsum("...d,dr->...r", y, w_down))
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, w_up))
    return y * gate.astype(dtype)
