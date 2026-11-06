# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-serial GDN-2 oracle, independent of the chunkwise kernel algebra."""

import jax
import jax.numpy as jnp


def gdn2_reference(q, k, v, w, b, g, scale, h0):
    """Apply channel-wise decay, erase, and write in FP32 at every token.

    Inputs have shape (batch, length, heads, channels); the state has shape
    (batch, heads, key_channels, value_channels). Returns FP32 outputs and
    final state without clipping, damping, or non-finite replacement.
    """
    tokens = tuple(jnp.moveaxis(x.astype(jnp.float32), 1, 0) for x in (q, k, v, w, b, g))

    def step(state, inputs):
        query, key, value, write, erase, log_decay = inputs
        decayed = state * jnp.exp(log_decay)[..., :, None]
        prediction = jnp.sum(decayed * (erase * key)[..., :, None], axis=-2)
        residual = write * value - prediction
        updated = decayed + key[..., :, None] * residual[..., None, :]
        output = jnp.sum(updated * (query * scale)[..., :, None], axis=-2)
        return updated, output

    final_state, output = jax.lax.scan(step, h0.astype(jnp.float32), tokens)
    return jnp.moveaxis(output, 0, 1), final_state
