# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-native default, linear, Llama3 and YaRN rotary embeddings."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from levanter.layers.rotary import RotaryEmbeddingsConfig


def apply_rotary(
    x: Float[Array, "B S H D"], position_ids: Int[Array, "B S"], config: RotaryEmbeddingsConfig
) -> Float[Array, "B S H D"]:
    """Apply split-half RoPE using caller-supplied token positions."""
    dim = x.shape[-1]
    with jax.ensure_compile_time_eval():
        inv_freq = config.inverse_frequencies(dim)
    frequencies = position_ids[..., None] * inv_freq
    angles = jnp.concatenate((frequencies, frequencies), axis=-1)[..., None, :]
    cos, sin = config.scale_coefficients(jnp.cos(angles).astype(x.dtype), jnp.sin(angles).astype(x.dtype))
    first, second = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    return (x * cos + rotated * sin).astype(x.dtype)
