# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""YaRN frequencies for Hero's partial, interleaved rotary embeddings."""

import math

import jax.numpy as jnp
import numpy as np
from levanter.layers.rotary import YarnRotaryEmbeddingsConfig


def yarn_frequencies(rotary_dim: int, config: YarnRotaryEmbeddingsConfig) -> tuple[jnp.ndarray, float]:
    """Return corrected inverse frequencies and the Q/K rotary amplitude.

    Correction dimensions refer to the rotated subspace, which is half of a
    Hero attention head. The remaining coordinates and rope-free global layers
    keep their original amplitudes. This follows the YaRN beta-ramp convention
    used by Transformers, including the 0.001 denominator at a coincident ramp.
    """
    if rotary_dim < 2 or rotary_dim % 2:
        raise ValueError("YaRN requires an even, positive rotary dimension")
    if config.factor < 1 or config.theta <= 1 or config.original_max_position_embeddings <= 0:
        raise ValueError("Invalid YaRN context, factor or frequency base")
    if not config.beta_fast > config.beta_slow > 0:
        raise ValueError("YaRN beta_fast must exceed beta_slow > 0")
    if not all(math.isfinite(value) for value in (config.factor, config.theta, config.mscale)):
        raise ValueError("YaRN parameters must be finite")

    def correction(rotations: float) -> float:
        return (
            rotary_dim
            * math.log(config.original_max_position_embeddings / (rotations * 2 * math.pi))
            / (2 * math.log(config.theta))
        )

    low = max(math.floor(correction(config.beta_fast)), 0)
    # YaRN's reference and Transformers clamp the boundary to rotary_dim - 1,
    # even though the ramp contains rotary_dim // 2 frequencies. A boundary
    # beyond that vector intentionally leaves its last entry partly interpolated.
    high = min(math.ceil(correction(config.beta_slow)), rotary_dim - 1)
    denominator = high - low if high != low else 0.001
    ramp = np.clip((np.arange(rotary_dim // 2, dtype=np.float32) - low) / denominator, 0, 1)
    unscaled = 1 / config.theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    frequencies = unscaled * (1 - ramp) + (unscaled / config.factor) * ramp
    amplitude = 1.0 if config.factor == 1 else 1 + 0.1 * config.mscale * math.log(config.factor)
    if amplitude <= 0:
        raise ValueError("YaRN rotary amplitude must be positive")
    if config.factor == 1:
        # Preserve Hero's existing float32 power evaluation exactly at identity.
        half = rotary_dim // 2
        return 1 / (config.theta ** (jnp.arange(half, dtype=jnp.float32) / half)), 1.0
    return jnp.asarray(frequencies), amplitude
