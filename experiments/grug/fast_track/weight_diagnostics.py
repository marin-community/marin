# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight-shape diagnostics logged during training (``weight_diagnostics_every``).

For every projection-like matrix (used as ``x @ W``, so ``W`` is [..., in, out]):

- **stable rank** ``||W||_F^2 / ||W||_2^2``: how many directions carry the weight. MuonH holds the Frobenius
  norm; once the stable rank is low that norm no longer bounds the spectral norm well.
- **output-channel norm ratio** ``max_j ||W[:, j]|| / mean_j ||W[:, j]||``: one output channel with an outsized
  weight vector produces outsized activations in that channel.

For every RMSNorm the learned gain's distance from its initial 1. Stacked and expert tensors carry leading
batch axes; each statistic is computed per matrix and reported as the mean and the worst case over them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

POWER_ITERATIONS = 30


def spectral_norm(w: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Largest singular value of each [..., a, b] matrix by power iteration."""
    v = jax.random.normal(key, (*w.shape[:-2], w.shape[-1]), w.dtype)

    def body(_, v):
        u = jnp.einsum("...ab,...b->...a", w, v)
        v = jnp.einsum("...ab,...a->...b", w, u)
        return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-30)

    v = jax.lax.fori_loop(0, POWER_ITERATIONS, body, v)
    return jnp.linalg.norm(jnp.einsum("...ab,...b->...a", w, v), axis=-1)


def matrix_stats(w: jnp.ndarray, key: jax.Array) -> dict[str, jnp.ndarray]:
    """Mean and worst-case stable rank and output-channel norm ratio over the leading axes of [..., in, out]."""
    w = w.astype(jnp.float32)
    fro2 = jnp.sum(w * w, axis=(-2, -1))
    stable_rank = fro2 / jnp.maximum(spectral_norm(w, key) ** 2, 1e-30)
    channel = jnp.linalg.norm(w, axis=-2)
    ratio = jnp.max(channel, axis=-1) / jnp.maximum(jnp.mean(channel, axis=-1), 1e-30)
    return {
        "stable_rank_mean": jnp.mean(stable_rank),
        "stable_rank_min": jnp.min(stable_rank),
        "channel_ratio_mean": jnp.mean(ratio),
        "channel_ratio_max": jnp.max(ratio),
    }


def gain_stats(gain: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Distance of a learned RMSNorm gain from its initial 1."""
    dev = jnp.abs(gain.astype(jnp.float32) - 1.0)
    return {
        "gain_dev_max": jnp.max(dev),
        "gain_dev_mean": jnp.mean(dev),
        "gain_max": jnp.max(gain),
        "gain_min": jnp.min(gain),
    }
