# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Gaussian-process kernel and normalization constants."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

KERNEL_VARIANCE = 0.1
MAD_SCALE = 1.482602218505602
DISTANCE_BATCH_SIZE = 32


def squared_distances(x: jax.Array, y: jax.Array) -> jax.Array:
    # Direct differences keep identical profiles at exactly zero distance.
    # Bound the GPU reduction shape: a full vmap can make XLA compilation exhaust host RAM.
    return jax.lax.map(lambda row: jnp.square(row - y).sum(axis=-1), x, batch_size=DISTANCE_BATCH_SIZE)


@jax.jit
def kernel(x: ArrayLike, y: ArrayLike, lengthscale: ArrayLike, variance: ArrayLike = KERNEL_VARIANCE) -> jax.Array:
    """Matérn-5/2 covariance between rows of two feature matrices."""
    x, y = jnp.asarray(x), jnp.asarray(y)
    variance = jnp.asarray(variance)
    distance_squared = squared_distances(x, y)
    # The floor avoids sqrt's singular derivative at identical profiles.
    distance = jnp.sqrt(jnp.maximum(distance_squared, jnp.finfo(x.dtype).tiny))
    radius = jnp.sqrt(5) * distance / lengthscale
    return variance * (1 + radius + radius**2 / 3) * jnp.exp(-radius)
