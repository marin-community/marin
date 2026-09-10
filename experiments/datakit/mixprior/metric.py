# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metric feature construction, Gaussian conditioning, and expected objective scores."""

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from experiments.datakit.mixprior.objective import STANDARDIZED_METRIC_CAP

SPECTRUM_JITTER = 1e-6


def raw_features(weights: jax.Array, high_quality_membership: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [
            jnp.sqrt(weights).reshape(len(weights), -1),
            jnp.sqrt(weights.mean(axis=1)),
            jnp.sqrt(weights @ high_quality_membership).reshape(len(weights), -1),
        ],
        axis=1,
    )


def expected_score(mean: jax.Array, variance: jax.Array, target: jax.Array, hinge: jax.Array) -> jax.Array:
    """Integrate the clipped metric rewards and positive-part penalties."""
    sd = jnp.sqrt(jnp.maximum(variance, 1e-24))
    cap = STANDARDIZED_METRIC_CAP

    def positive(value: jax.Array) -> jax.Array:
        standardized = value / sd
        return sd * jnp.exp(-0.5 * standardized**2) / jnp.sqrt(2 * jnp.pi) + value * ndtr(standardized)

    upper = positive(mean - cap)
    clipped = -cap + positive(mean + cap) - upper
    penalty = positive(mean) - upper
    return -(clipped @ target + penalty @ hinge)


@jax.jit
def condition_metrics(
    covariance: jax.Array,
    counts: jax.Array,
    targets: jax.Array,
    noise: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    root_count = jnp.sqrt(counts)
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance * root_count[:, None] * root_count[None, :])
    inverse = 1 / (jnp.maximum(eigenvalues, 0)[:, None] + noise[None, :] + SPECTRUM_JITTER)
    projection = root_count[:, None] * eigenvectors
    alpha = projection @ ((projection.T @ targets) * inverse)
    return projection, inverse, alpha
