# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PolyNorm expert activations."""

import jax
import jax.numpy as jnp

_POLYNORM_POWERS = 3


def _power_rms_terms(u: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
    """``(u^n, RMS(u^n))`` for n = 1..3 in float32, the RMS over the last (hidden-unit) axis per token."""
    z = u.astype(jnp.float32)
    powers = [z, z * z, z * z * z]
    return [(p, jnp.sqrt(jnp.mean(jnp.square(p), axis=-1, keepdims=True) + 1e-6)) for p in powers]


def polynorm(u: jax.Array) -> jax.Array:
    """PolyNorm with equal fixed coefficients (``UngatedExpertActivation.POLYNORM``)."""
    return (sum(p / rms for p, rms in _power_rms_terms(u)) / _POLYNORM_POWERS).astype(u.dtype)


def polynorm_over_input(u: jax.Array) -> jax.Array:
    """``polynorm(u) / u``, defined at 0 (no bias term): the gate activation for backends that tie the gate to
    ``W_up`` and compute ``act(u) * u``."""
    (_, rms1), (_, rms2), (_, rms3) = _power_rms_terms(u)
    z = u.astype(jnp.float32)
    return ((1.0 / rms1 + z / rms2 + z * z / rms3) / _POLYNORM_POWERS).astype(u.dtype)
