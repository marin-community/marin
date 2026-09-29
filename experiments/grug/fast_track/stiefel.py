# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Skewon (arXiv 2608.06218): Muon on the Stiefel manifold, for matrices kept at a scaled semi-orthogonal point.

A leaf ``X = c * Xn`` (stacked ``[L, n, p]``, transposed when it is wide) keeps ``Xn^T Xn = I``: all singular
values equal ``c``, which is fixed at init. Per step, with Nesterov momentum ``M``:

- ``N = skew(M Xn^T)`` (``n x n``) and ``B = -msign(N) Xn``: the exact solution of Muon's subproblem
  ``min <M, B>`` over the tangent space with ``||B||_2 <= 1`` (the paper's Proposition 1).
- ``Xn <- polar(Xn + eta B)``, with ``||eta B||_F = lr * ||Xn||_F``: the same relative Frobenius step as MuonH's
  hyperball, so the MuonH learning rate carries over. ``msign`` uses quintic Newton-Schulz; the polar retraction
  uses cubic Newton-Schulz, which converges in a few iterations because the step leaves ``Xn`` nearly orthogonal.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS

MSIGN_STEPS = 5
RETRACTION_ITERS = 4


class StiefelState(NamedTuple):
    momentum: optax.Updates


def _msign(x: jax.Array, steps: int = MSIGN_STEPS) -> jax.Array:
    """Approximate matrix sign (polar factor) of square ``[..., n, n]`` matrices by quintic Newton-Schulz."""
    coeffs = NEWTON_SCHULZ_COEFFICIENTS["quintic"]
    x = x / (jnp.sqrt(jnp.sum(jnp.square(x), axis=(-2, -1), keepdims=True)) + 1e-7)
    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        gram = x @ jnp.swapaxes(x, -1, -2)
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x


def polar_retract(z: jax.Array, iters: int = RETRACTION_ITERS) -> jax.Array:
    """Polar factor of near-orthonormal-column ``[..., n, p]`` matrices by cubic Newton-Schulz."""
    eye = jnp.eye(z.shape[-1], dtype=z.dtype)
    for _ in range(iters):
        z = z @ (1.5 * eye - 0.5 * jnp.swapaxes(z, -1, -2) @ z)
    return z


def stiefel_step(x: jax.Array, direction: jax.Array, learning_rate) -> jax.Array:
    """The update (new minus old) that moves ``x`` one Skewon step against ``direction`` on its scaled Stiefel
    manifold. Leading axes are independent matrices (the layer stack)."""
    spec = jax.typeof(x).sharding.spec
    replicated = P(*(None,) * x.ndim)
    x32 = reshard(x.astype(jnp.float32), replicated)
    m32 = reshard(direction.astype(jnp.float32), replicated)
    wide = x.shape[-2] < x.shape[-1]
    if wide:
        x32, m32 = jnp.swapaxes(x32, -1, -2), jnp.swapaxes(m32, -1, -2)
    p = x32.shape[-1]
    scale = jnp.sqrt(jnp.sum(jnp.square(x32), axis=(-2, -1), keepdims=True) / p)
    xn = x32 / scale
    mx = m32 @ jnp.swapaxes(xn, -1, -2)
    tangent = -_msign(0.5 * (mx - jnp.swapaxes(mx, -1, -2))) @ xn
    tangent_norm = jnp.sqrt(jnp.sum(jnp.square(tangent), axis=(-2, -1), keepdims=True))
    eta = learning_rate * jnp.sqrt(float(p)) / jnp.maximum(tangent_norm, 1e-12)
    delta = scale * polar_retract(xn + eta * tangent) - x32
    if wide:
        delta = jnp.swapaxes(delta, -1, -2)
    return reshard(delta.astype(x.dtype), spec)


def scale_with_stiefel_muon(*, momentum: float, nesterov: bool, learning_rate) -> optax.GradientTransformation:
    """Skewon for every leaf of its group (see the module docstring). Emits the full parameter update, so no
    ``scale(-lr)`` follows it."""

    def init_fn(params):
        return StiefelState(momentum=jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params))

    def update_fn(updates, state, params):
        buffers = jax.tree.map(lambda m, g: momentum * m + g.astype(jnp.float32), state.momentum, updates)
        directions = (
            jax.tree.map(lambda g, m: g.astype(jnp.float32) + momentum * m, updates, buffers) if nesterov else buffers
        )
        new_updates = jax.tree.map(lambda x, d: stiefel_step(x, d, learning_rate), params, directions)
        return new_updates, StiefelState(momentum=buffers)

    return optax.GradientTransformation(init_fn, update_fn)
