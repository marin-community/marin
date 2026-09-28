# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EMA-Nesterov reproduces the modded-nanogpt record's lookahead trajectory, and MuonSphere pins every
matrix's spectral norm to its radius."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.grug.fast_track.optimizer import _spectral_sphere_init, _spectral_sphere_updates, ema_nesterov


def test_ema_nesterov_matches_the_record_lookahead():
    """The record (track-3 #40) moves the weights by ``s e`` before the forward, steps from there and never
    moves back; our parameters hold that lookahead point, so they equal the record's gradient points and end
    on its weights once the window closes."""
    a = jnp.diag(jnp.linspace(0.5, 3.0, 5))
    grad = lambda x: a @ x - 1.0  # noqa: E731
    lr, scale, decay, start, end, steps = 0.1, 0.3, 0.99, 3, 12, 20

    x = prev = ema = jnp.zeros(5)
    record_points = []
    for it in range(steps):
        x = x + (scale if start <= it < end else 0.0) * ema
        record_points.append(x)
        x = x - lr * grad(x)
        ema = decay * ema + (1 - decay) * (x - prev)
        prev = x

    tx = ema_nesterov(
        optax.sgd(lr),
        lambda p: jax.tree.map(lambda _: "adam", p),
        scale=scale,
        decay=decay,
        start_step=start,
        end_step=end,
        frobenius_groups=frozenset(),
    )
    params = {"w": jnp.zeros(5)}
    state = tx.init(params)
    for it in range(steps):
        np.testing.assert_allclose(params["w"], record_points[it], atol=1e-6)
        updates, state = tx.update({"w": grad(params["w"])}, state, params)
        params = optax.apply_updates(params, updates)
    np.testing.assert_allclose(params["w"], x, atol=1e-6)


def test_muonsphere_retracts_each_matrix_to_its_radius():
    """Stacked ``[L, E, in, out]`` weights: each step's retraction puts every matrix's sigma_1 at
    ``c sqrt(out / in)`` (warm-started power iteration), and the step itself moves it by at most ``lr R``."""
    key_w, key_u = jax.random.split(jax.random.PRNGKey(0))
    params = {"w": 0.05 * jax.random.normal(key_w, (2, 3, 16, 8))}
    radius = 2.0 * (8 / 16) ** 0.5
    lr = 0.05
    state = _spectral_sphere_init(params, 2.0)
    for step in range(3):
        direction = {"w": jax.random.normal(jax.random.fold_in(key_u, step), params["w"].shape)}
        updates, state = _spectral_sphere_updates(params, direction, lr, state)
        params = optax.apply_updates(params, updates)
        sigma = np.linalg.norm(np.asarray(params["w"]), ord=2, axis=(-2, -1))
        assert np.all(np.abs(sigma - radius) <= lr * radius * 1.01)
    # A zero-length step leaves only the retraction.
    updates, _ = _spectral_sphere_updates(params, direction, 0.0, state)
    retracted = np.asarray(optax.apply_updates(params, updates)["w"])
    np.testing.assert_allclose(np.linalg.norm(retracted, ord=2, axis=(-2, -1)), radius, rtol=1e-3)
