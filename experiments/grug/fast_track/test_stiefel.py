# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semi-orthogonal LatentMoE projections: the init, Skewon's manifold-preserving step, and the optimizer routing."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import _latent_proj_init
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.stiefel import scale_with_stiefel_muon, stiefel_step


def _singular_values(x) -> np.ndarray:
    return np.linalg.svd(np.asarray(x, np.float64), compute_uv=False)


def test_orthogonal_init_is_semi_orthogonal_at_the_normal_inits_norm():
    cfg = t._config(latent_orthogonal_init=True, initializer_std=0.02)
    for shape in ((64, 16), (16, 64)):
        w = _latent_proj_init(cfg, jax.random.PRNGKey(0), shape)
        s = _singular_values(w)
        np.testing.assert_allclose(s, 0.02 * np.sqrt(64), rtol=1e-5)
        normal = _latent_proj_init(t._config(initializer_std=0.02), jax.random.PRNGKey(0), shape)
        np.testing.assert_allclose(np.linalg.norm(w), np.linalg.norm(normal), rtol=0.1)


def test_skewon_stays_on_the_manifold_and_maximizes_alignment():
    """Maximize <T, X> over scaled semi-orthogonal X: the optimum is c * polar(T), worth c * ||T||_nuclear."""
    rng = np.random.default_rng(0)
    c = 0.5
    x = jnp.asarray(c * np.linalg.qr(rng.standard_normal((3, 40, 12)))[0], jnp.float32)
    target = jnp.asarray(rng.standard_normal((3, 40, 12)), jnp.float32)
    alignment = []
    with jax.set_mesh(t._mesh()):
        for _ in range(300):
            # The loss -<T, X> has gradient -T.
            x = x + stiefel_step(x, -target, 0.05)
            alignment.append(float(jnp.sum(target * x)))
    for layer in np.asarray(x):
        np.testing.assert_allclose(_singular_values(layer), c, rtol=1e-4)
    optimum = c * sum(_singular_values(layer).sum() for layer in np.asarray(target))
    assert alignment[-1] > 0.99 * optimum
    assert alignment[-1] > alignment[0]


def test_wide_matrices_keep_orthonormal_rows():
    rng = np.random.default_rng(1)
    x = jnp.asarray(np.linalg.qr(rng.standard_normal((30, 10)))[0].T, jnp.float32)  # [10, 30], orthonormal rows
    with jax.set_mesh(t._mesh()):
        for _ in range(20):
            x = x + stiefel_step(x, jnp.asarray(rng.standard_normal((10, 30)), jnp.float32), 0.1)
    np.testing.assert_allclose(np.asarray(x @ x.T), np.eye(10), atol=1e-4)


def test_latent_projections_route_to_frozen_or_stiefel():
    _, model = t._model(latent_orthogonal_init=True)
    params = eqx.filter(model, eqx.is_inexact_array)
    for update, group in (("frozen", "frozen"), ("stiefel", "stiefel"), ("muonh", "muonh")):
        mask = GrugMoeMuonHConfig(latent_proj_update=update).create_mask(params)
        mlp = mask.kda_blocks.stacked.mlp
        assert mlp.w_latent_down == group and mlp.w_latent_up == group
        assert mlp.expert_mlp.w_up == "muonh"


def test_stiefel_transform_uses_nesterov_momentum_and_keeps_the_manifold():
    rng = np.random.default_rng(2)
    params = {"w": jnp.asarray(0.3 * np.linalg.qr(rng.standard_normal((2, 24, 8)))[0], jnp.float32)}
    transform = scale_with_stiefel_muon(momentum=0.9, nesterov=True, learning_rate=0.02)
    state = transform.init(params)
    with jax.set_mesh(t._mesh()):
        for _ in range(10):
            grads = {"w": jnp.asarray(rng.standard_normal((2, 24, 8)), jnp.float32)}
            updates, state = transform.update(grads, state, params)
            params = optax.apply_updates(params, updates)
    for layer in np.asarray(params["w"]):
        np.testing.assert_allclose(_singular_values(layer), 0.3, rtol=1e-4)
    assert float(jnp.abs(state.momentum["w"]).sum()) > 0
