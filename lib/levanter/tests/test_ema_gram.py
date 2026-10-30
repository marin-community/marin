# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.models.qwen import Qwen3Config
from levanter.optim.ema_gram import (
    EmaGramConfig,
    EmaGramState,
    GramState,
    exact_polar,
    gram_direction,
    init_gram_state,
    relative_gram_power,
    scale_by_ema_gram,
    update_gram_state,
)
from levanter.optim.util import zeropower_via_newtonschulz5
from levanter.trainer_state import TrainerState


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (2, 3, 2)])
def test_ema_gram_accumulates_reduced_gradient_before_direction(shape):
    rng = np.random.default_rng(8)
    gradients = rng.normal(size=(4, *shape)).astype(np.float32)
    state = init_gram_state(jnp.asarray(gradients[0]))
    step = jax.jit(partial(update_gram_state, beta1=0.95, beta2=0.99))
    for gradient in gradients:
        state = step(gradient, state)
    # Closed-form weighted history, independent of the recurrent implementation.
    weights = 0.01 * 0.99 ** np.arange(3, -1, -1)
    left = np.einsum("t,t...ik,t...jk->...ij", weights, gradients, gradients)
    right = np.einsum("t,t...ki,t...kj->...ij", weights, gradients, gradients)
    momentum = np.einsum("t,t...ij->...ij", 0.05 * 0.95 ** np.arange(3, -1, -1), gradients)
    for actual, expected in zip(state, (momentum, left, right), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    # Opposite local gradients must cancel before constructing Grams.
    cancelled = step((gradients[0] - gradients[0]) / 2, init_gram_state(gradients[0]))
    np.testing.assert_array_equal(cancelled.left, 0)


@pytest.mark.parametrize("alpha", [0, 0.125, 0.25, 1 / 3, 0.5])
def test_ema_gram_exact_direction_matches_diagonal_spectral_oracle(alpha):
    m = jnp.array([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
    left = jnp.diag(jnp.array([1.0, 4.0]))
    right = jnp.diag(jnp.array([2.0, 5.0, 0.0]))
    state = GramState(m, left, right)
    result = jax.jit(partial(gram_direction, alpha=alpha, ridge=1e-2, polar_steps=5, polar_method="svd"))(state)
    expected = np.zeros((2, 3))
    expected[0, 0] = (1 / 2.5 + 0.01) ** -alpha * (2 / (7 / 3) + 0.01) ** -alpha
    expected[1, 1] = -((4 / 2.5 + 0.01) ** -alpha) * (5 / (7 / 3) + 0.01) ** -alpha
    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_zero_modes_and_zero_grams_do_not_acquire_updates():
    matrix = jnp.array([[3.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(exact_polar(matrix), [[1, 0], [0, 0]])
    np.testing.assert_array_equal(relative_gram_power(jnp.zeros((2, 2)), 0.5, 0.01), np.eye(2))
    for alpha in (0, 0.25, 0.5):
        direction = gram_direction(init_gram_state(matrix), alpha=alpha, ridge=0.01, polar_steps=5)
        np.testing.assert_array_equal(direction, 0)


def test_alpha_zero_tracks_passive_grams_and_matches_muon_polar():
    gradient = jnp.array([[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0]])
    tx = scale_by_ema_gram(alpha=0, ridge=1e-4)
    actual, state = jax.jit(tx.update)(gradient, tx.init(gradient))
    expected = zeropower_via_newtonschulz5(0.05 * gradient)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(state.matrices[0].left, 0.01 * (gradient @ gradient.T), atol=1e-6)


def test_scanned_qwen_optimizer_roundtrip_continues_identically(tmp_path):
    cfg = Qwen3Config(
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
        scan_layers=True,
    )
    model = cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(1))
    tx = EmaGramConfig(learning_rate=0.001, alpha=0.25, ridge=0.01).build(20)
    state = TrainerState.init(tx, model, key=jax.random.PRNGKey(2))
    grads = jax.tree.map(lambda x: jnp.sin(x) if eqx.is_array(x) else x, state.trainable_model)
    step = eqx.filter_jit(lambda s: s.take_step(grads, key=jax.random.PRNGKey(3))[0])
    state = step(state)
    grams = jax.tree.leaves(state.opt_state, is_leaf=lambda x: isinstance(x, EmaGramState))
    assert any(isinstance(x, EmaGramState) and len(x.matrices) > 0 for x in grams)
    save_checkpoint(state.saveable_state, 1, tmp_path / "checkpoint")
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
    restored = load_checkpoint(state, tmp_path / "checkpoint", mesh=mesh, axis_mapping={}, allow_partial=False)
    expected, actual = step(state), step(restored)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(actual.model))


@pytest.mark.tpu
def test_sharded_batch_reduction_precedes_gram_construction():
    devices = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devices.reshape((1, -1)), ("model", "data"))
    base = jnp.arange(48, dtype=jnp.float32).reshape(6, 8)
    contributions = jnp.stack([base, -base + 1, 2 * base, -2 * base + 1])
    contributions = jnp.tile(contributions, (len(devices), 1, 1))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None, None))
    with jax.set_mesh(mesh):
        reduced_step = jax.jit(
            lambda batches: update_gram_state(jnp.mean(batches, axis=0), init_gram_state(base), 0.95, 0.99),
            in_shardings=sharding,
        )
        state = reduced_step(jax.device_put(contributions, sharding))
    np.testing.assert_allclose(state.left, np.full((6, 6), 0.02), atol=1e-6)
    np.testing.assert_allclose(state.right, np.full((8, 8), 0.015), atol=1e-6)
