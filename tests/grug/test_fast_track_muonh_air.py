# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonh_air`` (with ``muonh_radial_softcap``): the matrices start at sqrt(8) x their init, the matrix
sphere and log-radius steps follow the AIR schedule with 1.05 x the recipe's summed matrix LR, the head's LR follows
``lr_max s(t)^2`` and the head logit law is on."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.optimizer import (
    AIR_CUMULATIVE_MULT,
    GrugMoeMuonHConfig,
    _air_matrix_lr,
    _power_decay_schedule,
)

_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "output_proj": (_D, _V),
    "stacked_blocks": {"stacked": {"attn": {"w_q": (2, _D, _D), "w_o": (2, _D, _D)}, "norm": {"weight": (2, _D)}}},
}
_W_Q = ("stacked_blocks", "stacked", "attn", "w_q")
_W_O = ("stacked_blocks", "stacked", "attn", "w_o")
_HEAD = ("output_proj",)


def _params(seed=0):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _air(**overrides):
    return GrugMoeMuonHConfig(
        learning_rate=0.1, warmup=0.0, muonh_radial_softcap=2.0, muonh_air=True, embed_lr_mult=4.0, **overrides
    )


def test_air_matrix_lr_decays_from_its_peak_with_5pct_more_total_lr():
    num_train_steps, peak_lr, floor = 10767, 0.0135, 0.05
    warmup = round(0.01 * num_train_steps)
    recipe = _power_decay_schedule(peak_lr, floor * peak_lr, warmup, num_train_steps, 0.7)
    schedule, peak = _air_matrix_lr(recipe, num_train_steps, warmup, floor * peak_lr)
    got = np.asarray(jax.vmap(schedule)(jnp.arange(num_train_steps)), np.float64)
    base = np.asarray(jax.vmap(recipe)(jnp.arange(num_train_steps)), np.float64)

    assert np.sum(got) == pytest.approx(AIR_CUMULATIVE_MULT * np.sum(base), rel=1e-5)
    np.testing.assert_allclose(got[:warmup], base[:warmup], rtol=1e-6)
    assert got[warmup] == pytest.approx(peak, rel=1e-6)
    assert peak / peak_lr == pytest.approx(1.39, abs=0.02)
    assert np.all(np.diff(got[warmup:]) < 0)
    assert got[-1] == pytest.approx(floor * peak_lr, rel=0.01)
    # alpha = K/4: q falls to 1/sqrt(2) a quarter of the way through, so f is (1/sqrt(2) - 1/sqrt(5)) / (1 - 1/sqrt(5)).
    quarter = warmup + (num_train_steps - warmup) // 4
    f_quarter = (2**-0.5 - 5**-0.5) / (1 - 5**-0.5)
    assert got[quarter] == pytest.approx(floor * peak_lr + (peak - floor * peak_lr) * f_quarter, rel=1e-3)


def test_sphere_radius_and_head_steps_follow_their_schedules():
    num_train_steps, step = 40, 12
    on = _air()
    params, grads = _params(1), _params(2)
    air, _ = _air_matrix_lr(on.lr_scheduler(num_train_steps), num_train_steps, 0, on.min_lr_ratio * on.learning_rate)
    lr = float(on.lr_scheduler(num_train_steps)(step))

    def update_at(config):
        optimizer = config.build(num_train_steps=num_train_steps)
        state = optimizer.init(params)
        zeros = jax.tree.map(jnp.zeros_like, params)
        for _ in range(step):
            _, state = optimizer.update(zeros, state, params)
        updates, _ = optimizer.update(grads, state, params)
        return updates

    def softcap_at(rate):
        return GrugMoeMuonHConfig(
            learning_rate=rate,
            adam_lr=on.adam_lr,
            warmup=0.0,
            lr_schedule="constant",
            muonh_radial_softcap=2.0,
            embed_lr_mult=4.0,
        )

    got = update_at(on)
    want_matrix = update_at(softcap_at(float(air(step))))
    want_head = update_at(softcap_at(lr * lr / on.learning_rate))
    for path in (_W_Q, _W_O):
        np.testing.assert_allclose(_leaf(got, path), _leaf(want_matrix, path), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(_leaf(got, _HEAD), _leaf(want_head, _HEAD), rtol=1e-4, atol=1e-6)


def test_matrices_start_at_sqrt8_and_the_head_law_is_on():
    config = _air()
    params = _params()
    scaled = config.scale_initial_params(params)
    for path in (_W_Q, _W_O):
        np.testing.assert_allclose(_leaf(scaled, path), math.sqrt(8) * _leaf(params, path), rtol=1e-6)
    for path in (_HEAD, ("token_embed",), ("stacked_blocks", "stacked", "norm", "weight")):
        np.testing.assert_array_equal(_leaf(scaled, path), _leaf(params, path))

    optimizer = config.build(num_train_steps=20)
    updates, _ = optimizer.update(jax.tree.map(lambda p: 0.01 * p, scaled), optimizer.init(scaled), scaled)
    assert all(np.all(np.isfinite(np.asarray(u))) for u in jax.tree.leaves(updates))
    assert config.head_logit_mult(20, hidden_dim=_D, vocab_size=16384, initializer_std=0.02) > 1.0
    off = dataclasses.replace(config, muonh_air=False)
    assert off.head_logit_mult(20, hidden_dim=_D, vocab_size=16384, initializer_std=0.02) == 0.0


def test_needs_muonh_radial_softcap():
    with pytest.raises(ValueError, match="muonh_air"):
        GrugMoeMuonHConfig(muonh_air=True)
