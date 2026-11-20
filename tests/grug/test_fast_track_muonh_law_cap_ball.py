# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonh_law_cap_ball``: matrices take a plain Muon step at the late-annealed matrix LR inside a per-slice
Frobenius ball of radius ``kappa |U|``, and the head takes AdamH at ``lr_max s(t)^2``."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.adamh import scale_by_adamh
from experiments.grug.fast_track.optimizer import (
    KDA_BETA_LR_MULT,
    LATE_ANNEAL_FRAC,
    LAW_CAP_LAMBDA,
    LAW_CAP_TAU,
    GrugMoeMuonHConfig,
)

_NUM_TRAIN_STEPS = 20
_LR = 0.1
# A 4-D expert stack, a 3-D layer stack and a 2-D matrix in ``muonh``, the KDA write-strength stack (``kda_beta``,
# same shape as ``w_o`` so the two see the same Muon direction for the same gradient), and the output head.
_SHAPES = {
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, 8, 4)}, "mlp": {"w_experts": (2, 3, 4, 8)}}},
    "kda_blocks": {"stacked": {"attn": {"w_beta": (2, 8, 4)}}},
    "gated": {"w_up": (4, 8)},
    "output_proj": (8, 16),
}
_W_O = ("stacked_blocks", "stacked", "attn", "w_o")
_EXPERTS = ("stacked_blocks", "stacked", "mlp", "w_experts")
_W_BETA = ("kda_blocks", "stacked", "attn", "w_beta")
_W_UP = ("gated", "w_up")
_HEAD = ("output_proj",)
_MATRICES = (_W_O, _EXPERTS, _W_BETA, _W_UP)


def _config():
    return GrugMoeMuonHConfig(learning_rate=_LR, adam_lr=0.01, warmup=0.0, muonh_law_cap_ball=True)


def _random_tree(key, scale=1.0):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [scale * jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _with_leaf(tree, path, value):
    tree = jax.tree.map(lambda x: x, tree)
    node = tree
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return tree


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _update_at(params, grads, step):
    """The update for ``grads`` at ``step``, after ``step`` zero-gradient updates (which leave every moment at zero)."""
    optimizer = _config().build(num_train_steps=_NUM_TRAIN_STEPS)
    state = optimizer.init(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    for _ in range(step):
        _, state = optimizer.update(zeros, state, params)
    updates, _ = optimizer.update(grads, state, params)
    return updates


def _slice_norms(x):
    return np.sqrt(np.sum(np.square(x), axis=tuple(range(1, x.ndim)) if x.ndim > 2 else None))


def _expected_cap(shape):
    fan_in, fan_out = shape[-2:]
    num_matrices = math.prod(shape[1:-2])
    kappa = math.sqrt(LAW_CAP_TAU * _LR / (2 * LAW_CAP_LAMBDA))
    return kappa * math.sqrt(num_matrices * min(fan_in, fan_out) * max(1.0, fan_out / fan_in))


def test_matrices_outside_the_ball_land_on_their_cap():
    params = _random_tree(jax.random.key(0), scale=20.0)
    updates = _update_at(params, _random_tree(jax.random.key(1)), step=2)
    for path in _MATRICES:
        new = _leaf(params, path) + _leaf(updates, path)
        np.testing.assert_allclose(_slice_norms(new), _expected_cap(new.shape), rtol=1e-4)


def test_inside_the_ball_the_step_is_plain_muon_at_the_annealed_lr():
    params = _random_tree(jax.random.key(2), scale=1e-3)
    grads = _random_tree(jax.random.key(3))
    # The KDA write strength sees w_o's gradient, so its update is w_o's at KDA_BETA_LR_MULT x the LR.
    grads = _with_leaf(grads, _W_BETA, jnp.asarray(_leaf(grads, _W_O)))
    schedule = _config().lr_scheduler(_NUM_TRAIN_STEPS)
    late_anneal_start = round(LATE_ANNEAL_FRAC * _NUM_TRAIN_STEPS)
    before, after = late_anneal_start - 4, late_anneal_start + 2

    early = _update_at(params, grads, before)
    late = _update_at(params, grads, after)
    lr_before, lr_after = float(schedule(before)), float(schedule(after))
    matrix_lr_after = lr_after * lr_after / float(schedule(late_anneal_start))
    for path in (_W_O, _EXPERTS, _W_UP):
        np.testing.assert_allclose(
            _leaf(late, path), _leaf(early, path) * matrix_lr_after / lr_before, rtol=1e-4, atol=1e-8
        )
    np.testing.assert_allclose(_leaf(early, _W_BETA), KDA_BETA_LR_MULT * _leaf(early, _W_O), rtol=1e-4, atol=1e-8)

    # A plain Muon step: no decay pull and no projection, so the step does not depend on the matrix's scale.
    larger = _update_at(jax.tree.map(lambda p: 10.0 * p, params), grads, before)
    for path in _MATRICES:
        np.testing.assert_allclose(_leaf(larger, path), _leaf(early, path), rtol=1e-4, atol=1e-8)
        assert np.linalg.norm(_leaf(early, path)) > 0.1 * lr_before


def test_head_takes_adamh_at_the_squared_schedule():
    params = _random_tree(jax.random.key(4))
    grads = _random_tree(jax.random.key(5))
    step = 7
    lr = float(_config().lr_scheduler(_NUM_TRAIN_STEPS)(step))
    head_lr = lr * lr / _LR

    reference = scale_by_adamh(0.9, 0.95, 1e-8, head_lr)
    head_params, head_grads = jnp.asarray(_leaf(params, _HEAD)), jnp.asarray(_leaf(grads, _HEAD))
    state = reference.init(head_params)
    for _ in range(step):
        _, state = reference.update(jnp.zeros_like(head_grads), state, head_params)
    expected, _ = reference.update(head_grads, state, head_params)
    updates = _update_at(params, grads, step)
    np.testing.assert_allclose(_leaf(updates, _HEAD), np.asarray(expected), rtol=1e-4, atol=1e-8)
