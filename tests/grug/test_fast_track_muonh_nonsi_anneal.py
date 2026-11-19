# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonh_nonsi_anneal``: MuonH spheres learn an unbounded log-radius, the head's LR follows
``lr_max s(t)^2`` and, from ``t0``, the MuonH sphere step follows ``lr_t s(t) / s(t0)``."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import LATE_ANNEAL_FRAC, GrugMoeMuonHConfig

# A free 2-D matrix and a 3-D layer stack (one sphere per layer) beside the frozen q projection, MLA KV
# down-projection and MoE latent down-projection, and the output head.
_SHAPES = {
    "stacked_blocks": {
        "stacked": {
            "attn": {"w_q": (2, 8, 8), "w_dkv": (2, 8, 4), "w_o": (2, 8, 8)},
            "mlp": {"w_latent_down": (2, 8, 4)},
        }
    },
    "gated": {"w_up": (4, 8)},
    "output_proj": (8, 16),
}
_W_Q = ("stacked_blocks", "stacked", "attn", "w_q")
_W_O = ("stacked_blocks", "stacked", "attn", "w_o")
_HEAD = ("output_proj",)
_FROZEN = (_W_Q, ("stacked_blocks", "stacked", "attn", "w_dkv"), ("stacked_blocks", "stacked", "mlp", "w_latent_down"))


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _sphere_norms(x):
    axes = tuple(range(x.ndim)) if x.ndim == 2 else tuple(range(1, x.ndim))
    return np.sqrt(np.sum(np.square(np.asarray(x)), axis=axes))


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def _train(config, params, grad_sign, num_steps):
    """``num_steps`` steps on the loss ``grad_sign * |W|^2 / 2`` of every leaf (gradient ``grad_sign * W``)."""
    optimizer = config.build(num_train_steps=1000)
    state = optimizer.init(params)
    for _ in range(num_steps):
        grads = jax.tree.map(lambda p: grad_sign * p, params)
        updates, state = optimizer.update(grads, state, params)
        params = jax.tree.map(jnp.add, params, updates)
    return params


def _update_at(config, num_train_steps, params, grads, step):
    """The update for ``grads`` at ``step``, after ``step`` zero-gradient updates (which move nothing)."""
    optimizer = config.build(num_train_steps=num_train_steps)
    state = optimizer.init(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    for _ in range(step):
        _, state = optimizer.update(zeros, state, params)
    updates, _ = optimizer.update(grads, state, params)
    return updates


def test_radius_grows_without_bound_while_frozen_matrices_keep_muonh():
    params = _random_tree(jax.random.key(0))
    off = GrugMoeMuonHConfig(learning_rate=0.2, warmup=0.0, lr_schedule="constant")
    on = dataclasses.replace(off, muonh_nonsi_anneal=True)

    labels = on.create_mask(params)
    attn = labels["stacked_blocks"]["stacked"]["attn"]
    assert (attn["w_q"], attn["w_dkv"], attn["w_o"]) == ("muonh", "muonh", "muonh_radial")
    assert labels["gated"]["w_up"] == "muonh_radial"

    # A negative gradient along W pulls every radius outward, past where muonh_radial_softcap=2 would stop it.
    grown_on = _train(on, params, grad_sign=-1.0, num_steps=40)
    grown_off = _train(off, params, grad_sign=-1.0, num_steps=40)
    for path in (_W_O, ("gated", "w_up")):
        ratio = _sphere_norms(_leaf(grown_on, path)) / _sphere_norms(_leaf(params, path))
        assert np.all(ratio > math.exp(3.0)), ratio
    for path in _FROZEN:
        np.testing.assert_array_equal(_leaf(grown_on, path), _leaf(grown_off, path))


def test_head_and_late_matrix_steps_follow_the_squared_schedule():
    num_train_steps = 20
    late_anneal_start = round(LATE_ANNEAL_FRAC * num_train_steps)
    params = _random_tree(jax.random.key(1))
    grads = _random_tree(jax.random.key(2))
    on = GrugMoeMuonHConfig(learning_rate=0.1, warmup=0.0, muonh_nonsi_anneal=True)
    schedule = on.lr_scheduler(num_train_steps)

    def plain_at(lr):
        return GrugMoeMuonHConfig(learning_rate=lr, adam_lr=on.adam_lr, warmup=0.0, lr_schedule="constant")

    for step in (late_anneal_start - 4, late_anneal_start + 2):
        lr = float(schedule(step))
        matrix_lr = lr * lr / float(schedule(late_anneal_start)) if step >= late_anneal_start else lr
        got = _update_at(on, num_train_steps, params, grads, step)
        want_matrix = _update_at(plain_at(matrix_lr), num_train_steps, params, grads, step)
        want_head = _update_at(plain_at(lr * lr / on.learning_rate), num_train_steps, params, grads, step)
        np.testing.assert_allclose(_leaf(got, _W_Q), _leaf(want_matrix, _W_Q), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(_leaf(got, _HEAD), _leaf(want_head, _HEAD), rtol=1e-4, atol=1e-6)
