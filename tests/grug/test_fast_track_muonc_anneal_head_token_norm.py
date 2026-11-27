# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_head_token_norm``: under ``muonc_anneal_adamc`` the head ``[d, V]`` steps along a
per-token normalized momentum direction at ``lr_h = lr_max s(t)^2``, ``W <- W - lr_h 0.02 sqrt(d) U - 12.5 lr_h^2 W``;
the router and embedding keep ``muonc_anneal_adamc``'s AdamC decay."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_HEAD_DECAY_MULT = 12.5
_ADAM_DECAY = 0.3
_NUM_TRAIN_STEPS = 20
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, _D, _D)}, "mlp": {"router": (2, _D, 4)}}},
    "output_proj": (_D, _V),
}


def _config():
    return GrugMoeMuonHConfig(
        learning_rate=0.1,
        adam_lr=0.01,
        warmup=0.0,
        muonc_anneal_decay=0.2,
        muonc_anneal_adamc=_HEAD_DECAY_MULT,
        adamc_adam_decay=_ADAM_DECAY,
        muonc_anneal_head_token_norm=True,
    )


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _reference_direction(n):
    n = n - n.mean(axis=-1, keepdims=True)
    n = n / (np.linalg.norm(n, axis=-2, keepdims=True) + 1e-8)
    return n - n.mean(axis=-1, keepdims=True)


def test_head_follows_token_normalized_nesterov_momentum():
    config = _config()
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    params = _random_tree(jax.random.key(0))
    state = optimizer.init(params)
    schedule = config.lr_scheduler(_NUM_TRAIN_STEPS)
    head = np.asarray(params["output_proj"])
    momentum = np.zeros_like(head)
    for step in range(3):
        grads = _random_tree(jax.random.key(10 + step))
        updates, state = optimizer.update(grads, state, params)
        g = np.asarray(grads["output_proj"])
        momentum = 0.95 * momentum + 0.05 * g
        u = _reference_direction(0.05 * g + 0.95 * momentum)
        lr = float(schedule(step))
        head_lr = lr * lr / config.learning_rate
        expected = -head_lr * 0.02 * np.sqrt(_D) * u - _HEAD_DECAY_MULT * head_lr**2 * head
        np.testing.assert_allclose(np.asarray(updates["output_proj"]), expected, rtol=1e-4, atol=1e-7)
        # The step moves no logit uniformly across the vocabulary, and every token column has a unit-sized direction.
        np.testing.assert_allclose(u.sum(axis=-1), 0.0, atol=1e-5)
        assert np.all(np.linalg.norm(u, axis=-2) > 0.5)


def test_zero_gradient_head_update_is_the_adamc_decay():
    config = _config()
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    params = _random_tree(jax.random.key(1))
    state = optimizer.init(params)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), state, params)
    lr = float(config.lr_scheduler(_NUM_TRAIN_STEPS)(0))
    head_lr = lr * lr / config.learning_rate
    np.testing.assert_allclose(
        np.asarray(updates["output_proj"]),
        -_HEAD_DECAY_MULT * head_lr**2 * np.asarray(params["output_proj"]),
        rtol=1e-4,
    )
