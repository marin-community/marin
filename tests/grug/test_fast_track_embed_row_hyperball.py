# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``embed_row_hyperball``: ``token_embed`` takes a per-row hyperball step on Nesterov momentum,
``W_i <- r_i normalize(W_i - theta_t r_i N_i / |N_i|)``, ``theta_t = theta adam_lr_t / adam_lr``, ``r_i`` the row's init
norm; the head, router and matrices keep the X0250 updates."""

import jax
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_THETA = 0.012
_NUM_TRAIN_STEPS = 20
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, _D, _D)}, "mlp": {"router": (2, _D, 4)}}},
    "output_proj": (_D, _V),
}
_UNCHANGED = (
    ("stacked_blocks", "stacked", "attn", "w_o"),
    ("stacked_blocks", "stacked", "mlp", "router"),
    ("output_proj",),
)


def _config(**overrides):
    return GrugMoeMuonHConfig(
        learning_rate=0.1,
        adam_lr=0.01,
        warmup=0.0,
        muonc_anneal_decay=0.2,
        muonc_anneal_adamc=12.5,
        adamc_adam_decay=0.3,
        **overrides,
    )


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_token_embed_follows_the_row_hyperball_step():
    config = _config(embed_row_hyperball=_THETA)
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    params = _random_tree(jax.random.key(0))
    state = optimizer.init(params)
    adam_schedule = config.lr_scheduler(_NUM_TRAIN_STEPS, override_lr=config.adam_lr)
    embed = np.asarray(params["token_embed"])
    init_norms = np.linalg.norm(embed, axis=-1, keepdims=True)
    momentum = np.zeros_like(embed)
    for step in range(4):
        grads = _random_tree(jax.random.key(10 + step))
        updates, state = optimizer.update(grads, state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
        g = np.asarray(grads["token_embed"])
        momentum = 0.95 * momentum + 0.05 * g
        n = 0.05 * g + 0.95 * momentum
        unit = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
        theta = _THETA * float(adam_schedule(step)) / config.adam_lr
        expected = init_norms * _normalize(embed - theta * init_norms * unit)
        np.testing.assert_allclose(np.asarray(params["token_embed"]), expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(expected, axis=-1, keepdims=True), init_norms, rtol=1e-5)
        cos = np.sum(_normalize(expected) * _normalize(embed), axis=-1)
        np.testing.assert_allclose(np.arccos(np.clip(cos, -1, 1)), np.arctan(theta * np.ones_like(cos)), atol=5e-3)
        embed = expected


def test_other_groups_keep_their_updates():
    params = _random_tree(jax.random.key(1))
    grads = _random_tree(jax.random.key(2))
    base = _config().build(num_train_steps=_NUM_TRAIN_STEPS)
    variant = _config(embed_row_hyperball=_THETA).build(num_train_steps=_NUM_TRAIN_STEPS)
    base_updates, _ = base.update(grads, base.init(params), params)
    variant_updates, _ = variant.update(grads, variant.init(params), params)
    for path in _UNCHANGED:
        np.testing.assert_allclose(_leaf(variant_updates, path), _leaf(base_updates, path), rtol=1e-6, atol=1e-9)
