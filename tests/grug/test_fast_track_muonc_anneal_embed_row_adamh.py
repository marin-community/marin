# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_embed_row_adamh``: under ``muonc_anneal_adamc`` each ``token_embed`` row takes the plain
Adam step at the Adam LR schedule and is rescaled back to its norm; rows absent from the batch are untouched, and the
router keeps ``muonc_anneal_adamc``'s AdamC decay."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_NUM_TRAIN_STEPS = 20
_D, _V = 8, 16
_ABSENT_ROWS = slice(0, 4)
_SHAPES = {
    "token_embed": (_V, _D),
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, _D, _D)}, "mlp": {"router": (2, _D, 4)}}},
    "output_proj": (_D, _V),
}


def _config(row_adamh: bool):
    return GrugMoeMuonHConfig(
        learning_rate=0.1,
        adam_lr=0.01,
        warmup=0.0,
        muonc_anneal_decay=0.2,
        muonc_anneal_adamc=12.5,
        adamc_adam_decay=0.3,
        muonc_anneal_embed_row_adamh=row_adamh,
    )


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _grads(step):
    grads = _random_tree(jax.random.key(10 + step))
    grads["token_embed"] = grads["token_embed"].at[_ABSENT_ROWS].set(0.0)
    return grads


def test_token_embed_rows_take_adam_then_return_to_their_norm():
    config = _config(row_adamh=True)
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    params = _random_tree(jax.random.key(0))
    state = optimizer.init(params)
    adam_schedule = config.lr_scheduler(_NUM_TRAIN_STEPS, override_lr=config.adam_lr)
    m = np.zeros((_V, _D))
    v = np.zeros((_V, _D))
    for step in range(3):
        grads = _grads(step)
        updates, state = optimizer.update(grads, state, params)
        g = np.asarray(grads["token_embed"])
        m = config.beta1 * m + (1 - config.beta1) * g
        v = config.beta2 * v + (1 - config.beta2) * g * g
        u = (m / (1 - config.beta1 ** (step + 1))) / (np.sqrt(v / (1 - config.beta2 ** (step + 1))) + config.epsilon)
        w = np.asarray(params["token_embed"])
        stepped = w - float(adam_schedule(step)) * u
        renormed = stepped * np.linalg.norm(w, axis=-1, keepdims=True) / np.linalg.norm(stepped, axis=-1, keepdims=True)
        expected = renormed - w
        expected[_ABSENT_ROWS] = 0.0
        delta = np.asarray(updates["token_embed"])
        np.testing.assert_allclose(delta, expected, rtol=1e-4, atol=1e-6)
        params = jax.tree.map(lambda p, d: p + d, params, updates)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(params["token_embed"]), axis=-1), np.linalg.norm(w, axis=-1), rtol=1e-5
        )


def test_router_and_head_keep_the_adamc_rules():
    params = _random_tree(jax.random.key(1))
    zeros = jax.tree.map(jnp.zeros_like, params)
    on_optimizer = _config(row_adamh=True).build(num_train_steps=_NUM_TRAIN_STEPS)
    off_optimizer = _config(row_adamh=False).build(num_train_steps=_NUM_TRAIN_STEPS)
    on, _ = on_optimizer.update(zeros, on_optimizer.init(params), params)
    off, _ = off_optimizer.update(zeros, off_optimizer.init(params), params)
    np.testing.assert_array_equal(np.asarray(on["token_embed"]), 0.0)
    assert np.all(np.abs(np.asarray(off["token_embed"])) > 0)
    for name in ("stacked_blocks", "output_proj"):
        jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), on[name], off[name])
