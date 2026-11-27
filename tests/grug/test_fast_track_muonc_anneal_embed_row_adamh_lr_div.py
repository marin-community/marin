# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_embed_row_adamh_lr_div``: the row-wise AdamH step on ``token_embed`` takes
``adam_lr_t / c``, each row still rescaled back to its norm; every other leaf is unchanged."""

import jax
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_LR_DIV = 3.65
_NUM_TRAIN_STEPS = 20
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, _D, _D)}, "mlp": {"router": (2, _D, 4)}}},
    "output_proj": (_D, _V),
}


def _config(lr_div: float):
    return GrugMoeMuonHConfig(
        learning_rate=0.1,
        adam_lr=0.01,
        warmup=0.0,
        muonc_anneal_decay=0.2,
        muonc_anneal_adamc=12.5,
        adamc_adam_decay=0.3,
        muonc_anneal_embed_row_adamh=True,
        muonc_anneal_embed_row_adamh_lr_div=lr_div,
    )


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def test_token_embed_rows_turn_at_the_divided_adam_lr():
    config = _config(_LR_DIV)
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    params = _random_tree(jax.random.key(0))
    state = optimizer.init(params)
    adam_schedule = config.lr_scheduler(_NUM_TRAIN_STEPS, override_lr=config.adam_lr)
    m = np.zeros((_V, _D))
    v = np.zeros((_V, _D))
    for step in range(3):
        grads = _random_tree(jax.random.key(10 + step))
        updates, state = optimizer.update(grads, state, params)
        g = np.asarray(grads["token_embed"])
        m = config.beta1 * m + (1 - config.beta1) * g
        v = config.beta2 * v + (1 - config.beta2) * g * g
        u = (m / (1 - config.beta1 ** (step + 1))) / (np.sqrt(v / (1 - config.beta2 ** (step + 1))) + config.epsilon)
        w = np.asarray(params["token_embed"])
        stepped = w - float(adam_schedule(step)) / _LR_DIV * u
        renormed = stepped * np.linalg.norm(w, axis=-1, keepdims=True) / np.linalg.norm(stepped, axis=-1, keepdims=True)
        np.testing.assert_allclose(np.asarray(updates["token_embed"]), renormed - w, rtol=1e-4, atol=1e-6)
        params = jax.tree.map(lambda p, d: p + d, params, updates)


def test_other_leaves_match_the_undivided_run():
    params = _random_tree(jax.random.key(1))
    grads = _random_tree(jax.random.key(2))
    divided = _config(_LR_DIV).build(num_train_steps=_NUM_TRAIN_STEPS)
    undivided = _config(0.0).build(num_train_steps=_NUM_TRAIN_STEPS)
    on, _ = divided.update(grads, divided.init(params), params)
    off, _ = undivided.update(grads, undivided.init(params), params)
    for name in ("stacked_blocks", "output_proj"):
        jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), on[name], off[name])
    assert np.linalg.norm(np.asarray(on["token_embed"])) < np.linalg.norm(np.asarray(off["token_embed"]))
