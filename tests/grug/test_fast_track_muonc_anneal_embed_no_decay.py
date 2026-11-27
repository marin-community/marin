# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_embed_no_decay``: under ``muonc_anneal_adamc`` ``token_embed`` takes Adam at the Adam LR
schedule with no weight decay; the router keeps ``muonc_anneal_adamc``'s AdamC decay."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_NUM_TRAIN_STEPS = 20
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, _D, _D)}, "mlp": {"router": (2, _D, 4)}}},
    "output_proj": (_D, _V),
}


def _config(no_decay: bool):
    return GrugMoeMuonHConfig(
        learning_rate=0.1,
        adam_lr=0.01,
        warmup=0.0,
        muonc_anneal_decay=0.2,
        muonc_anneal_adamc=12.5,
        adamc_adam_decay=0.3,
        muonc_anneal_embed_no_decay=no_decay,
    )


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _updates(config, params, grads_list):
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    state = optimizer.init(params)
    out = []
    for grads in grads_list:
        updates, state = optimizer.update(grads, state, params)
        out.append(updates)
    return out


def test_token_embed_takes_plain_adam_without_decay():
    config = _config(no_decay=True)
    params = _random_tree(jax.random.key(0))
    grads_list = [_random_tree(jax.random.key(10 + step)) for step in range(3)]
    adam_schedule = config.lr_scheduler(_NUM_TRAIN_STEPS, override_lr=config.adam_lr)
    m = np.zeros((_V, _D))
    v = np.zeros((_V, _D))
    for step, updates in enumerate(_updates(config, params, grads_list)):
        g = np.asarray(grads_list[step]["token_embed"])
        m = config.beta1 * m + (1 - config.beta1) * g
        v = config.beta2 * v + (1 - config.beta2) * g * g
        m_hat = m / (1 - config.beta1 ** (step + 1))
        v_hat = v / (1 - config.beta2 ** (step + 1))
        expected = -float(adam_schedule(step)) * m_hat / (np.sqrt(v_hat) + config.epsilon)
        np.testing.assert_allclose(np.asarray(updates["token_embed"]), expected, rtol=1e-4, atol=1e-7)


def test_zero_gradient_leaves_token_embed_but_still_decays_the_router():
    params = _random_tree(jax.random.key(1))
    zeros = [jax.tree.map(jnp.zeros_like, params)]
    (on,) = _updates(_config(no_decay=True), params, zeros)
    (off,) = _updates(_config(no_decay=False), params, zeros)
    np.testing.assert_array_equal(np.asarray(on["token_embed"]), 0.0)
    assert np.all(np.abs(np.asarray(off["token_embed"])) > 0)
    for name in ("stacked_blocks", "output_proj"):
        jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), on[name], off[name])
    assert np.all(np.abs(np.asarray(on["stacked_blocks"]["stacked"]["mlp"]["router"])) > 0)
