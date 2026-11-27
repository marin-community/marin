# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_head_adamw``: under ``muonc_anneal_adamc`` the head ``[d, V]`` takes plain AdamW,
``W <- W - lr_h (u + 0.05 W)`` with ``u`` the per-element Adam direction and ``lr_h = lr_max s(t)^2``; the router and
embedding keep ``muonc_anneal_adamc``'s AdamC decay."""

import jax
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_HEAD_DECAY = 0.05
_NUM_TRAIN_STEPS = 20
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, _D, _D)}, "mlp": {"router": (2, _D, 4)}}},
    "output_proj": (_D, _V),
}


def _config(head_adamw: bool):
    return GrugMoeMuonHConfig(
        learning_rate=0.1,
        adam_lr=0.01,
        warmup=0.0,
        muonc_anneal_decay=0.2,
        muonc_anneal_adamc=12.5,
        adamc_adam_decay=0.3,
        muonc_anneal_head_adamw=head_adamw,
    )


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _updates(config, params, steps):
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    state = optimizer.init(params)
    out = []
    for step in range(steps):
        grads = _random_tree(jax.random.key(10 + step))
        updates, state = optimizer.update(grads, state, params)
        out.append((grads, updates))
    return out


def test_head_takes_per_element_adamw_at_the_squared_schedule():
    config = _config(head_adamw=True)
    params = _random_tree(jax.random.key(0))
    schedule = config.lr_scheduler(_NUM_TRAIN_STEPS)
    head = np.asarray(params["output_proj"])
    m = np.zeros_like(head)
    v = np.zeros_like(head)
    for step, (grads, updates) in enumerate(_updates(config, params, 3)):
        g = np.asarray(grads["output_proj"])
        m = config.beta1 * m + (1 - config.beta1) * g
        v = config.beta2 * v + (1 - config.beta2) * g * g
        m_hat = m / (1 - config.beta1 ** (step + 1))
        v_hat = v / (1 - config.beta2 ** (step + 1))
        u = m_hat / (np.sqrt(v_hat) + config.epsilon)
        lr = float(schedule(step))
        head_lr = lr * lr / config.learning_rate
        expected = -head_lr * (u + _HEAD_DECAY * head)
        np.testing.assert_allclose(np.asarray(updates["output_proj"]), expected, rtol=1e-4, atol=1e-7)


def test_router_embed_and_matrices_keep_the_adamc_rules():
    params = _random_tree(jax.random.key(1))
    with_adamw = _updates(_config(head_adamw=True), params, 2)
    adamc = _updates(_config(head_adamw=False), params, 2)
    for (_, on), (_, off) in zip(with_adamw, adamc, strict=True):
        for name in ("token_embed", "stacked_blocks"):
            jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), on[name], off[name])
        assert not np.allclose(np.asarray(on["output_proj"]), np.asarray(off["output_proj"]))
