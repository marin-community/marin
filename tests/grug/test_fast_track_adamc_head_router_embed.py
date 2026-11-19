# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``adamc_head_router_embed``: the head leaves its AdamH sphere and settles where AdamC's decay balances
the step, and the router weights and ``token_embed`` take decay ``(adam_lr_t^2 / adam_lr) 0.1`` in place of the
router's annealed decay, leaving ``attn_gate`` and ``router_bias`` unchanged."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import ADAMC_ADAM_DECAY, GrugMoeMuonHConfig

_SHAPES = {
    "token_embed": (64, 32),
    "output_proj": (32, 64),
    "stacked_blocks": {
        "stacked": {
            "mlp": {"router": (2, 32, 8), "router_bias": (2, 8)},
            "attn": {"attn_gate": (2, 32, 4)},
        }
    },
}


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _head_norm_ratio(config, params, steps=300):
    """Train on independent random gradients; the head's mean norm over the last third, relative to init."""
    optimizer = config.build(num_train_steps=1000)
    state = optimizer.init(params)
    current = params
    late = []
    for step, key in enumerate(jax.random.split(jax.random.key(1), steps)):
        updates, state = optimizer.update(_random_tree(key), state, current)
        current = jax.tree.map(jnp.add, current, updates)
        if step >= 2 * steps // 3:
            late.append(np.linalg.norm(current["output_proj"]) / np.linalg.norm(params["output_proj"]))
    return np.mean(late)


def test_head_norm_settles_where_adamc_decay_balances_the_step():
    lr, decay_mult = 0.2, 2.5
    params = _random_tree(jax.random.key(0))
    # No first moment, so consecutive steps are independent.
    base = GrugMoeMuonHConfig(
        learning_rate=lr, warmup=0.0, lr_schedule="constant", beta1=0.0, muonc_decay_mult=decay_mult
    )

    adamh_ratio = _head_norm_ratio(base, params)
    adamc_ratio = _head_norm_ratio(dataclasses.replace(base, adamc_head_router_embed=True), params)

    np.testing.assert_allclose(adamh_ratio, 1.0, rtol=1e-4)
    # Fixed point of |W|^2 <- (1 - d)^2 |W|^2 + lr^2 r0^2 with d = decay_mult lr^2.
    decay = decay_mult * lr**2
    np.testing.assert_allclose(adamc_ratio, lr / np.sqrt(1.0 - (1.0 - decay) ** 2), rtol=0.15)


def test_router_and_embed_decay_follows_the_squared_adam_lr():
    total, step = 100, 50
    params = _random_tree(jax.random.key(0))
    base = GrugMoeMuonHConfig(adam_lr=0.01, warmup=0.0, lr_schedule="linear", min_lr_ratio=0.1, muonc_decay_mult=12.5)
    on = dataclasses.replace(base, adamc_head_router_embed=True)
    off_opt, on_opt = base.build(total), on.build(total)
    off_state, on_state = off_opt.init(params), on_opt.init(params)
    # Same gradients at the same parameters, so both runs hold the same Adam moments; only the decay differs.
    for key in jax.random.split(jax.random.key(1), step + 1):
        grads = _random_tree(key)
        off_updates, off_state = off_opt.update(grads, off_state, params)
        on_updates, on_state = on_opt.update(grads, on_state, params)

    adam_lr = float(base.lr_scheduler(total, override_lr=base.adam_lr)(step))
    assert adam_lr < base.adam_lr
    adamc = adam_lr**2 / base.adam_lr * ADAMC_ADAM_DECAY
    annealed = adam_lr * (1.0 - step / total) * base.gate_router_weight_decay

    on_router = on_updates["stacked_blocks"]["stacked"]["mlp"]["router"]
    off_router = off_updates["stacked_blocks"]["stacked"]["mlp"]["router"]
    router = params["stacked_blocks"]["stacked"]["mlp"]["router"]
    np.testing.assert_allclose(on_router - off_router, (annealed - adamc) * router, atol=1e-6)
    np.testing.assert_allclose(
        on_updates["token_embed"] - off_updates["token_embed"], -adamc * params["token_embed"], atol=1e-6
    )
    for path in (
        ("stacked_blocks", "stacked", "mlp", "router_bias"),
        ("stacked_blocks", "stacked", "attn", "attn_gate"),
    ):
        on_leaf, off_leaf = on_updates, off_updates
        for key in path:
            on_leaf, off_leaf = on_leaf[key], off_leaf[key]
        np.testing.assert_array_equal(on_leaf, off_leaf)
