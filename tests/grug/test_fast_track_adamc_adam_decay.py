# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``adamc_adam_decay``: the router weights and ``token_embed`` take decay
``(adam_lr_t^2 / adam_lr) adamc_adam_decay`` in place of ``ADAMC_ADAM_DECAY``; every other leaf is unchanged."""

import dataclasses

import jax
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


def test_router_and_embed_decay_takes_the_set_lambda():
    total, step, decay = 100, 50, 0.3
    params = _random_tree(jax.random.key(0))
    base = GrugMoeMuonHConfig(
        adam_lr=0.01,
        warmup=0.0,
        lr_schedule="linear",
        min_lr_ratio=0.1,
        muonc_decay_mult=12.5,
        adamc_head_router_embed=True,
    )
    on = dataclasses.replace(base, adamc_adam_decay=decay)
    off_opt, on_opt = base.build(total), on.build(total)
    off_state, on_state = off_opt.init(params), on_opt.init(params)
    # Same gradients at the same parameters, so both runs hold the same Adam moments; only the decay differs.
    for key in jax.random.split(jax.random.key(1), step + 1):
        grads = _random_tree(key)
        off_updates, off_state = off_opt.update(grads, off_state, params)
        on_updates, on_state = on_opt.update(grads, on_state, params)

    adam_lr = float(base.lr_scheduler(total, override_lr=base.adam_lr)(step))
    assert adam_lr < base.adam_lr
    extra = adam_lr**2 / base.adam_lr * (decay - ADAMC_ADAM_DECAY)

    router = params["stacked_blocks"]["stacked"]["mlp"]["router"]
    on_router = on_updates["stacked_blocks"]["stacked"]["mlp"]["router"]
    off_router = off_updates["stacked_blocks"]["stacked"]["mlp"]["router"]
    np.testing.assert_allclose(on_router - off_router, -extra * router, atol=1e-6)
    np.testing.assert_allclose(
        on_updates["token_embed"] - off_updates["token_embed"], -extra * params["token_embed"], atol=1e-6
    )
    for path in (
        ("output_proj",),
        ("stacked_blocks", "stacked", "mlp", "router_bias"),
        ("stacked_blocks", "stacked", "attn", "attn_gate"),
    ):
        on_leaf, off_leaf = on_updates, off_updates
        for key in path:
            on_leaf, off_leaf = on_leaf[key], off_leaf[key]
        np.testing.assert_array_equal(on_leaf, off_leaf)
