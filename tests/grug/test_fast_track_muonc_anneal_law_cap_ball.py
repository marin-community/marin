# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_law_cap_ball``: the matrices take ``muonh_law_cap_ball``'s update while the head, router,
embedding and other Adam leaves take ``muonc_anneal_decay`` + ``muonc_anneal_adamc``'s."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_NUM_TRAIN_STEPS = 20
_SHAPES = {
    "token_embed": (16, 8),
    "output_proj": (8, 16),
    "stacked_blocks": {
        "stacked": {
            "attn": {"w_o": (2, 8, 4), "attn_gate": (2, 8, 4)},
            "mlp": {"w_experts": (2, 3, 4, 8), "router": (2, 8, 3)},
        }
    },
    "kda_blocks": {"stacked": {"attn": {"w_beta": (2, 8, 4)}}},
}
_MATRICES = (
    ("stacked_blocks", "stacked", "attn", "w_o"),
    ("stacked_blocks", "stacked", "mlp", "w_experts"),
    ("kda_blocks", "stacked", "attn", "w_beta"),
)
_ADAM_SIDE = (
    ("output_proj",),
    ("token_embed",),
    ("stacked_blocks", "stacked", "mlp", "router"),
    ("stacked_blocks", "stacked", "attn", "attn_gate"),
)
_X0250 = GrugMoeMuonHConfig(
    learning_rate=0.1,
    adam_lr=0.01,
    warmup=0.0,
    muonc_anneal_decay=0.2,
    muonc_anneal_adamc=12.5,
    adamc_adam_decay=0.3,
)
_LAW_CAP = GrugMoeMuonHConfig(learning_rate=0.1, adam_lr=0.01, warmup=0.0, muonh_law_cap_ball=True)
_ON = dataclasses.replace(_X0250, muonc_anneal_law_cap_ball=True)


def _random_tree(key, scale=1.0):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [scale * jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _updates(config, params, steps):
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    state = optimizer.init(params)
    for key in jax.random.split(jax.random.key(7), steps):
        updates, state = optimizer.update(_random_tree(key), state, params)
    return updates


# Steps before and after the late anneal start (round(0.7937 * 20) = 16); params inside and outside the ball.
@pytest.mark.parametrize("steps", [5, 18])
@pytest.mark.parametrize("scale", [1e-3, 20.0])
def test_matrices_follow_the_law_cap_ball_and_the_rest_follow_x0250(steps, scale):
    params = _random_tree(jax.random.key(0), scale=scale)
    on = _updates(_ON, params, steps)
    law_cap = _updates(_LAW_CAP, params, steps)
    x0250 = _updates(_X0250, params, steps)
    for path in _MATRICES:
        np.testing.assert_allclose(_leaf(on, path), _leaf(law_cap, path), rtol=1e-5, atol=1e-8)
    for path in _ADAM_SIDE:
        np.testing.assert_allclose(_leaf(on, path), _leaf(x0250, path), rtol=1e-5, atol=1e-8)


def test_no_matrix_decay_inside_the_ball():
    params = _random_tree(jax.random.key(1), scale=1e-3)
    small = _updates(_ON, params, 5)
    larger = _updates(_ON, jax.tree.map(lambda p: 10.0 * p, params), 5)
    for path in _MATRICES:
        np.testing.assert_allclose(_leaf(larger, path), _leaf(small, path), rtol=1e-4, atol=1e-8)
        assert float(jnp.linalg.norm(_leaf(small, path))) > 0
