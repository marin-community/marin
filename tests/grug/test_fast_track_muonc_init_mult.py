# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_init_mult``: under ``muonc_anneal_decay`` the matrices of the ``muonh`` and ``kda_beta`` groups
start at ``muonc_init_mult`` times their init; the embedding, head, router, gains and Adam leaves keep their init."""

import jax
import numpy as np
import pytest

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_MULT = 2.0
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "output_proj": (_D, _V),
    "kda_blocks": {"stacked": {"attn": {"w_beta": (2, _D, 4), "w_q": (2, _D, _D), "a_log": (2, 4)}}},
    "stacked_blocks": {
        "stacked": {
            "attn": {"w_o": (2, _D, _D)},
            "mlp": {"router": (2, _D, 4), "w_up": (2, 4, _D, _D)},
            "attn_gated_norm": {"w_down": (2, _D, 2), "w_up": (2, 2, _D)},
            "norm": {"weight": (2, _D)},
        }
    },
}
_SCALED = (
    ("kda_blocks", "stacked", "attn", "w_beta"),
    ("kda_blocks", "stacked", "attn", "w_q"),
    ("stacked_blocks", "stacked", "attn", "w_o"),
    ("stacked_blocks", "stacked", "mlp", "w_up"),
    ("stacked_blocks", "stacked", "attn_gated_norm", "w_down"),
    ("stacked_blocks", "stacked", "attn_gated_norm", "w_up"),
)
_KEPT = (
    ("token_embed",),
    ("output_proj",),
    ("kda_blocks", "stacked", "attn", "a_log"),
    ("stacked_blocks", "stacked", "mlp", "router"),
    ("stacked_blocks", "stacked", "norm", "weight"),
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


def _params():
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(jax.random.key(0), len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def test_only_the_muonc_matrices_are_scaled():
    params = _params()
    scaled = _config(muonc_init_mult=_MULT).scale_initial_params(params)
    for path in _SCALED:
        np.testing.assert_array_equal(_leaf(scaled, path), _MULT * _leaf(params, path))
    for path in _KEPT:
        np.testing.assert_array_equal(_leaf(scaled, path), _leaf(params, path))


def test_scaled_params_train_with_the_x0250_optimizer():
    config = _config(muonc_init_mult=_MULT)
    params = config.scale_initial_params(_params())
    optimizer = config.build(num_train_steps=20)
    state = optimizer.init(params)
    grads = jax.tree.map(lambda p: 0.01 * p, params)
    updates, _ = optimizer.update(grads, state, params)
    assert all(np.all(np.isfinite(np.asarray(u))) for u in jax.tree.leaves(updates))


def test_needs_muonc_anneal_decay():
    with pytest.raises(ValueError, match="muonc_init_mult"):
        GrugMoeMuonHConfig(muonc_init_mult=_MULT).build(num_train_steps=20)
