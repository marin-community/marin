# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_matrix_lr_mult``: under ``muonc_anneal_decay`` the ``muonh`` and ``kda_beta`` steps take ``c``
times their LR, ``W <- W - c lr_m (O + lambda_t W)``, while the head and the Adam groups keep their updates."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_MULT = math.sqrt(2.0)
_NUM_TRAIN_STEPS = 20
_D, _V = 8, 16
_SHAPES = {
    "token_embed": (_V, _D),
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, _D, _D)}, "mlp": {"router": (2, _D, 4)}}},
    "output_proj": (_D, _V),
}
_MATRIX = ("stacked_blocks", "stacked", "attn", "w_o")
_UNCHANGED = (("token_embed",), ("stacked_blocks", "stacked", "mlp", "router"), ("output_proj",))


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


def _updates(config, steps=3):
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    params = _random_tree(jax.random.key(0))
    state = optimizer.init(params)
    out = []
    for step in range(steps):
        updates, state = optimizer.update(_random_tree(jax.random.key(10 + step)), state, params)
        out.append(updates)
    return params, out


def test_matrix_step_scales_and_other_groups_are_unchanged():
    _, base = _updates(_config())
    _, scaled = _updates(_config(muonc_matrix_lr_mult=_MULT))
    for base_step, scaled_step in zip(base, scaled, strict=True):
        np.testing.assert_allclose(_leaf(scaled_step, _MATRIX), _MULT * _leaf(base_step, _MATRIX), rtol=1e-5, atol=1e-8)
        for path in _UNCHANGED:
            np.testing.assert_array_equal(_leaf(scaled_step, path), _leaf(base_step, path))


def test_shrink_per_step_matches_the_unscaled_run_at_lambda_over_c():
    """``c lr`` with ``lambda / c`` decays each matrix by the same ``lr_m lambda_t W`` as ``lr`` with ``lambda``."""
    zeros = jax.tree.map(jnp.zeros_like, _random_tree(jax.random.key(0)))

    def zero_gradient_update(config):
        optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
        params = _random_tree(jax.random.key(0))
        updates, _ = optimizer.update(zeros, optimizer.init(params), params)
        return _leaf(updates, _MATRIX)

    base = zero_gradient_update(_config())
    scaled = zero_gradient_update(
        GrugMoeMuonHConfig(
            learning_rate=0.1,
            adam_lr=0.01,
            warmup=0.0,
            muonc_anneal_decay=0.2 / _MULT,
            muonc_anneal_adamc=12.5,
            adamc_adam_decay=0.3,
            muonc_matrix_lr_mult=_MULT,
        )
    )
    np.testing.assert_allclose(scaled, base, rtol=1e-5)
