# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_decay_mult``: the ``muonh`` group leaves its spheres and settles at the norm AdamC's decay
balances, ``r0 / sqrt(2 decay_mult)`` for independent steps, while the ``kda_beta`` group keeps MuonH."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_LR = 0.2
_STEPS = 300
# A 2-D matrix and a 3-D layer stack (one sphere per layer) in ``muonh``, and the KDA write-strength projection.
_SHAPES = {
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, 32, 32)}}},
    "gated": {"w_up": (32, 64)},
    "kda_blocks": {"stacked": {"attn": {"w_beta": (2, 32, 8)}}},
}
_W_O = ("stacked_blocks", "stacked", "attn", "w_o")
_W_UP = ("gated", "w_up")
_W_BETA = ("kda_blocks", "stacked", "attn", "w_beta")


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _sphere_norms(x):
    axes = tuple(range(x.ndim)) if x.ndim == 2 else tuple(range(1, x.ndim))
    return np.sqrt(np.sum(np.square(np.asarray(x)), axis=axes))


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def _mean_late_norm_ratio(config, params):
    """Train on independent random gradients; each leaf's sphere norms over the last third, relative to init."""
    optimizer = config.build(num_train_steps=1000)
    state = optimizer.init(params)
    late = {path: [] for path in (_W_O, _W_UP, _W_BETA)}
    current = params
    for step, key in enumerate(jax.random.split(jax.random.key(1), _STEPS)):
        updates, state = optimizer.update(_random_tree(key), state, current)
        current = jax.tree.map(jnp.add, current, updates)
        if step >= 2 * _STEPS // 3:
            for path, norms in late.items():
                norms.append(_sphere_norms(_leaf(current, path)) / _sphere_norms(_leaf(params, path)))
    return {path: np.mean(norms, axis=0) for path, norms in late.items()}


@pytest.mark.parametrize("decay_mult", [0.5, 2.5])
def test_norm_settles_where_adamc_decay_balances_the_step(decay_mult):
    params = _random_tree(jax.random.key(0))
    # No momentum, so consecutive steps are independent.
    base = GrugMoeMuonHConfig(learning_rate=_LR, warmup=0.0, lr_schedule="constant", momentum=0.0)
    config = dataclasses.replace(base, muonc_decay_mult=decay_mult)

    ratios = _mean_late_norm_ratio(config, params)

    # Fixed point of |W|^2 <- (1 - d)^2 |W|^2 + lr^2 r0^2 with d = decay_mult lr^2.
    decay = decay_mult * _LR**2
    expected = _LR / np.sqrt(1.0 - (1.0 - decay) ** 2)
    for path in (_W_O, _W_UP):
        np.testing.assert_allclose(ratios[path], expected, rtol=0.15)
    # The write-strength projection stays on its MuonH sphere.
    np.testing.assert_allclose(ratios[_W_BETA], 1.0, rtol=1e-4)
