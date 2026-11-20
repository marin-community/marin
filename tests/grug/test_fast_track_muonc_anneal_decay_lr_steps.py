# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_decay_lr_steps``: the MuonC matrix ``lambda`` scales by ``lr_steps / (lr_max N)``, so the
total matrix decay ``lambda lr_max N`` is the same at every learning rate and run length."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_LAMBDA = 0.2
_SHAPES = {"stacked_blocks": {"stacked": {"attn": {"w_o": (2, 8, 8)}}}, "gated": {"w_up": (4, 8)}}
_PATHS = (("stacked_blocks", "stacked", "attn", "w_o"), ("gated", "w_up"))
_STEP = 3


def _params():
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(jax.random.key(0), len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _zero_gradient_update(config, num_train_steps):
    """The update after ``_STEP`` zero-gradient steps: a pure matrix decay ``lr_m lambda_t W``."""
    params = _params()
    optimizer = config.build(num_train_steps=num_train_steps)
    state = optimizer.init(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    for _ in range(_STEP):
        _, state = optimizer.update(zeros, state, params)
    updates, _ = optimizer.update(zeros, state, params)
    return params, updates


@pytest.mark.parametrize(("learning_rate", "num_train_steps"), [(0.1, 40), (0.05, 100)])
def test_matrix_decay_scales_to_the_reference_lr_steps(learning_rate, num_train_steps):
    lr_steps = 0.1 * 40
    config = GrugMoeMuonHConfig(
        learning_rate=learning_rate,
        warmup=0.0,
        muonc_anneal_decay=_LAMBDA,
        muonc_anneal_decay_lr_steps=lr_steps,
    )
    lr = float(config.lr_scheduler(num_train_steps)(_STEP))
    effective_lambda = _LAMBDA * lr_steps / (learning_rate * num_train_steps)
    params, updates = _zero_gradient_update(config, num_train_steps)
    for path in _PATHS:
        np.testing.assert_allclose(
            _leaf(updates, path), -lr * effective_lambda * lr / learning_rate * _leaf(params, path), rtol=1e-4
        )


def test_reference_rung_keeps_its_lambda():
    learning_rate, num_train_steps = 0.1, 40
    base = GrugMoeMuonHConfig(learning_rate=learning_rate, warmup=0.0, muonc_anneal_decay=_LAMBDA)
    scaled = GrugMoeMuonHConfig(
        learning_rate=learning_rate,
        warmup=0.0,
        muonc_anneal_decay=_LAMBDA,
        muonc_anneal_decay_lr_steps=learning_rate * num_train_steps,
    )
    _, base_updates = _zero_gradient_update(base, num_train_steps)
    _, scaled_updates = _zero_gradient_update(scaled, num_train_steps)
    for path in _PATHS:
        np.testing.assert_allclose(_leaf(scaled_updates, path), _leaf(base_updates, path), rtol=1e-6)
