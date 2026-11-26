# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_decay_floor_frac``: the MuonC matrix decay is ``lambda ((1 - f) lr_m / lr_max + f)``, so
a fraction ``f`` of it stays on as the matrix LR anneals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.optimizer import KDA_BETA_LR_MULT, LATE_ANNEAL_FRAC, GrugMoeMuonHConfig

_LAMBDA = 0.2
_FRAC = 0.05
_LR = 0.1
_NUM_TRAIN_STEPS = 40
_SHAPES = {
    "stacked_blocks": {"stacked": {"attn": {"w_o": (2, 8, 8)}}},
    "kda_blocks": {"stacked": {"attn": {"w_beta": (2, 8, 8)}}},
}
_W_O = ("stacked_blocks", "stacked", "attn", "w_o")
_W_BETA = ("kda_blocks", "stacked", "attn", "w_beta")


def _params():
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(jax.random.key(0), len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _zero_gradient_update(config, step):
    """The update after ``step`` zero-gradient steps: a pure matrix decay ``lr_m lambda_t W``."""
    params = _params()
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    state = optimizer.init(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    for _ in range(step):
        _, state = optimizer.update(zeros, state, params)
    updates, _ = optimizer.update(zeros, state, params)
    return params, updates


# Early, and late in the matrix anneal (t0 = round(0.7937 * 40) = 32).
@pytest.mark.parametrize("step", [3, 38])
def test_matrix_decay_keeps_a_constant_fraction(step):
    config = GrugMoeMuonHConfig(
        learning_rate=_LR,
        warmup=0.0,
        muonc_anneal_decay=_LAMBDA,
        muonc_anneal_decay_floor_frac=_FRAC,
    )
    schedule = config.lr_scheduler(_NUM_TRAIN_STEPS)
    lr = float(schedule(step))
    late_anneal_start = round(LATE_ANNEAL_FRAC * _NUM_TRAIN_STEPS)
    matrix_lr = lr * lr / float(schedule(late_anneal_start)) if step >= late_anneal_start else lr
    decay = _LAMBDA * ((1 - _FRAC) * matrix_lr / _LR + _FRAC)
    params, updates = _zero_gradient_update(config, step)
    np.testing.assert_allclose(_leaf(updates, _W_O), -matrix_lr * decay * _leaf(params, _W_O), rtol=1e-4)
    np.testing.assert_allclose(
        _leaf(updates, _W_BETA), -KDA_BETA_LR_MULT * matrix_lr * decay * _leaf(params, _W_BETA), rtol=1e-4
    )
