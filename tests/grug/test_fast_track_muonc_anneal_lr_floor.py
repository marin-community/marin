# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonc_anneal_lr_floor``: from ``t0`` the MuonC matrix LR is ``max(lr_t s(t) / s(t0), min_lr_ratio
lr_max)``, and the matrix decay ``lr_m lambda_t W`` with ``lambda_t = lambda lr_m / lr_max`` follows the clamped LR."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import LATE_ANNEAL_FRAC, GrugMoeMuonHConfig

_LAMBDA = 0.2
_LEARNING_RATE = 0.1
_MIN_LR_RATIO = 0.1
_NUM_TRAIN_STEPS = 20
_SHAPES = {"stacked_blocks": {"stacked": {"attn": {"w_o": (2, 8, 8)}}}, "gated": {"w_up": (4, 8)}}
_PATHS = (("stacked_blocks", "stacked", "attn", "w_o"), ("gated", "w_up"))


def _params():
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(jax.random.key(0), len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _leaf(tree, path):
    for key in path:
        tree = tree[key]
    return np.asarray(tree)


def _config(floor: bool) -> GrugMoeMuonHConfig:
    return GrugMoeMuonHConfig(
        learning_rate=_LEARNING_RATE,
        min_lr_ratio=_MIN_LR_RATIO,
        warmup=0.0,
        muonc_anneal_decay=_LAMBDA,
        muonc_anneal_lr_floor=floor,
    )


def _zero_gradient_update(config, step):
    """The update at ``step`` after ``step`` zero-gradient steps: a pure matrix decay ``lr_m lambda_t W``."""
    params = _params()
    optimizer = config.build(num_train_steps=_NUM_TRAIN_STEPS)
    state = optimizer.init(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    for _ in range(step):
        _, state = optimizer.update(zeros, state, params)
    updates, _ = optimizer.update(zeros, state, params)
    return params, updates


def test_late_matrix_decay_uses_the_clamped_lr():
    schedule = _config(floor=True).lr_scheduler(_NUM_TRAIN_STEPS)
    late_anneal_start = round(LATE_ANNEAL_FRAC * _NUM_TRAIN_STEPS)
    step = _NUM_TRAIN_STEPS - 1
    lr = float(schedule(step))
    annealed = lr * lr / float(schedule(late_anneal_start))
    floor = _MIN_LR_RATIO * _LEARNING_RATE
    assert annealed < floor  # the clamp binds at this step
    params, updates = _zero_gradient_update(_config(floor=True), step)
    for path in _PATHS:
        expected = -floor * (_LAMBDA * floor / _LEARNING_RATE) * _leaf(params, path)
        np.testing.assert_allclose(_leaf(updates, path), expected, rtol=1e-4)


def test_updates_before_the_floor_binds_are_unchanged():
    late_anneal_start = round(LATE_ANNEAL_FRAC * _NUM_TRAIN_STEPS)
    for step in (late_anneal_start - 2, late_anneal_start):
        _, base = _zero_gradient_update(_config(floor=False), step)
        _, floored = _zero_gradient_update(_config(floor=True), step)
        for path in _PATHS:
            np.testing.assert_allclose(_leaf(floored, path), _leaf(base, path), rtol=1e-6)
