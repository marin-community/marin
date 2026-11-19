# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonh_init_removal``: every MuonH sphere loses its initial weights over the middle third of training
and keeps its radius."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.grugmuon_stacked import _grug_scale_with_muon
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_NUM_STEPS = 30
# A 2-D matrix and a 3-D layer stack (one sphere per layer).
_SHAPES = {"w": (8, 16), "stacked": {"w_o": (2, 8, 16)}}


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def _spheres(x):
    x = np.asarray(x, np.float64)
    return [x] if x.ndim == 2 else list(x)


def _init_coefficients(params, init, grads):
    """Per sphere, the coefficient of the init in the least-squares fit ``W = c0 W0 + c1 O``.

    A constant gradient gives a constant Muon direction ``O``, so every step stays in that plane."""
    muon = _grug_scale_with_muon()
    directions, _ = muon.update(grads, muon.init(grads), grads)
    coefficients = []
    for leaf, leaf_init, leaf_direction in zip(*(jax.tree.leaves(t) for t in (params, init, directions)), strict=True):
        for w, w0, o in zip(_spheres(leaf), _spheres(leaf_init), _spheres(leaf_direction), strict=True):
            fit, *_ = np.linalg.lstsq(np.stack([w0.ravel(), o.ravel()], 1), w.ravel(), rcond=None)
            coefficients.append(fit[0])
    return np.array(coefficients)


def _trajectory(config, params, grads):
    optimizer = config.build(num_train_steps=_NUM_STEPS)
    state = optimizer.init(params)
    trajectory = [params]
    for _ in range(_NUM_STEPS):
        updates, state = optimizer.update(grads, state, params)
        params = jax.tree.map(jnp.add, params, updates)
        trajectory.append(params)
    return trajectory


def test_init_is_removed_over_the_middle_third_on_the_sphere():
    init = _random_tree(jax.random.key(0))
    grads = _random_tree(jax.random.key(1))
    off = GrugMoeMuonHConfig(learning_rate=0.02, warmup=0.0, lr_schedule="constant")
    on = dataclasses.replace(off, muonh_init_removal=True)
    assert on.create_mask(init) == {"w": "muonh", "stacked": {"w_o": "muonh"}}

    on_path = _trajectory(on, init, grads)
    off_path = _trajectory(off, init, grads)
    window_start, window_end = _NUM_STEPS // 3, 2 * _NUM_STEPS // 3

    # Before the window the optimizer is plain MuonH.
    for a, b in zip(jax.tree.leaves(on_path[window_start]), jax.tree.leaves(off_path[window_start]), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    # The init's share falls through the window and is gone at its end and after it; without the flag it stays.
    start, middle, end, last = (
        _init_coefficients(on_path[step], init, grads) for step in (window_start, 15, window_end, _NUM_STEPS)
    )
    assert np.all(start > 0.5), start
    assert np.all((middle > 0.05) & (middle < start)), middle
    # Newton-Schulz runs in bfloat16, so the direction moves slightly between steps.
    np.testing.assert_allclose(end, 0.0, atol=1e-2)
    np.testing.assert_allclose(last, 0.0, atol=1e-2)
    assert np.all(_init_coefficients(off_path[_NUM_STEPS], init, grads) > 0.3)

    # Every sphere keeps its radius.
    for leaf, leaf_init in zip(jax.tree.leaves(on_path[_NUM_STEPS]), jax.tree.leaves(init), strict=True):
        for w, w0 in zip(_spheres(leaf), _spheres(leaf_init), strict=True):
            np.testing.assert_allclose(np.linalg.norm(w), np.linalg.norm(w0), rtol=1e-5)
