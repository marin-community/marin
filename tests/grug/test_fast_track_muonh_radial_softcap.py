# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``muonh_radial_softcap``: MuonH spheres learn a log-radius bounded by the cap, while the matrices
behind a scale-removing norm keep plain MuonH."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_CAP = 2.0
# A free 2-D matrix and a 3-D layer stack (one sphere per layer) beside the frozen q projection, MLA KV
# down-projection and MoE latent down-projection.
_SHAPES = {
    "stacked_blocks": {
        "stacked": {
            "attn": {"w_q": (2, 8, 8), "w_dkv": (2, 8, 4), "w_o": (2, 8, 8)},
            "mlp": {"w_latent_down": (2, 8, 4)},
        }
    },
    "gated": {"w_up": (4, 8)},
}


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


def _train(config, params, grad_sign, num_steps, state=None):
    """``num_steps`` steps on the loss ``grad_sign * |W|^2 / 2`` of every leaf (gradient ``grad_sign * W``)."""
    optimizer = config.build(num_train_steps=1000)
    state = optimizer.init(params) if state is None else state
    for _ in range(num_steps):
        grads = jax.tree.map(lambda p: grad_sign * p, params)
        updates, state = optimizer.update(grads, state, params)
        params = jax.tree.map(jnp.add, params, updates)
    return params, state


def test_radius_grows_to_the_cap_and_returns_while_frozen_matrices_keep_muonh():
    params = _random_tree(jax.random.key(0))
    off = GrugMoeMuonHConfig(learning_rate=0.2, warmup=0.0, lr_schedule="constant")
    on = dataclasses.replace(off, muonh_radial_softcap=_CAP)

    labels = on.create_mask(params)
    attn = labels["stacked_blocks"]["stacked"]["attn"]
    assert (attn["w_q"], attn["w_dkv"], attn["w_o"]) == ("muonh", "muonh", "muonh_radial")
    assert labels["stacked_blocks"]["stacked"]["mlp"]["w_latent_down"] == "muonh"
    assert labels["gated"]["w_up"] == "muonh_radial"

    # A negative gradient along W pulls every radius outward.
    grown_on, state = _train(on, params, grad_sign=-1.0, num_steps=60)
    grown_off, _ = _train(off, params, grad_sign=-1.0, num_steps=60)
    for path in (("stacked_blocks", "stacked", "attn", "w_o"), ("gated", "w_up")):
        start, end = (_sphere_norms(_leaf(tree, path)) for tree in (params, grown_on))
        ratio = end / start
        assert np.all(ratio > 2.5), ratio
        assert np.all(ratio <= math.exp(_CAP) * (1 + 1e-5)), ratio
    for path in (
        ("stacked_blocks", "stacked", "attn", "w_q"),
        ("stacked_blocks", "stacked", "attn", "w_dkv"),
        ("stacked_blocks", "stacked", "mlp", "w_latent_down"),
    ):
        np.testing.assert_array_equal(_leaf(grown_on, path), _leaf(grown_off, path))

    # When the pull reverses, the radius shrinks back at once.
    w_o = ("stacked_blocks", "stacked", "attn", "w_o")
    shrunk, _ = _train(on, grown_on, grad_sign=1.0, num_steps=20, state=state)
    assert np.all(_sphere_norms(_leaf(shrunk, w_o)) < _sphere_norms(_leaf(grown_on, w_o)))
