# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``embed_lr_mult``: the input embedding table moves to its own copy of the ``adam`` group at a multiple
of the Adam LR, and every other leaf's update is unchanged."""

import dataclasses

import jax
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_MULT = 4.0
# The input table beside other ``adam`` leaves: a second table, value embeddings, a norm gain and a decayed gate.
_SHAPES = {
    "token_embed": (16, 8),
    "token_embed_bytes": (3, 8),
    "value_embed": (16, 8),
    "norm": {"weight": (8,)},
    "attn": {"attn_gate": (8, 2)},
}


def _random_tree(key):
    leaves, treedef = jax.tree.flatten(_SHAPES, is_leaf=lambda x: isinstance(x, tuple))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(treedef, [jax.random.normal(k, s) for k, s in zip(keys, leaves, strict=True)])


def test_token_embed_steps_at_the_multiple_and_nothing_else_moves():
    params, grads = _random_tree(jax.random.key(0)), _random_tree(jax.random.key(1))
    # No warmup, so the first step runs at the full Adam LR.
    off = GrugMoeMuonHConfig(warmup=0.0)
    on = dataclasses.replace(off, embed_lr_mult=_MULT)

    def two_updates(config):
        optimizer = config.build(num_train_steps=100)
        first, state = optimizer.update(grads, optimizer.init(params), params)
        return first, optimizer.update(grads, state, params)[0]

    labels_on = on.create_mask(params)
    assert labels_on["token_embed"] == "embed_lr"
    assert {**labels_on, "token_embed": "adam"} == off.create_mask(params)
    for updates_off, updates_on in zip(two_updates(off), two_updates(on), strict=True):
        assert np.linalg.norm(updates_off["token_embed"]) > 0
        np.testing.assert_allclose(updates_on["token_embed"], _MULT * updates_off["token_embed"], rtol=1e-6)
        rest_on = {**updates_on, "token_embed": updates_off["token_embed"]}
        for got, want in zip(jax.tree.leaves(rest_on), jax.tree.leaves(updates_off), strict=True):
            np.testing.assert_array_equal(got, want)
