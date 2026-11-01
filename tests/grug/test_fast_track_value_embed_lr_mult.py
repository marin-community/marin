# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track value-embedding LR multiplier: the ``value_embed`` tables step at ``value_embed_lr_mult`` x the Adam LR
while their mixing weights keep the Adam LR."""

import dataclasses

import jax
import numpy as np

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

TOTAL_STEPS = 100
LR_MULT = 0.3


def _params():
    keys = jax.random.split(jax.random.key(0), 2)
    return {"attn": {"value_embed": jax.random.normal(keys[0], (32, 8)), "ve_gate": jax.random.normal(keys[1], (8, 2))}}


def _grads(step: int):
    keys = jax.random.split(jax.random.key(step + 1), 2)
    return {"attn": {"value_embed": jax.random.normal(keys[0], (32, 8)), "ve_gate": jax.random.normal(keys[1], (8, 2))}}


def _updates(config: GrugMoeMuonHConfig, num_steps: int):
    optimizer = config.build(TOTAL_STEPS)
    params = _params()
    state = optimizer.init(params)
    updates = []
    for step in range(num_steps):
        update, state = optimizer.update(_grads(step), state, params)
        updates.append(jax.tree.map(lambda u: np.asarray(u, np.float32), update))
    return updates


def test_value_embed_lr_mult_scales_only_the_table():
    base = GrugMoeMuonHConfig(adam_lr=1e-3, warmup=0.0)
    scaled = dataclasses.replace(base, value_embed_lr_mult=LR_MULT)

    assert scaled.create_mask(_params())["attn"]["value_embed"] == "value_embed"
    assert base.create_mask(_params())["attn"]["value_embed"] == "adam"

    for base_update, scaled_update in zip(_updates(base, 3), _updates(scaled, 3), strict=True):
        np.testing.assert_allclose(
            scaled_update["attn"]["value_embed"], LR_MULT * base_update["attn"]["value_embed"], rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            scaled_update["attn"]["ve_gate"], base_update["attn"]["ve_gate"], rtol=1e-6, atol=1e-9
        )
