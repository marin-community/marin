# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe_hero_ep.model import QbEstimator
from experiments.grug.moe_hero_pipeline.pipeline import (
    split_transformer,
)
from experiments.grug.moe_hero_pipeline.tiny_hero import packed_batch, tiny_hero


@pytest.mark.parametrize("remat_mode", ["recompute_all", "offload_carry"])
def test_hero_pipeline_loss_and_gradients_match_unsplit_model(remat_mode):
    # Two layers cover local and global attention across a stage boundary
    # without compiling the two gradient paths for five layers per mode.
    mesh, model = tiny_hero(QbEstimator.HIST, num_layers=2, devices=jax.local_devices())
    batch = packed_batch()

    def ordinary_loss(params):
        return params.next_token_loss(batch.tokens, batch.loss_weight, mask=batch.attn_mask, logsumexp_weight=0.01)

    def pipeline_loss(params):
        params = dataclasses.replace(params, config=dataclasses.replace(params.config, remat_mode=remat_mode))
        stages = split_transformer(params, 2, layer_counts=(1, 1))
        hidden = stages[0].embed(batch.tokens)
        for stage in stages:
            hidden, _ = stage.run_blocks(hidden, batch.attn_mask)
        return stages[-1].cross_entropy_loss(
            stages[-1].finish(hidden), batch.tokens, batch.loss_weight, logsumexp_weight=0.01
        )

    with jax.set_mesh(mesh):
        expected_loss, expected_grads = jax.jit(jax.value_and_grad(ordinary_loss))(model)
        actual_loss, pullback = jax.vjp(jax.jit(pipeline_loss), model)
        if remat_mode == "offload_carry":
            host_residuals = [
                value
                for value in jax.tree.leaves(pullback)
                if isinstance(value, jax.Array) and value.sharding.memory_kind == "pinned_host"
            ]
            # Every layer input, including the stage boundary, must live on host
            # between forward and backward rather than merely carry a remat name.
            assert len(host_residuals) == model.config.num_layers
            assert all(value.shape == (*batch.tokens.shape, model.config.hidden_dim) for value in host_residuals)
        (actual_grads,) = pullback(jnp.ones_like(actual_loss))

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)
    for actual, expected in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
