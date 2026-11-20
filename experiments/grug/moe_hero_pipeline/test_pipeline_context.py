# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.testing.devices import use_test_devices

from experiments.grug.moe_hero_ep.model import QbEstimator
from experiments.grug.moe_hero_pipeline.pipeline import (
    split_transformer,
)
from experiments.grug.moe_hero_pipeline.tiny_hero import packed_batch, tiny_hero


@pytest.mark.timeout(300)
@pytest.mark.multidevice
def test_context_pipeline_loss_and_gradients_match_unsharded_model():
    with use_test_devices(2) as devices:
        results = []
        for context_size in (1, 2):
            mesh, model = tiny_hero(QbEstimator.HIST, context_size, devices=devices)
            model = dataclasses.replace(model, config=dataclasses.replace(model.config, remat_mode="offload_carry"))
            batch = packed_batch()
            spec = NamedSharding(mesh, P(None, "context"))
            with jax.set_mesh(mesh):
                tokens = jax.device_put(batch.tokens, spec)
                weights = jax.device_put(batch.loss_weight, spec)
                mask = batch.attn_mask

                def loss(params, *, context_size=context_size, tokens=tokens, weights=weights, mask=mask):
                    if context_size == 1:
                        return params.next_token_loss(tokens, weights, mask=mask, logsumexp_weight=0.01)
                    stages = split_transformer(params, 2, layer_counts=(2, 3))
                    hidden = stages[0].embed(tokens)
                    for stage in stages:
                        hidden, _ = stage.run_blocks(hidden, mask)
                    assert jax.typeof(hidden).sharding.spec[1] == "context"
                    return stages[-1].cross_entropy_loss(
                        stages[-1].finish(hidden), tokens, weights, logsumexp_weight=0.01
                    )

                result = eqx.filter_jit(jax.value_and_grad(loss))(model)
                results.append([np.asarray(x) for x in jax.tree.leaves(result)])
        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
