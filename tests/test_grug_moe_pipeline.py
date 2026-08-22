# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe.grug_moe_pipeline import (
    GrugMoePipelineConfig,
    batches_for_pipeline,
    make_pipeline_state,
    merge_stages,
    microbatched_staged_loss,
    split_transformer,
    staged_loss,
)
from experiments.grug.moe.model import GrugModelConfig, Transformer


def _tiny_model() -> tuple[Mesh, Transformer]:
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        sliding_window=4,
        router_z_loss_coef=0.1,
        attention_implementation="reference",
        moe_implementation="scatter",
    )
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(0))
    return mesh, model


def _batch() -> GrugLmExample:
    return GrugLmExample(
        tokens=jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32),
        loss_weight=jnp.ones((2, 4), dtype=jnp.float32),
    )


def _assert_trees_close(actual, expected) -> None:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-5, atol=1e-5)


def test_split_transformer_assigns_only_boundary_parameters_to_boundary_stages():
    _, model = _tiny_model()

    first, last = split_transformer(model, 2)

    assert first.token_embed is model.token_embed
    assert first.output_proj is None
    assert last.token_embed is None
    assert last.output_proj is model.output_proj
    assert (first.start_layer, first.end_layer) == (0, 1)
    assert (last.start_layer, last.end_layer) == (1, 2)
    _assert_trees_close(merge_stages((first, last)), model)


def test_staged_loss_and_gradients_match_the_unsplit_model():
    mesh, model = _tiny_model()
    batch = _batch()

    def ordinary_loss(params):
        return params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=0.01,
        )

    def pipeline_loss(params):
        return staged_loss(split_transformer(params, 2), batch, logsumexp_weight=0.01)

    with jax.set_mesh(mesh):
        ordinary_value, ordinary_grads = jax.value_and_grad(ordinary_loss)(model)
        pipeline_value, pipeline_grads = jax.value_and_grad(pipeline_loss)(model)

    np.testing.assert_allclose(pipeline_value, ordinary_value, rtol=1e-5, atol=1e-5)
    _assert_trees_close(pipeline_grads, ordinary_grads)


def test_pipeline_accepts_fewer_microbatches_than_stages():
    _, model = _tiny_model()
    config = GrugMoePipelineConfig(stages=2, microbatches=1)

    batches = batches_for_pipeline(_batch(), config)
    state = make_pipeline_state(model, optax.sgd(0.1), num_stages=config.stages)

    assert len(batches) == 1
    assert len(batches[0]) == 2
    assert batches[0][0].tokens is not batches[0][1].tokens
    np.testing.assert_array_equal(batches[0][0].tokens, batches[0][1].tokens)
    assert len(state.params) == 2


def test_microbatched_loss_matches_full_batch_with_uneven_loss_weights():
    mesh, model = _tiny_model()
    batch = dataclasses.replace(
        _batch(),
        loss_weight=jnp.array([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=jnp.float32),
    )

    with jax.set_mesh(mesh):
        ordinary_loss = model.next_token_loss(batch.tokens, batch.loss_weight, mask=batch.attn_mask)
        pipeline_loss = microbatched_staged_loss(split_transformer(model, 2), batch, num_microbatches=2)

    np.testing.assert_allclose(pipeline_loss, ordinary_loss, rtol=1e-5, atol=1e-5)
