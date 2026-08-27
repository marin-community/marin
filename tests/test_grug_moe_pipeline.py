# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe.benchmark_grug_moe_pipeline import _validate_local_mesh
from experiments.grug.moe.grug_moe_pipeline import (
    GrugMoePipelineConfig,
    automatic_stage_to_mpmd_indices,
    batches_for_pipeline,
    make_automatic_pipeline_state,
    merge_stages,
    microbatched_staged_loss,
    split_transformer,
    stage_forward_with_pullback,
    stage_pullback_input_gradient,
    stage_pullback_weight_gradient,
    staged_loss,
)
from experiments.grug.moe.model import GrugModelConfig, Transformer


def _tiny_model(*, num_layers: int = 2) -> tuple[Mesh, Transformer]:
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=num_layers,
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


def test_split_and_merge_transformer_round_trip_with_uneven_stages():
    _, model = _tiny_model(num_layers=4)

    stages = split_transformer(model, 2, layer_counts=(3, 1))

    _assert_trees_close(merge_stages(stages), model)


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


def test_reusable_stage_pullback_matches_combined_backward():
    mesh, model = _tiny_model(num_layers=4)
    stage = split_transformer(model, 2)[1]
    batch = _batch()
    activation_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None))
    hidden = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(1), (2, 4, model.config.hidden_dim)),
        activation_sharding,
    )
    hidden_cotangent = jax.device_put(
        jax.random.normal(jax.random.PRNGKey(2), hidden.shape),
        activation_sharding,
    )
    qb_betas = jnp.zeros((len(stage.blocks), model.config.num_experts), dtype=jnp.float32)
    mp_policy = jmp.get_policy("f32")

    def projected_loss(params, stage_hidden):
        params = mp_policy.cast_to_compute(params)
        output, metrics = params.run_blocks(stage_hidden, batch.attn_mask)
        activation_term = jnp.sum(output.astype(jnp.float32) * hidden_cotangent.astype(jnp.float32))
        return activation_term + params.local_router_loss(metrics)

    with jax.set_mesh(mesh):
        expected_grads, expected_input_gradient = jax.grad(projected_loss, argnums=(0, 1))(stage, hidden)
        _, _, _, pullback = stage_forward_with_pullback(
            stage,
            qb_betas,
            hidden,
            batch,
            mp_policy,
            router_loss_scale=1.0,
        )
        input_gradient = jax.jit(stage_pullback_input_gradient)(pullback, hidden_cotangent)
        grads = jax.jit(stage_pullback_weight_gradient)(pullback, hidden_cotangent)

        np.testing.assert_allclose(input_gradient, expected_input_gradient, rtol=1e-5, atol=1e-5)
        _assert_trees_close(grads, expected_grads)


def test_pipeline_accepts_fewer_microbatches_than_stages():
    config = GrugMoePipelineConfig(stages=2, microbatches=1)

    batches = batches_for_pipeline(_batch(), config)

    assert len(batches) == 1
    assert len(batches[0]) == 2
    assert batches[0][0].tokens is not batches[0][1].tokens
    np.testing.assert_array_equal(batches[0][0].tokens, batches[0][1].tokens)


def test_dualpipe_v_maps_two_logical_stages_to_each_physical_rank():
    config = GrugMoePipelineConfig(stages=4, physical_stages=2, microbatches=4)

    assert automatic_stage_to_mpmd_indices(config, "dualpipe_v") == (0, 1, 1, 0)


def test_automatic_pipeline_excludes_qb_bias_from_differentiated_parameters():
    _, model = _tiny_model(num_layers=4)

    state, _ = make_automatic_pipeline_state(model, optax.sgd(0.1), num_stages=2)

    for trainable_stage in state.trainable_params:
        for trainable_block in trainable_stage.blocks:
            assert trainable_block.mlp.router_bias is None


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


def test_pipeline_mesh_validation_uses_full_stage_shard_count():
    # Regression: a stage may contain both FSDP and expert axes, so its device
    # count can exceed the expert-axis size.
    _validate_local_mesh(
        local_device_count=8,
        expert_axis_size=4,
        batch_size=64,
        microbatches=4,
    )

    with pytest.raises(ValueError):
        _validate_local_mesh(
            local_device_count=8,
            expert_axis_size=4,
            batch_size=16,
            microbatches=4,
        )
