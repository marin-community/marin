# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep.model import QbEstimator, Transformer
from experiments.grug.moe_hero_pipeline.pipeline import (
    GrugMoePipelineStage,
    QbBiasMode,
    _copy_array_to_host,
    split_transformer,
    updated_pipeline_state,
)
from experiments.grug.moe_hero_pipeline.tiny_hero import packed_batch, pipeline_state, tiny_hero


@pytest.mark.parametrize("qb_bias_mode", [QbBiasMode.ADAPTIVE, QbBiasMode.FROZEN])
def test_pipeline_update_trains_weights_with_adaptive_or_frozen_router_bias(qb_bias_mode):
    mesh, model = tiny_hero(QbEstimator.HIST, devices=jax.local_devices())
    optimizer = optax.sgd(0.1, momentum=0.5)
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    with jax.set_mesh(mesh):
        state = pipeline_state(model, optimizer, (2, 3))
        betas = state.pending_qb_betas
        grads = eqx.filter_grad(lambda values: sum(jnp.sum(value**2) for value in jax.tree.leaves(values)))(
            state.trainable_params
        )
        measured_beta_sums = tuple(jnp.full_like(beta, 12.0) for beta in betas)
        updated = updated_pipeline_state(
            state, grads, measured_beta_sums, optimizer, policy, microbatches=4, qb_bias_mode=qb_bias_mode
        )
        second = updated_pipeline_state(
            updated, grads, measured_beta_sums, optimizer, policy, microbatches=4, qb_bias_mode=qb_bias_mode
        )

    for original, actual, second_actual in zip(
        jax.tree.leaves(state.trainable_params),
        jax.tree.leaves(updated.trainable_params),
        jax.tree.leaves(second.trainable_params),
        strict=True,
    ):
        # d(sum(p**2))/dp = 2p. Momentum carries the first gradient into step two.
        np.testing.assert_allclose(actual, 0.8 * original, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(second_actual, 0.5 * original, rtol=1e-5, atol=1e-5)
        assert actual.dtype == jnp.float32
    for original, actual, second_actual in zip(betas, updated.pending_qb_betas, second.pending_qb_betas, strict=True):
        expected = original if qb_bias_mode == QbBiasMode.FROZEN else jnp.full_like(original, 3.0)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(second_actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pipeline_embedding_recompute_preserves_values_and_gradients(dtype):
    mesh, model = tiny_hero(QbEstimator.HIST, devices=jax.local_devices())
    tokens = packed_batch().tokens
    with jax.set_mesh(mesh):
        stage = split_transformer(model, 2, layer_counts=(2, 3))[0]
        stage = jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, stage)

        def reference(params):
            hidden = params.token_embed[tokens]
            return params.embed_gated_norm(params.embed_norm(hidden))

        expected, reference_pullback = jax.vjp(reference, stage)
        actual, pullback = jax.vjp(lambda params: params.embed(tokens), stage)
        cotangent = jax.random.normal(jax.random.key(73), actual.shape, dtype=dtype)
        (expected_grads,) = reference_pullback(jax.device_put(cotangent, expected.sharding))
        (actual_grads,) = pullback(jax.device_put(cotangent, actual.sharding))

    np.testing.assert_array_equal(actual, expected)
    for actual_grad, expected_grad in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads), strict=True):
        np.testing.assert_allclose(
            actual_grad.astype(jnp.float32), expected_grad.astype(jnp.float32), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("qb_estimator", [QbEstimator.TOPK, QbEstimator.HIST])
def test_hero_pipeline_preserves_hidden_states_and_router_statistics(qb_estimator):
    mesh, model = tiny_hero(qb_estimator, devices=jax.local_devices())
    batch = packed_batch()
    # Compile the scan and whole stages: executing shard_map primitives
    # separately exceeds the CPU compilation budget.
    with jax.set_mesh(mesh):
        expected_hidden, expected_metrics = eqx.filter_jit(Transformer.__call__)(model, batch.tokens, batch.attn_mask)
        stages = split_transformer(model, 2, layer_counts=(2, 3))
        hidden = stages[0].embed(batch.tokens)
        stage_metrics = []
        run_blocks = eqx.filter_jit(GrugMoePipelineStage.run_blocks)
        for stage in stages:
            hidden, metrics = run_blocks(stage, hidden, batch.attn_mask)
            stage_metrics.append(metrics)
        hidden = stages[-1].finish(hidden)

    np.testing.assert_allclose(hidden, expected_hidden, rtol=1e-5, atol=1e-5)
    for name in ("routing_counts_per_layer", "router_z_loss_per_layer", "qb_beta_per_layer"):
        actual = jnp.concatenate([metrics[name] for metrics in stage_metrics])
        np.testing.assert_allclose(actual, expected_metrics[name], rtol=1e-5, atol=1e-5)


def test_hero_pipeline_applies_log_scaling():
    mesh, off = tiny_hero(QbEstimator.TOPK, devices=jax.local_devices())
    _, model = tiny_hero(
        QbEstimator.TOPK,
        devices=jax.local_devices(),
        log_scaling_n_floor=2,
        log_scaling_alpha=0.1,
        log_scaling_power=1.0,
    )
    batch = packed_batch()
    with jax.set_mesh(mesh):
        expected_hidden, _ = eqx.filter_jit(Transformer.__call__)(model, batch.tokens, batch.attn_mask)
        off_hidden, _ = eqx.filter_jit(Transformer.__call__)(off, batch.tokens, batch.attn_mask)
        stages = split_transformer(model, 2, layer_counts=(2, 3))
        hidden = stages[0].embed(batch.tokens)
        run_blocks = eqx.filter_jit(GrugMoePipelineStage.run_blocks)
        for stage in stages:
            hidden, _ = run_blocks(stage, hidden, batch.attn_mask)
        hidden = stages[-1].finish(hidden)

    np.testing.assert_allclose(hidden, expected_hidden, rtol=1e-5, atol=1e-5)
    assert np.max(np.abs(np.asarray(expected_hidden) - np.asarray(off_hidden))) > 1e-4


def test_parked_host_copy_survives_device_alias_deletion_and_restores_bits():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("expert",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, P("expert"))
    values = np.array([0.0, -0.0, 1.25, -3.5, np.inf, np.nan], dtype=np.float32)
    original = jax.device_put(values, sharding)
    alias = original
    parked = _copy_array_to_host(original)
    parked.block_until_ready()
    assert parked.sharding.memory_kind == "pinned_host"
    original.delete()
    assert alias.is_deleted()
    np.testing.assert_array_equal(np.asarray(parked).view(np.uint32), values.view(np.uint32))
    restored = jax.device_put(parked, sharding, may_alias=False)
    restored.block_until_ready()
    parked.delete()
    assert restored.sharding == sharding
    np.testing.assert_array_equal(np.asarray(restored).view(np.uint32), values.view(np.uint32))
