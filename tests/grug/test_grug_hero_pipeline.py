# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_hero_ep.model import GrugModelConfig, QbEstimator, Transformer
from experiments.grug.moe_hero_pipeline.pipeline import _copy_array_to_host, split_transformer


def _tiny_hero(qb_estimator: QbEstimator) -> tuple[Mesh, Transformer]:
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        latent_dim=4,
        shared_expert_intermediate_dim=4,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=5,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=4,
        max_seq_len=6,
        sliding_window=2,
        global_every=3,
        sconv=True,
        sconv_kernel=3,
        qb_estimator=qb_estimator,
        qb_hist_bins=32,
        attention_implementation="reference",
        moe_implementation="scatter",
        initializer_std=0.2,
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.key(42))
        # Nonzero delayed convolution taps exercise packed-document boundaries.
        model = eqx.tree_at(
            lambda current: (
                current.stacked_blocks.stacked.attn.sconv_k.weight,
                current.stacked_blocks.stacked.sconv_attn.weight,
                current.stacked_blocks.stacked.sconv_mlp.weight,
            ),
            model,
            replace_fn=lambda weights: weights.at[:, 1:, :].set(0.1),
        )
    return mesh, model


def _packed_batch() -> GrugLmExample:
    segments = jnp.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    return GrugLmExample(
        tokens=jnp.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=jnp.int32),
        loss_weight=jnp.array([[1.0, 0.5, 0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.5, 0.0]]),
        attn_mask=AttentionMask.causal().with_segment_ids(segments),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pipeline_embedding_recompute_preserves_values_and_gradients(dtype):
    mesh, model = _tiny_hero(QbEstimator.HIST)
    tokens = _packed_batch().tokens
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
    mesh, model = _tiny_hero(qb_estimator)
    batch = _packed_batch()
    with jax.set_mesh(mesh):
        expected_hidden, expected_metrics = model(batch.tokens, batch.attn_mask)
        stages = split_transformer(model, 2, layer_counts=(2, 3))
        hidden = stages[0].embed(batch.tokens)
        stage_metrics = []
        for stage in stages:
            hidden, metrics = stage.run_blocks(hidden, batch.attn_mask)
            stage_metrics.append(metrics)
        hidden = stages[-1].finish(hidden)

    np.testing.assert_allclose(hidden, expected_hidden, rtol=1e-5, atol=1e-5)
    for name in ("routing_counts_per_layer", "router_z_loss_per_layer", "qb_beta_per_layer"):
        actual = jnp.concatenate([metrics[name] for metrics in stage_metrics])
        np.testing.assert_allclose(actual, expected_metrics[name], rtol=1e-5, atol=1e-5)


# Compiling both gradient paths can exceed the default minute on shared CI CPUs.
@pytest.mark.timeout(180)
@pytest.mark.parametrize("remat_mode", ["recompute_all", "offload_carry"])
def test_hero_pipeline_loss_and_gradients_match_unsplit_model(remat_mode):
    mesh, model = _tiny_hero(QbEstimator.HIST)
    batch = _packed_batch()

    def ordinary_loss(params):
        return params.next_token_loss(batch.tokens, batch.loss_weight, mask=batch.attn_mask, logsumexp_weight=0.01)

    def pipeline_loss(params):
        params = dataclasses.replace(params, config=dataclasses.replace(params.config, remat_mode=remat_mode))
        stages = split_transformer(params, 2, layer_counts=(2, 3))
        hidden = stages[0].embed(batch.tokens)
        for stage in stages:
            hidden, _ = stage.run_blocks(hidden, batch.attn_mask)
        return stages[-1].cross_entropy_loss(
            stages[-1].finish(hidden), batch.tokens, batch.loss_weight, logsumexp_weight=0.01
        )

    with jax.set_mesh(mesh):
        expected_loss, expected_grads = jax.value_and_grad(ordinary_loss)(model)
        actual_loss, pullback = jax.vjp(pipeline_loss, model)
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
