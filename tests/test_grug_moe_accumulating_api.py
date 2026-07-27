# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe.train import (
    _stage_expert_gradient_accumulators,
    _stage_with_expert_gradients,
    _stage_without_expert_gradients,
)


def _fake_triton_ragged_dot(monkeypatch) -> None:
    ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
        *,
        output_dtype=None,
    ):
        output = jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )
        return output if output_dtype is None else output.astype(output_dtype)

    def fake_accumulating_pallas_call(lhs, rhs, group_sizes, accumulator, accumulation_scale):
        fresh_gradient = fake_triton_pallas_call(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_module._DRHS_DIM_NUMS,
            output_dtype=jnp.float32,
        )
        return fresh_gradient + accumulation_scale * accumulator

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)
    monkeypatch.setattr(
        ragged_dot_module,
        "_triton_ragged_contracting_dim_accumulating_pallas_call",
        fake_accumulating_pallas_call,
    )


def _bfloat16_arrays(tree):
    return jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
        tree,
    )


def test_pipeline_stage_accumulating_weight_gradient_threads_every_block_token(monkeypatch):
    _fake_triton_ragged_dot(monkeypatch)
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        sliding_window=4,
        attention_implementation="reference",
        moe_implementation="ring",
        loss_implementation="reference",
        remat_mode="save_moe",
    )
    with jax.set_mesh(mesh):
        stage = Transformer.init(config, key=jax.random.key(0)).split_for_pipeline(1)[0]
        stage = _bfloat16_arrays(stage)

    hidden = jax.random.normal(jax.random.key(1), (1, 4, config.hidden_dim), dtype=jnp.bfloat16)
    output_cotangent = jax.random.normal(jax.random.key(2), hidden.shape, dtype=jnp.float32)
    zero_w13 = tuple(
        jnp.zeros(
            (
                config.num_experts,
                config.hidden_dim,
                2 * config.intermediate_dim,
            ),
            dtype=jnp.float32,
        )
        for _ in stage.blocks
    )
    zero_w2 = tuple(
        jnp.zeros(
            (
                config.num_experts,
                config.intermediate_dim,
                config.hidden_dim,
            ),
            dtype=jnp.float32,
        )
        for _ in stage.blocks
    )
    keys = jax.random.split(jax.random.key(3), 2 * len(stage.blocks))
    prior_w13 = tuple(jax.random.normal(keys[i], value.shape, dtype=jnp.float32) for i, value in enumerate(zero_w13))
    prior_w2 = tuple(
        jax.random.normal(keys[len(stage.blocks) + i], value.shape, dtype=jnp.float32) for i, value in enumerate(zero_w2)
    )
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    zero_w13 = tuple(jax.device_put(value, expert_sharding) for value in zero_w13)
    zero_w2 = tuple(jax.device_put(value, expert_sharding) for value in zero_w2)
    prior_w13 = tuple(jax.device_put(value, expert_sharding) for value in prior_w13)
    prior_w2 = tuple(jax.device_put(value, expert_sharding) for value in prior_w2)

    def stage_loss(stage, w13_accumulators, w2_accumulators):
        output, router_metrics, token = stage.block_range_accumulating_weight_gradient(
            hidden,
            w13_accumulators,
            w2_accumulators,
        )
        normalized_output_loss = jnp.mean(output.astype(jnp.float32) * output_cotangent)
        return normalized_output_loss + token, (output, router_metrics, token)

    value_and_grad = eqx.filter_value_and_grad(stage_loss, has_aux=True)
    with jax.set_mesh(mesh):
        (zero_value, zero_aux), zero_gradient = value_and_grad(stage, zero_w13, zero_w2)
        (prior_value, prior_aux), prior_gradient = value_and_grad(stage, prior_w13, prior_w2)

    np.testing.assert_array_equal(np.asarray(prior_aux[0]), np.asarray(zero_aux[0]))
    np.testing.assert_array_equal(
        np.asarray(prior_aux[1]["capacity_overflow_per_layer"]),
        np.asarray(zero_aux[1]["capacity_overflow_per_layer"]),
    )
    assert float(prior_aux[2]) == 0.0
    assert float(prior_value) == float(zero_value)

    for block_index, (w13_prior, w2_prior) in enumerate(zip(prior_w13, prior_w2, strict=True)):
        zero_expert_gradient = zero_gradient.blocks[block_index].mlp.expert_mlp
        prior_expert_gradient = prior_gradient.blocks[block_index].mlp.expert_mlp
        zero_w13_gradient = jnp.concatenate(
            (zero_expert_gradient.w_gate, zero_expert_gradient.w_up),
            axis=-1,
        )
        prior_w13_gradient = jnp.concatenate(
            (prior_expert_gradient.w_gate, prior_expert_gradient.w_up),
            axis=-1,
        )
        np.testing.assert_allclose(
            np.asarray(prior_w13_gradient),
            np.asarray(zero_w13_gradient + w13_prior),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(prior_expert_gradient.w_down),
            np.asarray(zero_expert_gradient.w_down + w2_prior),
            rtol=1e-5,
            atol=1e-5,
        )

    ordinary_gradient = _stage_without_expert_gradients(prior_gradient)
    expert_gradient = _stage_expert_gradient_accumulators(prior_gradient)
    restored_gradient = _stage_with_expert_gradients(ordinary_gradient, expert_gradient)
    for expected_block, restored_block in zip(prior_gradient.blocks, restored_gradient.blocks, strict=True):
        expected_expert = expected_block.mlp.expert_mlp
        restored_expert = restored_block.mlp.expert_mlp
        np.testing.assert_array_equal(np.asarray(restored_expert.w_gate), np.asarray(expected_expert.w_gate))
        np.testing.assert_array_equal(np.asarray(restored_expert.w_up), np.asarray(expected_expert.w_up))
        np.testing.assert_array_equal(np.asarray(restored_expert.w_down), np.asarray(expected_expert.w_down))
