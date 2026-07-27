# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.moe.launch_cw_scale import build_scale_model
from experiments.grug.moe.model import (
    CausalSelfAttention,
    DenseMLP,
    GrugModelConfig,
    MoEMLP,
    _capacity_balanced_top_k_local,
    _capacity_refilled_top_k_local,
    _compute_expert_load,
)
from experiments.grug.moe.train import (
    _compute_flops,
    _expert_capacity_overflow_rate,
    _receiver_capacity_overflow_rate,
    _updated_router_bias,
)


def test_nonexpert_weights_are_sharded_over_data_and_expert_axes():
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1)),
        ("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )

    with jax.set_mesh(mesh):
        dense = DenseMLP.init(16, 32, 0.02, key=jax.random.key(0))
        config = GrugModelConfig(
            vocab_size=128,
            hidden_dim=16,
            intermediate_dim=8,
            shared_expert_intermediate_dim=16,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
        )
        attention = CausalSelfAttention.init(config, key=jax.random.key(1))

    column_sharding = P(("data", "expert"), "model")
    row_sharding = P("model", ("data", "expert"))
    assert dense.w_gate.sharding.spec == column_sharding
    assert dense.w_up.sharding.spec == column_sharding
    assert dense.w_down.sharding.spec == row_sharding
    assert attention.w_q.sharding.spec == column_sharding
    assert attention.w_k.sharding.spec == column_sharding
    assert attention.w_v.sharding.spec == column_sharding
    assert attention.w_o.sharding.spec == row_sharding


def test_compute_flops_uses_sliding_window_for_attention():
    config = GrugModelConfig(
        vocab_size=128,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=32,
        sliding_window=8,
    )

    flops_per_example, summary = _compute_flops(model_config=config)
    expected_per_token = lm_flops_per_token(
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        shared_intermediate_dim=config.shared_expert_intermediate_dim,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        num_heads=config.num_heads,
        seq_len=config.sliding_window,
        vocab_size=config.vocab_size,
        glu=True,
        num_experts=config.num_experts,
        num_shared_experts=1,
        num_experts_per_tok=config.num_experts_per_token,
    )

    assert summary["throughput/flops_per_token_analytic"] == expected_per_token
    assert flops_per_example == 3 * expected_per_token * config.max_seq_len


def test_build_scale_model_reads_capacity_controls(monkeypatch):
    monkeypatch.setenv("SCALE_MOE_CAPACITY_FACTOR", "1.125")
    monkeypatch.setenv("SCALE_REPORT_CAPACITY_OVERFLOW", "1")
    monkeypatch.setenv("SCALE_MOE_QB_BIAS_UPDATE_RATE", "0.002")
    monkeypatch.setenv("SCALE_MOE_CAPACITY_BALANCED_ROUTING", "1")
    monkeypatch.setenv("SCALE_MOE_CAPACITY_BALANCE_ITERATIONS", "6")
    monkeypatch.setenv("SCALE_MOE_CAPACITY_BALANCE_TEMPERATURE", "0.025")
    monkeypatch.setenv("SCALE_MOE_CAPACITY_BALANCE_HARD_ITERATIONS", "3")
    monkeypatch.setenv("SCALE_MOE_CAPACITY_BALANCE_HARD_UPDATE_RATE", "0.15")

    config = build_scale_model()

    assert config.moe_capacity_factor == 1.125
    assert config.report_capacity_overflow is True
    assert config.qb_bias_update_rate == 0.002
    assert config.capacity_balanced_routing is True
    assert config.capacity_balance_iterations == 6
    assert config.capacity_balance_temperature == 0.025
    assert config.capacity_balance_hard_iterations == 3
    assert config.capacity_balance_hard_update_rate == 0.15


def test_model_config_rejects_nonpositive_capacity_factor():
    with pytest.raises(ValueError, match="moe_capacity_factor must be positive"):
        GrugModelConfig(vocab_size=128, moe_capacity_factor=0.0)


def test_model_config_rejects_multiple_load_balancers():
    with pytest.raises(ValueError, match="mutually exclusive"):
        GrugModelConfig(vocab_size=128, qb_routing=True, capacity_balanced_routing=True)


def test_model_config_rejects_refill_below_full_capacity():
    with pytest.raises(ValueError, match=r"requires moe_capacity_factor >= 1.0"):
        GrugModelConfig(vocab_size=128, capacity_refill_routing=True, moe_capacity_factor=0.99)


def test_capacity_balanced_top_k_limits_local_overflow():
    num_tokens = 512
    num_experts = 32
    topk = 4
    router_logits = jax.random.normal(jax.random.key(0), (num_tokens, num_experts)).at[:, :4].add(4.0)

    raw_selected = jax.lax.top_k(router_logits, topk)[1]
    raw_load = jnp.bincount(raw_selected.reshape(-1), length=num_experts)
    selected = _capacity_balanced_top_k_local(
        router_logits,
        topk=topk,
        iterations=2,
        temperature=1.0,
        hard_iterations=2,
        hard_update_rate=0.2,
    )
    load = jnp.bincount(selected.reshape(-1), length=num_experts)
    capacity = math.ceil(1.05 * num_tokens * topk / num_experts)
    raw_overflow = jnp.sum(jnp.maximum(raw_load - capacity, 0)) / raw_load.sum()
    overflow = jnp.sum(jnp.maximum(load - capacity, 0)) / load.sum()

    assert float(raw_overflow) > 0.3
    assert float(overflow) < 0.03
    assert np.all(np.diff(np.sort(np.asarray(selected), axis=-1), axis=-1) > 0)


def test_capacity_balanced_top_k_handles_correlated_logits():
    num_tokens = 512
    num_experts = 32
    topk = 4
    token_factors = jax.random.normal(jax.random.key(1), (num_tokens, 4))
    expert_factors = jax.random.normal(jax.random.key(2), (4, num_experts))
    router_logits = token_factors @ expert_factors

    selected = _capacity_balanced_top_k_local(
        router_logits,
        topk=topk,
        iterations=2,
        temperature=1.0,
        hard_iterations=2,
        hard_update_rate=0.2,
    )
    load = jnp.bincount(selected.reshape(-1), length=num_experts)
    capacity = math.ceil(1.2 * num_tokens * topk / num_experts)
    overflow = jnp.sum(jnp.maximum(load - capacity, 0)) / load.sum()

    assert float(overflow) < 0.03


def test_capacity_refilled_top_k_exactly_fills_experts():
    num_tokens = 512
    num_experts = 32
    topk = 4
    router_logits = jax.random.normal(jax.random.key(3), (num_tokens, num_experts)).at[:, :2].add(8.0)

    raw_selected = jax.lax.top_k(router_logits, topk)[1]
    raw_load = jnp.bincount(raw_selected.reshape(-1), length=num_experts)
    selected, slots, reported_raw_load, replacements = _capacity_refilled_top_k_local(
        router_logits,
        topk=topk,
        capacity_factor=1.0,
    )
    load = jnp.bincount(selected.reshape(-1), length=num_experts)
    capacity = num_tokens * topk // num_experts

    assert int(jnp.max(raw_load)) > capacity
    np.testing.assert_array_equal(np.asarray(reported_raw_load), np.asarray(raw_load))
    assert int(replacements) == int(jnp.sum(jnp.maximum(raw_load - capacity, 0)))
    np.testing.assert_array_equal(np.asarray(load), np.full((num_experts,), capacity))
    for expert in range(num_experts):
        expert_slots = np.asarray(slots)[np.asarray(selected) == expert]
        np.testing.assert_array_equal(np.sort(expert_slots), np.arange(capacity))


def test_loss_free_bias_update_penalizes_overloaded_experts():
    bias = np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    expert_loads = np.asarray([[8, 4, 2, 2]], dtype=np.int32)

    updated = np.asarray(_updated_router_bias(bias, expert_loads, update_rate=0.1))

    assert updated[0, 0] < updated[0, 1] < updated[0, 2]
    assert updated[0, 2] == updated[0, 3]
    np.testing.assert_allclose(updated.mean(axis=-1), 0.0, atol=1e-7)


def test_loss_free_balancing_counts_global_expert_load():
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    config = GrugModelConfig(vocab_size=32, num_experts=4, num_experts_per_token=2)
    selected_experts = jnp.asarray([[0, 1], [0, 2], [3, 0]], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        expert_load = _compute_expert_load(selected_experts, config)

    np.testing.assert_array_equal(expert_load, np.asarray([3, 1, 1, 1], dtype=np.int32))


def test_receiver_capacity_overflow_pools_local_experts():
    expert_loads = jnp.asarray(
        [
            [20, 20, 10, 10, 10, 10, 10, 10],
            [20, 10, 10, 10, 20, 10, 10, 10],
        ],
        dtype=jnp.int32,
    )

    overflow = _receiver_capacity_overflow_rate(
        expert_loads,
        assignments_per_layer=100,
        capacity_factor=1.0,
        expert_axis_size=2,
    )

    np.testing.assert_allclose(overflow, np.asarray([0.1, 0.0], dtype=np.float32))


def test_expert_capacity_overflow_counts_overloaded_experts():
    expert_loads = jnp.asarray(
        [
            [20, 20, 10, 10],
            [16, 16, 16, 16],
        ],
        dtype=jnp.int32,
    )

    overflow = _expert_capacity_overflow_rate(
        expert_loads,
        assignments_per_layer=64,
        capacity_factor=1.0,
    )

    np.testing.assert_allclose(overflow, np.asarray([0.125, 0.0], dtype=np.float32))


def test_loss_free_bias_adjusts_sigmoid_routing_scores():
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    config = GrugModelConfig(
        vocab_size=8,
        hidden_dim=2,
        intermediate_dim=2,
        shared_expert_intermediate_dim=0,
        num_experts=3,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        max_seq_len=1,
        qb_routing=True,
    )
    with jax.set_mesh(mesh):
        mlp = MoEMLP.init(config, key=jax.random.key(0))
        mlp = eqx.tree_at(
            lambda model: model.router,
            mlp,
            jnp.asarray([[4.0, 3.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32),
        )
        mlp = eqx.tree_at(
            lambda model: model.router_bias,
            mlp,
            jnp.asarray([-0.05, 0.05, 0.0], dtype=jnp.float32),
        )
        _, expert_load, _ = mlp(jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32))

    np.testing.assert_array_equal(expert_load, np.asarray([0, 1, 0], dtype=np.int32))
