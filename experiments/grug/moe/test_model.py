# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.moe.launch_cw_scale import build_scale_model
from experiments.grug.moe.model import CausalSelfAttention, DenseMLP, GrugModelConfig, _compute_expert_load
from experiments.grug.moe.train import _compute_flops, _updated_router_bias


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

    config = build_scale_model()

    assert config.moe_capacity_factor == 1.125
    assert config.report_capacity_overflow is True
    assert config.qb_bias_update_rate == 0.002


def test_model_config_rejects_nonpositive_capacity_factor():
    with pytest.raises(ValueError, match="moe_capacity_factor must be positive"):
        GrugModelConfig(vocab_size=128, moe_capacity_factor=0.0)


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
