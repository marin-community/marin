# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import numpy as np
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.launch_cw_scale import build_scale_model
from experiments.grug.moe.model import CausalSelfAttention, DenseMLP, GrugModelConfig, MoEMLP


def test_scale_and_direct_moe_resolve_the_same_capacity_default(monkeypatch):
    monkeypatch.delenv("SCALE_CAPACITY_FACTOR", raising=False)

    scale_config = build_scale_model()
    direct_moe = MoEExpertMlp.init(
        num_experts=4,
        hidden_dim=8,
        intermediate_dim=4,
        initializer_std=0.02,
        key=jax.random.key(0),
    )

    assert scale_config.capacity_factor == direct_moe.capacity_factor


def test_scale_capacity_factor_is_resolved_before_model_construction(monkeypatch):
    monkeypatch.setenv("SCALE_CAPACITY_FACTOR", "1.0625")
    config = dataclasses.replace(
        build_scale_model(),
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )
    monkeypatch.setenv("SCALE_CAPACITY_FACTOR", "1.25")
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )

    with jax.set_mesh(mesh):
        moe = MoEMLP.init(config, key=jax.random.key(1))

    assert moe.expert_mlp.capacity_factor == config.capacity_factor


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
