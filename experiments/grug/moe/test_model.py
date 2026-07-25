# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.model import CausalSelfAttention, DenseMLP, GrugModelConfig, Transformer


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


def test_drop_reporting_preserves_loss_and_gradients(monkeypatch):
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=4,
        moe_implementation="scatter",
        attention_implementation="reference",
    )
    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    loss_weight = jnp.ones_like(tokens, dtype=jnp.float32)

    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.key(0))

        def loss_fn(params):
            return params.next_token_loss(tokens, loss_weight)[0]

        monkeypatch.delenv("SCALE_REPORT_DROPS", raising=False)
        loss_without_reporting, grads_without_reporting = jax.value_and_grad(loss_fn)(model)

        monkeypatch.setenv("SCALE_REPORT_DROPS", "1")
        loss_with_reporting, grads_with_reporting = jax.value_and_grad(loss_fn)(model)
        _, (_, dropped_total) = model.next_token_loss(tokens, loss_weight)

    np.testing.assert_allclose(loss_with_reporting, loss_without_reporting, rtol=0, atol=0)
    for reported, unreported in zip(
        jax.tree.leaves(eqx.filter(grads_with_reporting, eqx.is_array)),
        jax.tree.leaves(eqx.filter(grads_without_reporting, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_allclose(reported, unreported, rtol=0, atol=0)
    assert int(dropped_total) == 0
