# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask

from experiments.grug.moe import train as grug_train
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
        loss_implementation="reference",
        remat_mode="recompute_all",
    )
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(0))
    return mesh, model


def _assert_tree_exact(actual, expected) -> None:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_automatic_pipeline_router_bias_partition_preserves_loss_and_gradients():
    mesh, model = _tiny_model()
    token_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    tokens = jax.device_put(jnp.array([[1, 2, 3, 4]], dtype=jnp.int32), token_sharding)
    loss_weight = jax.device_put(jnp.ones(tokens.shape, dtype=jnp.float32), token_sharding)
    qb_betas = jnp.array([[0.2, -0.1], [-0.3, 0.4]], dtype=jnp.float32)

    def model_loss(params):
        return params.next_token_loss(
            tokens,
            loss_weight,
            mask=AttentionMask.causal(),
            reduction="mean",
            logsumexp_weight=0.01,
            return_router_metrics=False,
        )

    with jax.set_mesh(mesh):

        def reference_loss(params):
            return model_loss(grug_train._apply_qb_betas(params, qb_betas))

        reference_value, reference_grads = jax.value_and_grad(reference_loss)(model)

        qb_params = grug_train._apply_qb_betas(model, qb_betas)
        pipeline_params, fixed_router_biases = grug_train._detach_router_biases(qb_params)

        def pipeline_loss(params):
            return model_loss(grug_train._replace_router_biases(params, fixed_router_biases))

        pipeline_value, pipeline_grads = jax.value_and_grad(pipeline_loss)(pipeline_params)
        restored_grads = grug_train._restore_zero_router_bias_gradients(pipeline_grads, fixed_router_biases)

    np.testing.assert_array_equal(pipeline_value, reference_value)
    for block in pipeline_params.blocks:
        assert block.mlp is not None
        assert block.mlp.router_bias is None
    for block in pipeline_grads.blocks:
        assert block.mlp is not None
        assert block.mlp.router_bias is None

    reference_non_router_grads, reference_router_grads = grug_train._detach_router_biases(reference_grads)
    restored_non_router_grads, restored_router_grads = grug_train._detach_router_biases(restored_grads)
    _assert_tree_exact(restored_non_router_grads, reference_non_router_grads)
    for reference_router_grad, restored_router_grad in zip(
        reference_router_grads,
        restored_router_grads,
        strict=True,
    ):
        np.testing.assert_array_equal(reference_router_grad, jnp.zeros_like(reference_router_grad))
        np.testing.assert_array_equal(restored_router_grad, jnp.zeros_like(restored_router_grad))
