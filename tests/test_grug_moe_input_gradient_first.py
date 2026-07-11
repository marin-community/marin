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
from levanter.grug.attention import AttentionMask

from experiments.grug.moe import train as grug_train
from experiments.grug.moe.model import GrugModelConfig, Transformer

_ATOL = 2e-5
_RTOL = 2e-5


def _assert_tree_allclose(actual, expected) -> None:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=_ATOL, rtol=_RTOL)


def _add_trees(left, right):
    return jax.tree.map(lambda x, y: x + y, left, right)


def _apply_update(params, updates):
    return jax.tree.map(
        lambda param, update: None if param is None else param + update,
        params,
        updates,
        is_leaf=lambda value: value is None,
    )


def _tiny_stage(remat_mode: str, stage_index: int):
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=6,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        sliding_window=4,
        router_z_loss_coef=0.1,
        attention_implementation="reference",
        moe_implementation="scatter",
        loss_implementation="reference",
        remat_mode=remat_mode,
    )
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(0))
    return mesh, model.split_for_pipeline(3)[stage_index]


def _batch(tokens: jax.Array) -> GrugLmExample:
    return GrugLmExample(
        tokens=tokens,
        loss_weight=jnp.ones(tokens.shape, dtype=jnp.float32),
        attn_mask=AttentionMask.causal(),
    )


def _shard_batch(batch: GrugLmExample, mesh: Mesh) -> GrugLmExample:
    token_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    return dataclasses.replace(
        batch,
        tokens=jax.device_put(batch.tokens, token_sharding),
        loss_weight=jax.device_put(batch.loss_weight, token_sharding),
    )


def _shard_activation(value: jax.Array, mesh: Mesh) -> jax.Array:
    sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None, None))
    return jax.device_put(value, sharding)


@pytest.mark.parametrize("remat_mode", ["recompute_all", "save_moe"])
def test_middle_stage_split_backward_matches_combined_backward_and_update(remat_mode: str):
    mesh, stage = _tiny_stage(remat_mode, stage_index=1)
    mp = jmp.get_policy("f32")
    qb_betas = jnp.array([[0.2, -0.1], [-0.3, 0.4]], dtype=jnp.float32)
    batches = tuple(
        _shard_batch(_batch(tokens), mesh)
        for tokens in (
            jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
            jnp.array([[4, 3, 2, 1]], dtype=jnp.int32),
        )
    )
    hiddens = tuple(_shard_activation(jax.random.normal(jax.random.PRNGKey(key), (1, 4, 8)), mesh) for key in (1, 2))
    output_cotangents = tuple(
        _shard_activation(jax.random.normal(jax.random.PRNGKey(key), (1, 4, 8)), mesh) for key in (3, 4)
    )

    combined_grads = []
    split_grads = []
    with jax.set_mesh(mesh):
        for batch, hidden, output_cotangent in zip(batches, hiddens, output_cotangents, strict=True):

            def activation_projection(
                stage_params,
                stage_hidden,
                batch=batch,
                output_cotangent=output_cotangent,
            ):
                compute_params = grug_train._compute_stage(stage_params, qb_betas, mp)
                block_output, _ = compute_params.block_range(stage_hidden, mask=batch.attn_mask)
                return jnp.sum(block_output.astype(jnp.float32) * output_cotangent.astype(jnp.float32))

            combined_grad, combined_d_hidden = jax.grad(activation_projection, argnums=(0, 1))(stage, hidden)
            split_d_hidden, residuals = grug_train._stage_input_gradient_backward(
                stage, qb_betas, hidden, batch, output_cotangent, mp
            )
            split_grad = grug_train._stage_weight_backward(stage, qb_betas, residuals, batch, mp)

            np.testing.assert_allclose(split_d_hidden, combined_d_hidden, atol=_ATOL, rtol=_RTOL)
            _assert_tree_allclose(split_grad, combined_grad)
            combined_grads.append(combined_grad)
            split_grads.append(split_grad)

        combined_accumulated = _add_trees(combined_grads[0], combined_grads[1])
        split_accumulated = _add_trees(split_grads[0], split_grads[1])
        _assert_tree_allclose(split_accumulated, combined_accumulated)

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(stage)
        combined_updates, _ = optimizer.update(combined_accumulated, opt_state, stage)
        split_updates, _ = optimizer.update(split_accumulated, opt_state, stage)
        _assert_tree_allclose(_apply_update(stage, split_updates), _apply_update(stage, combined_updates))


@pytest.mark.parametrize("remat_mode", ["recompute_all", "save_moe"])
def test_last_stage_split_backward_matches_combined_backward_and_update(remat_mode: str):
    mesh, stage = _tiny_stage(remat_mode, stage_index=2)
    mp = jmp.get_policy("f32")
    qb_betas = jnp.array([[0.2, -0.1], [-0.3, 0.4]], dtype=jnp.float32)
    batches = tuple(
        _shard_batch(_batch(tokens), mesh)
        for tokens in (
            jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
            jnp.array([[4, 3, 2, 1]], dtype=jnp.int32),
        )
    )
    hiddens = tuple(_shard_activation(jax.random.normal(jax.random.PRNGKey(key), (1, 4, 8)), mesh) for key in (5, 6))

    combined_losses = []
    split_losses = []
    combined_grads = []
    split_grads = []
    with jax.set_mesh(mesh):
        for batch, hidden in zip(batches, hiddens, strict=True):

            def loss_fn(stage_params, stage_hidden, batch=batch):
                compute_params = grug_train._compute_stage(stage_params, qb_betas, mp)
                block_output, router_metrics = compute_params.block_range(stage_hidden, mask=batch.attn_mask)
                final_hidden = compute_params.finalize_hidden(block_output)
                return compute_params.hidden_next_token_loss(
                    final_hidden,
                    batch.tokens,
                    batch.loss_weight,
                    router_metrics,
                    reduction="mean",
                    logsumexp_weight=0.01,
                )

            combined_loss, (combined_grad, combined_d_hidden) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
                stage, hidden
            )
            split_loss, _, split_d_hidden, residuals = grug_train._last_stage_input_gradient_backward(
                stage,
                qb_betas,
                hidden,
                batch,
                mp,
                logsumexp_weight=0.01,
            )
            split_grad = grug_train._last_stage_weight_backward(
                stage,
                qb_betas,
                residuals,
                batch,
                mp,
                logsumexp_weight=0.01,
            )

            np.testing.assert_allclose(split_loss, combined_loss, atol=_ATOL, rtol=_RTOL)
            np.testing.assert_allclose(split_d_hidden, combined_d_hidden, atol=_ATOL, rtol=_RTOL)
            _assert_tree_allclose(split_grad, combined_grad)
            combined_losses.append(combined_loss)
            split_losses.append(split_loss)
            combined_grads.append(combined_grad)
            split_grads.append(split_grad)

        np.testing.assert_allclose(
            jnp.mean(jnp.stack(split_losses)), jnp.mean(jnp.stack(combined_losses)), atol=_ATOL, rtol=_RTOL
        )
        combined_accumulated = _add_trees(combined_grads[0], combined_grads[1])
        split_accumulated = _add_trees(split_grads[0], split_grads[1])
        _assert_tree_allclose(split_accumulated, combined_accumulated)

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(stage)
        combined_updates, _ = optimizer.update(combined_accumulated, opt_state, stage)
        split_updates, _ = optimizer.update(split_accumulated, opt_state, stage)
        _assert_tree_allclose(_apply_update(stage, split_updates), _apply_update(stage, combined_updates))
