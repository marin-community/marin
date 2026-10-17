# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fray.types import ResourceConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from marin.training.training import TrainLmOnPodConfig

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.post_training.cruxeval_transfer.train_cruxeval import optimizer, run_training


def test_missing_online_key_fails_before_cache_loading_or_gpu_submission(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "")
    config = TrainLmOnPodConfig(train_config=None, resources=ResourceConfig(), env_vars={"WANDB_MODE": "online"})
    with pytest.raises(ValueError):
        run_training(config)


def test_adamh_preserves_zero_matrix_without_nan():
    params = {"matrix": jnp.zeros((2, 3))}
    transform = scale_by_adamh(learning_rate=1e-5)
    updates, _ = transform.update({"matrix": jnp.ones((2, 3))}, transform.init(params), params)
    np.testing.assert_array_equal(updates["matrix"], np.zeros((2, 3)))


def test_snowball_adamh_updates_weights_and_keeps_imported_router_biases():
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        initializer_std=0.02,
        attention_implementation="reference",
    )
    batch_axis = hax.Axis("batch", 2 * jax.device_count())
    position = hax.Axis("position", 8)
    mapping = {"batch": ("replica_dcn", "data", "expert"), "vocab": "model"}
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)), hax.axis_mapping(mapping):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), config, key=jax.random.key(0))
        imported_bias = jnp.array([-0.001, 0.001, 0.002, -0.002])
        model = eqx.tree_at(lambda m: m.transformer.blocks[0].mlp.router_bias, model, imported_bias)
        tokens = hax.named(jnp.broadcast_to(jnp.arange(8), (batch_axis.size, 8)), (batch_axis, position))
        example = LmExample(
            tokens=tokens,
            loss_weight=hax.named(
                jnp.broadcast_to(jnp.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=jnp.float32), (batch_axis.size, 8)),
                (batch_axis, position),
            ),
        )
        example = hax.shard(example, mapping)

        def loss(current):
            return current.compute_next_token_loss(example).scalar()

        initial_loss, gradients = eqx.filter_value_and_grad(loss)(model)
        # A fresh optimizer plus zero weight decay is important: the model uses
        # bias only for nondifferentiable expert selection and has no QB updates.
        np.testing.assert_array_equal(gradients.transformer.blocks[0].mlp.router_bias, np.zeros(4))
        transform = optimizer(1e-5).build(10)
        params = eqx.filter(model, eqx.is_inexact_array)
        state = transform.init(params)
        updates, state = transform.update(gradients, state, params)
        updated = eqx.apply_updates(model, updates)
        assert np.isfinite(float(initial_loss))
        np.testing.assert_array_equal(updated.transformer.blocks[0].mlp.router_bias, imported_bias)
        assert not np.array_equal(updated.transformer.blocks[0].mlp.router, model.transformer.blocks[0].mlp.router)
        before = model.transformer.blocks[0].mlp.expert_mlp.w_gate
        after = updated.transformer.blocks[0].mlp.expert_mlp.w_gate
        np.testing.assert_allclose(
            jnp.linalg.norm(after, axis=(-2, -1)), jnp.linalg.norm(before, axis=(-2, -1)), rtol=1e-5, atol=1e-7
        )
        # The export retains the frozen values in the field inference consumes.
        exported = updated.to_state_dict()
        np.testing.assert_array_equal(exported["model.layers.0.mlp.router.bias"], imported_bias)
