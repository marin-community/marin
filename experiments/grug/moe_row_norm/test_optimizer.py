# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh

from experiments.grug.moe import model as baseline_model
from experiments.grug.moe_row_norm import model
from experiments.grug.moe_row_norm.optimizer import (
    GrugMoeRowNormConfig,
    scale_by_vector_adamh,
    scale_with_row_muonh,
)
from experiments.grug.moe_row_norm.recipe import baseline_recipe, row_norm_recipe


@pytest.mark.parametrize("shape", [(5, 3), (2, 5, 3)])
def test_row_muonh_preserves_each_output_row_norm(shape):
    params = jax.random.normal(jax.random.key(0), shape)
    grads = jax.random.normal(jax.random.key(1), shape)
    optimizer = scale_with_row_muonh(momentum=0.0, nesterov=False, steps=2, learning_rate=0.1)

    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)

    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(updated), axis=-2),
        np.linalg.norm(np.asarray(params), axis=-2),
        rtol=1e-5,
        atol=1e-5,
    )
    assert not np.allclose(np.asarray(updated), np.asarray(params))


@pytest.mark.parametrize("shape", [(4,), (3, 4)])
def test_vector_adamh_preserves_one_l2_norm_per_linear(shape):
    params = jnp.ones(shape, dtype=jnp.float32)
    grads = jax.random.normal(jax.random.key(2), shape)
    optimizer = scale_by_vector_adamh(b1=0.9, b2=0.95, learning_rate=0.1)

    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)

    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(updated), axis=-1),
        np.linalg.norm(np.asarray(params), axis=-1),
        rtol=1e-5,
        atol=1e-5,
    )
    assert not np.allclose(np.asarray(updated), np.asarray(params))


def test_optimizer_mask_preserves_baseline_exceptions_and_routes_v_to_adamh():
    params = {
        "token_embed": jnp.ones((8, 4)),
        "output_proj": jnp.ones((4, 8)),
        "v_output_proj": jnp.ones((8,)),
        "blocks": {
            "0": {
                "attn": {
                    "w_q": jnp.ones((4, 4)),
                    "v_q": jnp.ones((4,)),
                    "attn_gate": jnp.zeros((4, 2)),
                    "v_attn_gate": jnp.ones((2,)),
                },
                "mlp": {
                    "router": jnp.ones((4, 2)),
                    "v_router": jnp.ones((2,)),
                    "router_bias": jnp.zeros((2,)),
                    "expert_mlp": {
                        "w_gate": jnp.ones((2, 4, 8)),
                        "v_gate": jnp.ones((2, 8)),
                    },
                },
            },
        },
    }

    mask = GrugMoeRowNormConfig().create_mask(params)
    block_mask = mask["blocks"]["0"]

    assert mask["output_proj"] == "adamh"
    assert mask["v_output_proj"] == "vector_adamh"
    assert mask["token_embed"] == "adam"
    assert block_mask["attn"]["w_q"] == "row_muonh"
    assert block_mask["attn"]["v_q"] == "vector_adamh"
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["attn"]["v_attn_gate"] == "vector_adamh"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["v_router"] == "vector_adamh"
    assert block_mask["mlp"]["expert_mlp"]["w_gate"] == "row_muonh"
    assert block_mask["mlp"]["expert_mlp"]["v_gate"] == "vector_adamh"


def test_factorized_model_starts_as_exact_july_model():
    mesh = Mesh(
        np.array([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    config_kwargs = dict(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=4,
        sliding_window=4,
    )
    key = jax.random.key(3)
    token_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    with jax.set_mesh(mesh):
        baseline = baseline_model.Transformer.init(baseline_model.GrugModelConfig(**config_kwargs), key=key)
        factorized = model.Transformer.init(model.GrugModelConfig(**config_kwargs), key=key)
        baseline_logits = baseline.logits(token_ids)
        factorized_logits = factorized.logits(token_ids)

    np.testing.assert_array_equal(np.asarray(factorized.token_embed), np.asarray(baseline.token_embed))
    np.testing.assert_array_equal(np.asarray(factorized.output_proj), np.asarray(baseline.output_proj))
    np.testing.assert_allclose(np.asarray(factorized_logits), np.asarray(baseline_logits), rtol=1e-5, atol=1e-5)

    block = factorized.blocks[0]
    baseline_block = baseline.blocks[0]
    weight_pairs = (
        (factorized.embed_gated_norm.w_down, baseline.embed_gated_norm.w_down),
        (factorized.embed_gated_norm.w_up, baseline.embed_gated_norm.w_up),
        (block.attn.w_q, baseline_block.attn.w_q),
        (block.attn.w_k, baseline_block.attn.w_k),
        (block.attn.w_v, baseline_block.attn.w_v),
        (block.attn.w_o, baseline_block.attn.w_o),
        (block.attn.attn_gate, baseline_block.attn.attn_gate),
        (block.mlp.router, baseline_block.mlp.router),
        (block.mlp.expert_mlp.w_gate, baseline_block.mlp.expert_mlp.w_gate),
        (block.mlp.expert_mlp.w_up, baseline_block.mlp.expert_mlp.w_up),
        (block.mlp.expert_mlp.w_down, baseline_block.mlp.expert_mlp.w_down),
        (block.shared.w_gate, baseline_block.shared.w_gate),
        (block.shared.w_up, baseline_block.shared.w_up),
        (block.shared.w_down, baseline_block.shared.w_down),
    )
    for factorized_weight, baseline_weight in weight_pairs:
        np.testing.assert_array_equal(np.asarray(factorized_weight), np.asarray(baseline_weight))

    scales = (
        factorized.v_output_proj,
        factorized.embed_gated_norm.v_down,
        factorized.embed_gated_norm.v_up,
        block.attn.v_q,
        block.attn.v_k,
        block.attn.v_v,
        block.attn.v_o,
        block.attn.v_attn_gate,
        block.mlp.v_router,
        block.mlp.expert_mlp.v_gate,
        block.mlp.expert_mlp.v_up,
        block.mlp.expert_mlp.v_down,
        block.shared.v_gate,
        block.shared.v_up,
        block.shared.v_down,
    )
    assert all(np.all(np.asarray(scale) == 1) for scale in scales)


def test_launch_recipes_match_exact_july_model_and_optimizer_scalars():
    baseline_config, baseline_optimizer = baseline_recipe()
    variant_config, variant_optimizer = row_norm_recipe()

    assert dataclasses.asdict(variant_config) == dataclasses.asdict(baseline_config)
    baseline_fields = dataclasses.asdict(baseline_optimizer)
    variant_fields = dataclasses.asdict(variant_optimizer)
    assert variant_fields.keys() == baseline_fields.keys()
    for field, baseline_value in baseline_fields.items():
        variant_value = variant_fields[field]
        if isinstance(baseline_value, float):
            assert variant_value == pytest.approx(baseline_value)
        else:
            assert variant_value == baseline_value
