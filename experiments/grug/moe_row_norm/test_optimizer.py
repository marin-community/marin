# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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


def test_factorized_model_starts_with_baseline_effective_state_dict():
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

    with jax.set_mesh(mesh):
        baseline = baseline_model.Transformer.init(baseline_model.GrugModelConfig(**config_kwargs), key=key)
        factorized = model.Transformer.init(model.GrugModelConfig(**config_kwargs), key=key)
        baseline_state = baseline.to_state_dict()
        factorized_state = factorized.to_state_dict()

    assert factorized_state.keys() == baseline_state.keys()
    for name in baseline_state:
        np.testing.assert_array_equal(np.asarray(factorized_state[name]), np.asarray(baseline_state[name]))

    block = factorized.blocks[0]
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
