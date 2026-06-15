# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig, _scale_invariant_hyperball_updates


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    params = {
        "blocks": {
            "0": {
                "attn": {
                    "attn_gate": jnp.ones((8, 2), dtype=jnp.float32),
                },
                "attn_gated_norm": {
                    "w_down": jnp.ones((8, 4), dtype=jnp.float32),
                },
                "mlp": {
                    "router": jnp.ones((8, 4), dtype=jnp.float32),
                    "expert_mlp": {
                        "w_gate_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
                    },
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
            },
        },
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
    }

    mask = GrugMoeAdamHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["attn_gated_norm"]["w_down"] == "adamh"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert mask["token_embed"] == "adam"


def test_grug_moe_muonh_mask_routes_may_recipe_groups():
    params = {
        "blocks": {
            "0": {
                "attn": {
                    "w_q": jnp.ones((8, 8), dtype=jnp.float32),
                    "attn_gate": jnp.ones((8, 2), dtype=jnp.float32),
                },
                "attn_gated_norm": {
                    "w_down": jnp.ones((8, 4), dtype=jnp.float32),
                },
                "mlp": {
                    "router": jnp.ones((8, 4), dtype=jnp.float32),
                    "expert_mlp": {
                        "w_gate_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
                    },
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
            },
        },
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
        "output_proj": jnp.ones((8, 128), dtype=jnp.float32),
    }

    mask = GrugMoeMuonHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["attn"]["w_q"] == "muonh"
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["attn_gated_norm"]["w_down"] == "muonh"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "muonh"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "muonh"
    assert block_mask["shared"]["w_gate"] == "muonh"
    assert mask["token_embed"] == "adam"
    assert mask["output_proj"] == "adamh"


def test_grug_moe_muonh_can_route_routed_expert_weights_to_adamh():
    params = {
        "blocks": {
            "0": {
                "mlp": {
                    "expert_mlp": {
                        "w_gate_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
                    },
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
            },
        },
    }

    mask = GrugMoeMuonHConfig(expert_3d_optimizer="adamh").create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "adamh"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh"
    assert block_mask["shared"]["w_gate"] == "muonh"


def test_scale_invariant_hyperball_updates_matches_materialized_formula():
    params = {
        "matrix": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7 + 0.25,
        "stack": jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 11 + 0.5,
    }
    updates = {
        "matrix": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 13 + 0.1,
        "stack": jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 17 + 0.2,
    }
    learning_rate = 0.03

    def materialized_update(param, update):
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    actual = _scale_invariant_hyperball_updates(params, updates, learning_rate)
    expected = jax.tree.map(materialized_update, params, updates)

    assert jnp.allclose(actual["matrix"], expected["matrix"], rtol=1e-5, atol=1e-6).item()
    assert jnp.allclose(actual["stack"], expected["stack"], rtol=1e-5, atol=1e-6).item()
