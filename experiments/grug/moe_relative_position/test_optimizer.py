# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe_relative_position.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    params = {
        "blocks": {
            "0": {
                "mlp": {
                    "router": jnp.ones((8, 4), dtype=jnp.float32),
                    "expert_mlp": {
                        "w_gate": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                        "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
                    },
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
                "attn": {
                    "w_r": jnp.ones((8, 16), dtype=jnp.float32),
                    "relative_position_embeddings": jnp.ones((16, 32), dtype=jnp.float32),
                },
            },
        },
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
    }

    mask = GrugMoeAdamHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_up"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert block_mask["attn"]["relative_position_embeddings"] == "adam"
    assert mask["token_embed"] == "adam"


def test_grug_moe_muonh_mask_routes_relative_queries_and_embeddings_to_distinct_groups():
    params = {
        "blocks": {
            "0": {
                "attn": {
                    "w_r": jnp.ones((8, 16), dtype=jnp.float32),
                    "relative_position_embeddings": jnp.ones((16, 32), dtype=jnp.float32),
                }
            }
        }
    }

    mask = GrugMoeMuonHConfig().create_mask(params)

    assert mask["blocks"]["0"]["attn"]["w_r"] == "muonh"
    assert mask["blocks"]["0"]["attn"]["relative_position_embeddings"] == "adam"
