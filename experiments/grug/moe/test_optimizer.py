# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig


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
