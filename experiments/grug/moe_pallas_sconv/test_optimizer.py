# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe_pallas_sconv.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig


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
                "sconv_attn": {
                    "weight": jnp.ones((8, 4), dtype=jnp.float32),
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
    assert block_mask["sconv_attn"]["weight"] == "adam"
    assert mask["token_embed"] == "adam"


def test_grug_moe_muonh_mask_routes_sconv_weights_to_adam():
    params = {"blocks": {"0": {"attn": {"sconv_k": {"weight": jnp.ones((8, 4))}}}}}

    mask = GrugMoeMuonHConfig().create_mask(params)

    assert mask["blocks"]["0"]["attn"]["sconv_k"]["weight"] == "adam"
