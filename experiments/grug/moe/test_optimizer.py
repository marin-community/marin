# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import jax
import jax.numpy as jnp
import optax

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig


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
    assert mask["token_embed"] == "adam"


def test_grug_moe_muonh_mask_routes_router_to_separate_group():
    params = {
        "blocks": {
            "mlp": {
                "router": jnp.ones((8, 4), dtype=jnp.float32),
                "router_bias": jnp.zeros((4,), dtype=jnp.float32),
            },
        },
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
    }

    mask = GrugMoeMuonHConfig().create_mask(params)

    assert mask["blocks"]["mlp"]["router"] == "router"
    assert mask["blocks"]["mlp"]["router_bias"] == "adam"
    assert mask["token_embed"] == "adam"


def test_grug_moe_muonh_zero_router_lr_freezes_only_router():
    params = {
        "blocks": {
            "mlp": {
                "router": jnp.ones((8, 4), dtype=jnp.float32),
                "router_bias": jnp.zeros((4,), dtype=jnp.float32),
            },
        },
        "token_embed": jnp.ones((16, 8), dtype=jnp.float32),
    }
    config = GrugMoeMuonHConfig(
        learning_rate=0.01,
        adam_lr=0.01,
        router_lr=0.0,
        lr_schedule="constant",
        warmup=0.0,
    )
    optimizer = config.build(num_train_steps=10)

    state = optimizer.init(params)
    updates, _ = optimizer.update(jax.tree.map(jnp.ones_like, params), state, params)
    next_params: Any = optax.apply_updates(params, updates)

    assert jnp.array_equal(next_params["blocks"]["mlp"]["router"], params["blocks"]["mlp"]["router"])
    assert not jnp.array_equal(next_params["token_embed"], params["token_embed"])
