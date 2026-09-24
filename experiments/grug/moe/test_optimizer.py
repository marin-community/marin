# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from levanter.optim.config import OptimizerConfig

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig as HistoricalAdamH
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig as HistoricalMuonH


def test_historical_and_current_optimizer_choices_can_coexist():
    assert OptimizerConfig.get_choice_class("grug_moe_adamh_v2") is GrugMoeAdamHConfig
    assert OptimizerConfig.get_choice_class("grug_moe_muonh_v1") is GrugMoeMuonHConfig
    assert OptimizerConfig.get_choice_class("june_tpu_67b_a2b_moe_adamh_v2") is HistoricalAdamH
    assert OptimizerConfig.get_choice_class("june_tpu_67b_a2b_moe_muonh_v1") is HistoricalMuonH


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
