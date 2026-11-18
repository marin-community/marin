# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig, scale_with_grug_muonh
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig as HeroMuonHConfig


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


@pytest.mark.parametrize("stacked", [False, True])
def test_muonh_groups_match_hero_for_stacked_and_unstacked_models(stacked):
    layer = (2,) if stacked else ()
    params = {
        "blocks": {
            "rms_attn": {"weight": jnp.ones((*layer, 4))},
            "rms_mlp": {"weight": jnp.ones((*layer, 4))},
            "mlp": {
                "expert_mlp": {name: jnp.ones((*layer, 2, 4, 3)) for name in ("w_gate", "w_up", "w_down")},
                "router": jnp.ones((*layer, 4, 2)),
            },
            "attn": {"w_q": jnp.ones((*layer, 4, 4)), "attn_gate": jnp.ones((*layer, 4))},
            "attn_gated_norm": {"w_up": jnp.ones((*layer, 4, 2))},
        },
        "token_embed": jnp.ones((8, 4)),
        "output_proj": jnp.ones((4, 8)),
    }
    actual = GrugMoeMuonHConfig().create_mask(params)
    assert actual == HeroMuonHConfig().create_mask(params)
    assert actual["blocks"]["rms_attn"]["weight"] == "adam"
    assert actual["blocks"]["rms_mlp"]["weight"] == "adam"
    assert set(actual["blocks"]["mlp"]["expert_mlp"].values()) == {"muonh"}
    assert actual["blocks"]["attn_gated_norm"]["w_up"] == "muonh"
    assert actual["blocks"]["mlp"]["router"] == "adam"
    assert actual["blocks"]["attn"]["w_q"] == "muonh"
    assert actual["blocks"]["attn"]["attn_gate"] == "adam"
    assert actual["token_embed"] == "adam"
    assert actual["output_proj"] == "adamh"


def test_muonh_four_dimensional_expert_update_matches_backend():
    params = {"expert_mlp": {"w_gate": jnp.arange(1, 49, dtype=jnp.float32).reshape(2, 2, 3, 4)}}
    grads = {"expert_mlp": {"w_gate": jnp.linspace(-0.5, 0.7, 48).reshape(2, 2, 3, 4)}}
    config = GrugMoeMuonHConfig(learning_rate=0.02, warmup=0, max_grad_norm=None)
    optimizer = config.build(100)
    update, _ = optimizer.update(grads, optimizer.init(params), params)
    backend = scale_with_grug_muonh(learning_rate=config.lr_scheduler(100)(0))
    expected, _ = backend.update(grads, backend.init(params), params)
    np.testing.assert_allclose(update["expert_mlp"]["w_gate"], expected["expert_mlp"]["w_gate"], rtol=1e-6, atol=1e-7)
