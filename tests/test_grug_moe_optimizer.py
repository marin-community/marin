# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``GrugMoeAdamHConfig`` param routing.

This branch (``moe_all_gated_norms_to_adam``) tests the "all GatedNorm
instances -> plain Adam (small ``adam_lr``)" variant. On ``main``,
routing was asymmetric: per-block ``attn_gated_norm`` landed in
``adam`` via a substring-match collision, while ``mlp_gated_norm`` and
model-level ``embed_gated_norm`` / ``final_gated_norm`` fell through to
``adamh``. This branch makes the routing symmetric by sending all four
GatedNorm instances to ``adam`` instead.
"""

import jax.numpy as jnp

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig


def test_grug_moe_adamh_routes_all_gated_norms_to_adam():
    params = {
        "token_embed": jnp.ones((128, 32), dtype=jnp.float32),
        "embed_norm": {"weight": jnp.ones((32,), dtype=jnp.float32)},
        "embed_gated_norm": {
            "w_down": jnp.ones((32, 128), dtype=jnp.float32),
            "w_up": jnp.ones((128, 32), dtype=jnp.float32),
        },
        "final_norm": {"weight": jnp.ones((32,), dtype=jnp.float32)},
        "final_gated_norm": {
            "w_down": jnp.ones((32, 128), dtype=jnp.float32),
            "w_up": jnp.ones((128, 32), dtype=jnp.float32),
        },
        "output_proj": jnp.ones((32, 128), dtype=jnp.float32),
        "blocks": (
            {
                "rms_attn": {"weight": jnp.ones((32,), dtype=jnp.float32)},
                "attn_gated_norm": {
                    "w_down": jnp.ones((32, 128), dtype=jnp.float32),
                    "w_up": jnp.ones((128, 32), dtype=jnp.float32),
                },
                "attn": {
                    "w_q": jnp.ones((32, 32), dtype=jnp.float32),
                    "attn_gate": jnp.ones((32, 2), dtype=jnp.float32),
                },
                "rms_mlp": {"weight": jnp.ones((32,), dtype=jnp.float32)},
                "mlp_gated_norm": {
                    "w_down": jnp.ones((32, 128), dtype=jnp.float32),
                    "w_up": jnp.ones((128, 32), dtype=jnp.float32),
                },
                "mlp": {
                    "router": jnp.ones((32, 4), dtype=jnp.float32),
                    "router_bias": jnp.ones((4,), dtype=jnp.float32),
                    "w_gate_up": jnp.ones((4, 32, 64), dtype=jnp.float32),
                    "w_down": jnp.ones((4, 64, 32), dtype=jnp.float32),
                },
                "shared": {
                    "w_up": jnp.ones((32, 64), dtype=jnp.float32),
                    "w_down": jnp.ones((64, 32), dtype=jnp.float32),
                },
            },
        ),
    }

    mask = GrugMoeAdamHConfig().create_mask(params)

    # All four GatedNorm instances route to plain Adam at adam_lr.
    block_mask = mask["blocks"][0]
    assert block_mask["attn_gated_norm"]["w_up"] == "adam"
    assert block_mask["attn_gated_norm"]["w_down"] == "adam"
    assert block_mask["mlp_gated_norm"]["w_up"] == "adam"
    assert block_mask["mlp_gated_norm"]["w_down"] == "adam"
    assert mask["embed_gated_norm"]["w_up"] == "adam"
    assert mask["embed_gated_norm"]["w_down"] == "adam"
    assert mask["final_gated_norm"]["w_up"] == "adam"
    assert mask["final_gated_norm"]["w_down"] == "adam"

    # Other params unaffected: small things stay in adam; expert weights in
    # adamh_expert; lm-head in adamh; attention matrices in adamh.
    assert mask["token_embed"] == "adam"
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["router_bias"] == "adam"
    assert block_mask["mlp"]["w_gate_up"] == "adamh_expert"
    assert block_mask["mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_up"] == "adamh_expert"
    assert mask["output_proj"] == "adamh"
    assert block_mask["attn"]["w_q"] == "adamh"
    assert mask["embed_norm"]["weight"] == "adam"
    assert mask["final_norm"]["weight"] == "adam"
    assert block_mask["rms_attn"]["weight"] == "adam"
    assert block_mask["rms_mlp"]["weight"] == "adam"
