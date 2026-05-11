# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``GrugMoeAdamHConfig`` param routing.

Regression test for the ``attn_gate`` substring bug: previously
``"attn_gate" in path_lower`` also matched ``attn_gated_norm.w_*`` paths
because their names start with the same substring; that sent the per-block
``attn_gated_norm`` matrices into the small-LR Adam group while the other
three GatedNorm instances (``mlp_gated_norm``, ``embed_gated_norm``,
``final_gated_norm``) fell through to the AdamH branch.

The fix uses ``path_lower.endswith(".attn_gate")`` so only the actual
attention-gate leaf is captured. All four GatedNorm instances now route
to ``adamh``.
"""

import jax.numpy as jnp

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig


def test_grug_moe_adamh_routes_all_gated_norms_to_adamh():
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

    # Adam group (small LR): embeddings, biases, routers, the actual attn gate.
    assert mask["token_embed"] == "adam"
    block_mask = mask["blocks"][0]
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["router_bias"] == "adam"

    # All four GatedNorm instances route to adamh now.
    assert block_mask["attn_gated_norm"]["w_up"] == "adamh"
    assert block_mask["attn_gated_norm"]["w_down"] == "adamh"
    assert block_mask["mlp_gated_norm"]["w_up"] == "adamh"
    assert block_mask["mlp_gated_norm"]["w_down"] == "adamh"
    assert mask["embed_gated_norm"]["w_up"] == "adamh"
    assert mask["embed_gated_norm"]["w_down"] == "adamh"
    assert mask["final_gated_norm"]["w_up"] == "adamh"
    assert mask["final_gated_norm"]["w_down"] == "adamh"

    # Expert MLP weights stay in adamh_expert; lm-head matrix in adamh.
    assert block_mask["mlp"]["w_gate_up"] == "adamh_expert"
    assert block_mask["mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_up"] == "adamh_expert"
    assert mask["output_proj"] == "adamh"

    # Attention-projection matrix matches the AdamH branch.
    assert block_mask["attn"]["w_q"] == "adamh"

    # RMSNorm weights (1D) fall through to plain Adam.
    assert mask["embed_norm"]["weight"] == "adam"
    assert mask["final_norm"]["weight"] == "adam"
    assert block_mask["rms_attn"]["weight"] == "adam"
    assert block_mask["rms_mlp"]["weight"] == "adam"


def test_attn_gated_norm_no_longer_collides_with_attn_gate_substring():
    """Direct regression test for the substring bug. Before the fix,
    ``"attn_gate" in path_lower`` matched ``attn_gated_norm`` paths."""
    params = {
        "blocks": (
            {
                "attn": {"attn_gate": jnp.ones((32, 2), dtype=jnp.float32)},
                "attn_gated_norm": {
                    "w_down": jnp.ones((32, 128), dtype=jnp.float32),
                    "w_up": jnp.ones((128, 32), dtype=jnp.float32),
                },
            },
        ),
    }
    mask = GrugMoeAdamHConfig().create_mask(params)
    block_mask = mask["blocks"][0]
    # The actual attn_gate field stays in adam.
    assert block_mask["attn"]["attn_gate"] == "adam"
    # attn_gated_norm matrices now correctly land in adamh.
    assert block_mask["attn_gated_norm"]["w_down"] == "adamh"
    assert block_mask["attn_gated_norm"]["w_up"] == "adamh"
