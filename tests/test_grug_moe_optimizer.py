# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig


def test_grug_moe_muonh_routes_every_matrix_except_lm_head_to_muonh():
    params = {
        "token_embed": jnp.ones((128, 32), dtype=jnp.float32),
        "output_proj": jnp.ones((32, 128), dtype=jnp.float32),
        "lm_head": jnp.ones((32, 128), dtype=jnp.float32),
        "blocks": (
            {
                "attn": {
                    "w_q": jnp.ones((32, 32), dtype=jnp.float32),
                    "attn_gate": jnp.ones((32, 2), dtype=jnp.float32),
                },
                "mlp": {
                    "router": jnp.ones((32, 4), dtype=jnp.float32),
                    "router_bias": jnp.ones((4,), dtype=jnp.float32),
                    "w_gate_up": jnp.ones((4, 32, 64), dtype=jnp.float32),
                    "w_down": jnp.ones((4, 64, 32), dtype=jnp.float32),
                },
                "shared": {
                    "w_up": jnp.ones((32, 64), dtype=jnp.float32),
                },
                "rms_attn": {
                    "weight": jnp.ones((32,), dtype=jnp.float32),
                },
            },
        ),
    }

    mask = GrugMoeMuonHConfig().create_mask(params)

    assert mask["token_embed"] == "muonh"
    assert mask["output_proj"] == "adamh"
    assert mask["lm_head"] == "adamh"
    block_mask = mask["blocks"][0]
    assert block_mask["attn"]["w_q"] == "muonh"
    assert block_mask["attn"]["attn_gate"] == "muonh"
    assert block_mask["mlp"]["router"] == "muonh"
    assert block_mask["mlp"]["w_gate_up"] == "muonh"
    assert block_mask["mlp"]["w_down"] == "muonh"
    assert block_mask["shared"]["w_up"] == "muonh"
    assert block_mask["mlp"]["router_bias"] == "adam"
    assert block_mask["rms_attn"]["weight"] == "adam"


def test_muonh_matrix_sweep_gate1_builds_readme_gate_steps():
    from experiments.grug.moe import muonh_matrix_sweep

    steps = muonh_matrix_sweep._build_steps("1")

    assert [step.name for step in steps] == [
        "grug/muonh_matrix_sweep/muonh-matrix-d512-2.19e17",
        "grug/muonh_matrix_sweep/muonh-matrix-d768-1.70e18",
    ]
    assert all(isinstance(step.config.optimizer.value, GrugMoeMuonHConfig) for step in steps)


def test_grug_moe_muonh_optimizer_update_runs_on_single_device():
    params = {
        "token_embed": jnp.ones((8, 4), dtype=jnp.float32),
        "output_proj": jnp.ones((4, 8), dtype=jnp.float32),
        "block": {
            "router": jnp.ones((4, 2), dtype=jnp.float32),
            "w_gate_up": jnp.ones((2, 4, 6), dtype=jnp.float32),
            "norm": jnp.ones((4,), dtype=jnp.float32),
        },
    }
    updates = jax.tree.map(lambda x: jnp.full_like(x, 0.1), params)
    optimizer = GrugMoeMuonHConfig(learning_rate=0.01, adam_lr=0.001, warmup=0).build(10)

    out, _ = optimizer.update(updates, optimizer.init(params), params)

    assert jax.tree.map(lambda x: x.shape, out) == jax.tree.map(lambda x: x.shape, params)
