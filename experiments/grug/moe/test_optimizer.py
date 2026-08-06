# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig, TiedExpertLrScale


class _ExpertBank(NamedTuple):
    w_gate: jax.Array


class _Parameters(NamedTuple):
    expert_banks: tuple[_ExpertBank, ...]
    ordinary: jax.Array


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    params = {
        "blocks": {
            "0": {
                "mlp": {
                    "router": jnp.ones((8, 4), dtype=jnp.float32),
                },
                "shared": {
                    "w_gate": jnp.ones((8, 16), dtype=jnp.float32),
                },
            },
        },
        "expert_banks": (
            {
                "w_gate": jnp.ones((4, 8, 16), dtype=jnp.float32),
                "w_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
                "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
            },
        ),
        "token_embed": jnp.ones((128, 8), dtype=jnp.float32),
    }

    mask = GrugMoeAdamHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["router"] == "adam"
    assert mask["expert_banks"][0]["w_gate"] == "adamh_expert"
    assert mask["expert_banks"][0]["w_up"] == "adamh_expert"
    assert mask["expert_banks"][0]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert mask["token_embed"] == "adam"


@pytest.mark.parametrize(
    ("scale", "expected_ratios"),
    [
        (TiedExpertLrScale.UNSCALED, (1.0, 1.0)),
        (TiedExpertLrScale.SQRT, (1.0 / jnp.sqrt(2.0), 0.5)),
        (TiedExpertLrScale.LINEAR, (0.5, 0.25)),
    ],
)
def test_grug_moe_muonh_tied_expert_lr_scales_only_reused_banks(scale, expected_ratios):
    matrix = jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4)
    params = _Parameters(
        expert_banks=tuple(_ExpertBank(w_gate=matrix) for _ in range(3)),
        ordinary=matrix,
    )
    grads = jax.tree.map(jnp.ones_like, params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=1e-3,
        adam_lr=1e-3,
        warmup=0.0,
        lr_schedule="constant",
        momentum=0.0,
        nesterov=False,
        expert_bank_group_sizes=(1, 2, 4),
        tied_expert_lr_scale=scale,
    ).build(num_train_steps=10)

    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    base_norm = jnp.linalg.norm(updates.expert_banks[0].w_gate)
    tied_ratios = tuple(jnp.linalg.norm(updates.expert_banks[bank_index].w_gate) / base_norm for bank_index in (1, 2))

    assert jnp.linalg.norm(updates.ordinary) == pytest.approx(float(base_norm), rel=1e-5)
    assert tied_ratios[0] == pytest.approx(float(expected_ratios[0]), rel=2e-3)
    assert tied_ratios[1] == pytest.approx(float(expected_ratios[1]), rel=2e-3)
