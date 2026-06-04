# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    mask = GrugMoeAdamHConfig().create_mask(_optimizer_test_params())

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert mask["token_embed"] == "adam"


def test_grug_moe_adamh_applies_one_outer_global_clip():
    optimizer = GrugMoeAdamHConfig(max_grad_norm=1.0).build(num_train_steps=10)
    state = optimizer.init(_optimizer_test_params())

    clip_state, grouped_state = state.inner_state
    assert type(clip_state).__name__ == "EmptyState"
    assert type(grouped_state).__name__ == "PartitionState"
    assert _inner_state_names(grouped_state, "adamh_expert") == ["ScaleByAdamHState"]


def test_grug_moe_adamh_disables_outer_clip_when_unset():
    optimizer = GrugMoeAdamHConfig(max_grad_norm=None).build(num_train_steps=10)
    state = optimizer.init(_optimizer_test_params())

    assert type(state.inner_state).__name__ == "PartitionState"
    assert _inner_state_names(state.inner_state, "adamh_expert") == ["ScaleByAdamHState"]


def _inner_state_names(partition_state, label):
    masked_state = partition_state.inner_states[label]
    return [type(inner_state).__name__ for inner_state in masked_state.inner_state]


def _optimizer_test_params():
    return {
        "blocks": {
            "0": {
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
