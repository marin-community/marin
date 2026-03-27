# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from levanter.optim.grugmuon import (
    GrugMuonConfig,
    STACK_BATCH_SHARDED,
    VMAP_REPLICATED,
    _grug_scale_with_muon,
    _zeropower_via_newtonschulz_batched_stack_sharded,
    _zeropower_via_newtonschulz_replicated,
)


def test_grug_scale_with_muon_orthogonalizes_matrix_trailing_dims():
    updates = {
        "matrix": jnp.ones((2, 3), dtype=jnp.float32),
        "moe_tensor": jnp.ones((2, 2, 2), dtype=jnp.float32),
        "vector": jnp.ones((3,), dtype=jnp.float32),
    }
    transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        use_kimi_scaling=False,
        orthogonalization_layout=VMAP_REPLICATED,
    )

    new_updates, _ = transform.update(updates, transform.init(updates))

    assert new_updates["matrix"].shape == updates["matrix"].shape
    assert new_updates["moe_tensor"].shape == updates["moe_tensor"].shape
    assert not jnp.array_equal(new_updates["matrix"], updates["matrix"])
    assert not jnp.array_equal(new_updates["moe_tensor"], updates["moe_tensor"])
    assert jnp.array_equal(new_updates["vector"], updates["vector"])


def test_grug_muon_mask_routes_stacked_expert_weights_to_muon():
    params = {
        "embed": jnp.ones((16, 8), dtype=jnp.float32),
        "router": jnp.ones((8, 4), dtype=jnp.float32),
        "moe": {
            "w_up_gate": jnp.ones((4, 8, 16), dtype=jnp.float32),
            "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
        },
        "vector": jnp.ones((8,), dtype=jnp.float32),
    }

    mask = GrugMuonConfig().create_mask(params)

    assert mask["embed"] == "adamw"
    assert mask["router"] == "muon"
    assert mask["moe"]["w_up_gate"] == "muon"
    assert mask["moe"]["w_down"] == "muon"
    assert mask["vector"] == "adamw"


def test_batched_stack_sharded_matches_vmap_replicated_without_mesh():
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)

    expected = jax.vmap(
        lambda matrix: _zeropower_via_newtonschulz_replicated(matrix, steps=2, eps=1e-7, coefficient_type="quintic")
    )(x)
    actual = _zeropower_via_newtonschulz_batched_stack_sharded(
        x,
        steps=2,
        eps=1e-7,
        coefficient_type="quintic",
    )

    assert jnp.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_grug_scale_with_muon_stack_batch_sharded_handles_stacked_expert_tensor():
    updates = {"moe_tensor": jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)}
    transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        use_kimi_scaling=False,
        orthogonalization_layout=STACK_BATCH_SHARDED,
    )

    new_updates, _ = transform.update(updates, transform.init(updates))

    assert new_updates["moe_tensor"].shape == updates["moe_tensor"].shape
    assert not jnp.array_equal(new_updates["moe_tensor"], updates["moe_tensor"])
