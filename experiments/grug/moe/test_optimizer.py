# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig, scale_with_grug_muonh
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig as HeroMuonHConfig
from experiments.grug.moe_hero_ep.train import clip_by_total_norm


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


@pytest.mark.parametrize("clip", [None, 0.1])
def test_muonh_update_matches_hero_with_optional_global_clip(clip):
    params = {
        "matrix": jnp.arange(1, 13, dtype=jnp.float32).reshape(3, 4),
        "stacked_matrix": jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4),
        "mlp": {"expert_mlp": {"w_gate": jnp.arange(1, 49, dtype=jnp.float32).reshape(2, 2, 3, 4)}},
        "rms_attn": {"weight": jnp.ones((2, 4))},
    }
    grads = jax.tree.map(lambda x: jnp.sin(x) * 0.2, params)
    settings = dict(learning_rate=0.02, warmup=0, max_grad_norm=None, use_syrk=False, expert_normalization="all_experts")
    actual = GrugMoeMuonHConfig(**settings, global_grad_norm=clip).build(100)
    reference = HeroMuonHConfig(**settings).build(100)
    reference_grads = grads
    if clip is not None:
        reference_grads = clip_by_total_norm(grads, clip).grads
    actual_updates, _ = actual.update(grads, actual.init(params), params)
    expected_updates, _ = reference.update(reference_grads, reference.init(params), params)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual_updates), jax.tree.leaves(expected_updates), strict=True
    ):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=0, atol=3e-6)
    np.testing.assert_array_equal(
        actual_updates["mlp"]["expert_mlp"]["w_gate"], expected_updates["mlp"]["expert_mlp"]["w_gate"]
    )


def test_muonh_rejects_simultaneous_global_and_group_clipping():
    with pytest.raises(ValueError, match="global_grad_norm"):
        GrugMoeMuonHConfig(global_grad_norm=1.0, max_grad_norm=1.0)


def test_global_clipping_preserves_optimizer_state_paths():
    params = {"matrix": jnp.ones((3, 4)), "mlp": {"expert_mlp": {"w_gate": jnp.ones((2, 2, 3, 4))}}}
    unclipped = GrugMoeMuonHConfig(use_syrk=False).build(100).init(params)
    clipped = GrugMoeMuonHConfig(use_syrk=False, global_grad_norm=1.0).build(100).init(params)
    assert unclipped.hyperparams.keys() == clipped.hyperparams.keys()
    assert [path for path, _ in jax.tree_util.tree_flatten_with_path(unclipped)[0]] == [
        path for path, _ in jax.tree_util.tree_flatten_with_path(clipped)[0]
    ]


@pytest.mark.parametrize("stacked", [False, True])
def test_muonh_state_matches_hero_layout(stacked):
    shape = (2, 2, 3, 4) if stacked else (2, 3, 4)
    params = {"mlp": {"expert_mlp": {"w_gate": jnp.ones(shape)}}, "rms_attn": {"weight": jnp.ones(4)}}
    actual = GrugMoeMuonHConfig(use_syrk=False).build(100).init(params)
    expected = HeroMuonHConfig(use_syrk=False).build(100).init(params)
    actual_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
    expected_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
    assert [(p, x.shape, x.dtype) for p, x in actual_leaves] == [(p, x.shape, x.dtype) for p, x in expected_leaves]


@pytest.mark.multidevice
def test_muonh_sharded_expert_update_matches_hero():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with jax.set_mesh(mesh):
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "data", None, None))
        param = jax.device_put(jnp.arange(1, 49, dtype=jnp.float32).reshape(2, 2, 3, 4), sharding)
        params = {"mlp": {"expert_mlp": {"w_gate": param}}}
        grads = jax.tree.map(jnp.sin, params)
        settings = dict(learning_rate=0.02, warmup=0, use_syrk=False, expert_normalization="all_experts")
        actual = GrugMoeMuonHConfig(**settings).build(100)
        reference = HeroMuonHConfig(**settings).build(100)
        actual_updates, _ = jax.jit(actual.update)(grads, actual.init(params), params)
        expected_updates, _ = jax.jit(reference.update)(grads, reference.init(params), params)
        np.testing.assert_allclose(
            actual_updates["mlp"]["expert_mlp"]["w_gate"],
            expected_updates["mlp"]["expert_mlp"]["w_gate"],
            rtol=0,
            atol=3e-6,
        )


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_muonh_unstacked_update_matches_existing_backend(shape):
    size = int(np.prod(shape))
    param = jnp.arange(1, size + 1, dtype=jnp.float32).reshape(shape)
    grad = jnp.linspace(-0.5, 0.7, size).reshape(shape)
    params = {"mlp": {"expert_mlp": {"w_gate": param}}}
    grads = {"mlp": {"expert_mlp": {"w_gate": grad}}}
    config = GrugMoeMuonHConfig(learning_rate=0.02, warmup=0, max_grad_norm=None, use_syrk=False)
    optimizer = config.build(100)
    update, _ = optimizer.update(grads, optimizer.init(params), params)
    backend = scale_with_grug_muonh(learning_rate=config.lr_scheduler(100)(0))
    expected, _ = backend.update(grads, backend.init(params), params)
    np.testing.assert_array_equal(update["mlp"]["expert_mlp"]["w_gate"], expected["mlp"]["expert_mlp"]["w_gate"])
