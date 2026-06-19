# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P, use_abstract_mesh

from levanter.optim.grugmuon import (
    GrugMuonConfig,
    STACK_BATCH_4D_SHARDED,
    STACK_BATCH_SHARDED,
    VMAP_REPLICATED,
    _assert_stack_batch_sharded,
    _batch_sharded_stack_target_pspec,
    _grug_scale_with_muon,
    _grouped_4d_stack_target,
    _grouped_4d_stack_target_pspec,
    _zeropower_via_newtonschulz_batched_stack_sharded,
    _zeropower_via_newtonschulz_grouped_4d_sharded,
    _zeropower_via_newtonschulz_replicated,
)


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


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
            "w_gate_up": jnp.ones((4, 8, 16), dtype=jnp.float32),
            "w_down": jnp.ones((4, 16, 8), dtype=jnp.float32),
        },
        "vector": jnp.ones((8,), dtype=jnp.float32),
    }

    mask = GrugMuonConfig().create_mask(params)

    assert mask["embed"] == "adamw"
    assert mask["router"] == "muon"
    assert mask["moe"]["w_up_gate"] == "muon"
    assert mask["moe"]["w_gate_up"] == "muon"
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


def test_grouped_4d_stack_sharded_matches_nested_vmap_replicated_without_mesh():
    x = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5) / 17 + 0.1

    expected = jax.vmap(
        jax.vmap(
            lambda matrix: _zeropower_via_newtonschulz_replicated(
                matrix,
                steps=2,
                eps=1e-7,
                coefficient_type="quintic",
            )
        )
    )(x)
    actual = _zeropower_via_newtonschulz_grouped_4d_sharded(
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


def test_grug_scale_with_muon_grouped_stack_matches_vmap_replicated():
    updates = {
        "layer0": {
            "w_gate_up": jnp.arange(1 * 3 * 4, dtype=jnp.float32).reshape(1, 3, 4) / 17 + 0.1,
            "w_down": jnp.arange(1 * 4 * 3, dtype=jnp.float32).reshape(1, 4, 3) / 19 + 0.2,
        },
        "layer1": {
            "w_gate_up": jnp.arange(1 * 3 * 4, dtype=jnp.float32).reshape(1, 3, 4) / 23 + 0.3,
            "w_down": jnp.arange(1 * 4 * 3, dtype=jnp.float32).reshape(1, 4, 3) / 29 + 0.4,
        },
        "layer2": {
            "w_gate_up": jnp.arange(1 * 3 * 4, dtype=jnp.float32).reshape(1, 3, 4) / 37 + 0.6,
            "w_down": jnp.arange(1 * 4 * 3, dtype=jnp.float32).reshape(1, 4, 3) / 41 + 0.7,
        },
        "dense": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 31 + 0.5,
        "vector": jnp.arange(5, dtype=jnp.float32),
    }
    grouped_transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        steps=2,
        use_kimi_scaling=False,
        orthogonalization_layout=STACK_BATCH_SHARDED,
        max_grouped_stack_size=2,
    )
    replicated_transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        steps=2,
        use_kimi_scaling=False,
        orthogonalization_layout=VMAP_REPLICATED,
    )

    grouped_updates, _ = grouped_transform.update(updates, grouped_transform.init(updates), updates)
    replicated_updates, _ = replicated_transform.update(updates, replicated_transform.init(updates), updates)

    assert jnp.allclose(grouped_updates["layer0"]["w_gate_up"], replicated_updates["layer0"]["w_gate_up"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer1"]["w_gate_up"], replicated_updates["layer1"]["w_gate_up"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer2"]["w_gate_up"], replicated_updates["layer2"]["w_gate_up"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer0"]["w_down"], replicated_updates["layer0"]["w_down"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer1"]["w_down"], replicated_updates["layer1"]["w_down"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer2"]["w_down"], replicated_updates["layer2"]["w_down"], atol=1e-5)
    assert jnp.allclose(grouped_updates["dense"], replicated_updates["dense"], atol=1e-5)
    assert jnp.array_equal(grouped_updates["vector"], updates["vector"])


def test_grug_scale_with_muon_grouped_4d_matches_vmap_replicated():
    updates = {
        "layer0": {
            "w_gate_up": jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 17 + 0.1,
            "w_down": jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 19 + 0.2,
        },
        "layer1": {
            "w_gate_up": jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 23 + 0.3,
            "w_down": jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 29 + 0.4,
        },
        "layer2": {
            "w_gate_up": jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 37 + 0.6,
            "w_down": jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 41 + 0.7,
        },
        "dense": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 31 + 0.5,
        "vector": jnp.arange(5, dtype=jnp.float32),
    }
    grouped_transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        steps=2,
        use_kimi_scaling=False,
        orthogonalization_layout=STACK_BATCH_4D_SHARDED,
    )
    replicated_transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        steps=2,
        use_kimi_scaling=False,
        orthogonalization_layout=VMAP_REPLICATED,
    )

    grouped_updates, _ = grouped_transform.update(updates, grouped_transform.init(updates), updates)
    replicated_updates, _ = replicated_transform.update(updates, replicated_transform.init(updates), updates)

    assert jnp.allclose(grouped_updates["layer0"]["w_gate_up"], replicated_updates["layer0"]["w_gate_up"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer1"]["w_gate_up"], replicated_updates["layer1"]["w_gate_up"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer2"]["w_gate_up"], replicated_updates["layer2"]["w_gate_up"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer0"]["w_down"], replicated_updates["layer0"]["w_down"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer1"]["w_down"], replicated_updates["layer1"]["w_down"], atol=1e-5)
    assert jnp.allclose(grouped_updates["layer2"]["w_down"], replicated_updates["layer2"]["w_down"], atol=1e-5)
    assert jnp.allclose(grouped_updates["dense"], replicated_updates["dense"], atol=1e-5)
    assert jnp.array_equal(grouped_updates["vector"], updates["vector"])


def test_grug_scale_with_muon_rejects_unknown_ns_compute_dtype():
    with pytest.raises(ValueError, match="ns_compute_dtype"):
        _grug_scale_with_muon(ns_compute_dtype="int8")


def test_grug_scale_with_muon_can_compute_ns_in_bf16_for_fp32_updates():
    updates = {
        "dense": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 31 + 0.5,
        "layer0": {"w_gate_up": jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4) / 17 + 0.1},
        "layer1": {"w_gate_up": jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4) / 19 + 0.2},
    }
    bf16_transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        steps=2,
        use_kimi_scaling=False,
        orthogonalization_layout=STACK_BATCH_4D_SHARDED,
        ns_compute_dtype="bf16",
    )
    fp32_transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        steps=2,
        use_kimi_scaling=False,
        orthogonalization_layout=STACK_BATCH_4D_SHARDED,
        ns_compute_dtype="input",
    )

    bf16_updates, _ = bf16_transform.update(updates, bf16_transform.init(updates), updates)
    fp32_updates, _ = fp32_transform.update(updates, fp32_transform.init(updates), updates)

    assert bf16_updates["dense"].dtype == jnp.float32
    assert bf16_updates["layer0"]["w_gate_up"].dtype == jnp.float32
    assert bf16_updates["layer1"]["w_gate_up"].dtype == jnp.float32
    assert not jnp.array_equal(bf16_updates["dense"], fp32_updates["dense"])
    assert not jnp.array_equal(bf16_updates["layer0"]["w_gate_up"], fp32_updates["layer0"]["w_gate_up"])


def test_replicated_newton_schulz_handles_concrete_sharding_without_abstract_mesh():
    mesh = jax.make_mesh((1,), ("data",))
    sharding = NamedSharding(mesh, P(None, "data"))
    x = jax.device_put(jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 11 + 0.1, sharding)

    with _reset_abstract_mesh():
        update = jax.jit(
            lambda y: _zeropower_via_newtonschulz_replicated(
                y,
                steps=2,
                eps=1e-7,
                coefficient_type="quintic",
            )
        )
        actual = update(x)

    assert actual.shape == x.shape
    assert isinstance(actual.sharding, NamedSharding)
    assert actual.sharding.spec == P(None, None)


def test_stack_sharded_target_uses_replica_data_expert_when_divisible():
    mesh = AbstractMesh(
        axis_sizes=(4, 4, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    param = jax.ShapeDtypeStruct(
        (256, 2560, 2560),
        jnp.float32,
        sharding=NamedSharding(mesh, P("expert", "data", "model")),
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        target_pspec = _batch_sharded_stack_target_pspec(param)

    assert target_pspec == P(("replica_dcn", "data", "expert"), None, None)


def test_grouped_4d_target_shards_group_over_data_and_stack_over_expert():
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    update = jax.ShapeDtypeStruct(
        (2, 8, 16, 16),
        jnp.float32,
        sharding=NamedSharding(mesh, P(None, "expert", None, None)),
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        target_pspec = _grouped_4d_stack_target_pspec(update)

    assert target_pspec == P("data", "expert", None, None)


def test_grouped_4d_target_shards_group_over_replica_and_data_when_divisible():
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    update = jax.ShapeDtypeStruct(
        (4, 8, 16, 16),
        jnp.float32,
        sharding=NamedSharding(mesh, P(None, "expert", None, None)),
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        target_pspec = _grouped_4d_stack_target_pspec(update)

    assert target_pspec == P(("replica_dcn", "data"), "expert", None, None)


def test_grouped_4d_target_uses_named_sharding_mesh_without_abstract_mesh():
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    update = jax.ShapeDtypeStruct(
        (2, 8, 16, 16),
        jnp.float32,
        sharding=NamedSharding(mesh, P(None, "expert", None, None)),
    )

    with _reset_abstract_mesh():
        target_pspec = _grouped_4d_stack_target_pspec(update)

    assert target_pspec == P("data", "expert", None, None)


def test_grouped_4d_target_returns_named_sharding_without_abstract_mesh():
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    update = jax.ShapeDtypeStruct(
        (2, 8, 16, 16),
        jnp.float32,
        sharding=NamedSharding(mesh, P(None, "expert", None, None)),
    )

    with _reset_abstract_mesh():
        target = _grouped_4d_stack_target(update)

    assert isinstance(target, NamedSharding)
    assert target.mesh is mesh
    assert target.spec == P("data", "expert", None, None)


def test_stack_sharded_target_uses_replica_data_expert_when_padding_is_small():
    mesh = AbstractMesh(
        axis_sizes=(16, 4, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    update = jax.ShapeDtypeStruct(
        (832, 2560, 2560),
        jnp.float32,
        sharding=NamedSharding(mesh, P(("replica_dcn", "expert"), None, None)),
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        target_pspec = _batch_sharded_stack_target_pspec(update)

    assert target_pspec == P(("replica_dcn", "data", "expert"), None, None)


def test_stack_sharded_target_prefers_widest_divisible_stack_axis():
    mesh = AbstractMesh(
        axis_sizes=(2, 8, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    update = jax.ShapeDtypeStruct(
        (256, 2560, 2560),
        jnp.float32,
        sharding=NamedSharding(mesh, P(("replica_dcn", "expert"), "data", "model")),
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        target_pspec = _batch_sharded_stack_target_pspec(update)

    assert target_pspec == P(("replica_dcn", "data", "expert"), None, None)


def test_stack_sharded_target_falls_back_when_full_axis_padding_is_too_large():
    mesh = AbstractMesh(
        axis_sizes=(16, 4, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    update = jax.ShapeDtypeStruct(
        (300, 2560, 2560),
        jnp.float32,
        sharding=NamedSharding(mesh, P("expert", "data", "model")),
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        target_pspec = _batch_sharded_stack_target_pspec(update)

    assert target_pspec == P(("replica_dcn", "data"), None, None)


def test_stack_batch_sharding_assertion_rejects_lost_stack_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    expected_pspec = P("expert", None, None)
    sharded = jax.ShapeDtypeStruct(
        (256, 2560, 2560),
        jnp.float32,
        sharding=NamedSharding(mesh, expected_pspec),
    )
    replicated = jax.ShapeDtypeStruct(
        (256, 2560, 2560),
        jnp.float32,
        sharding=NamedSharding(mesh, P(None, None, None)),
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        _assert_stack_batch_sharded(sharded, expected_pspec, "test")
        with pytest.raises(AssertionError, match="expected stacked Muon sharding"):
            _assert_stack_batch_sharded(replicated, expected_pspec, "test")


def test_grug_scale_with_muon_uses_momentum_sharding_callback():
    mesh = AbstractMesh(
        axis_sizes=(4, 4, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = {
        "expert": jax.ShapeDtypeStruct(
            (256, 2560, 5120),
            jnp.float32,
            sharding=NamedSharding(mesh, P("expert", "data", "model")),
        )
    }
    target_sharding = NamedSharding(mesh, P(("replica_dcn", "expert"), "data", "model"))
    transform = _grug_scale_with_muon(momentum_sharding_fn=lambda _: target_sharding)

    state = jax.eval_shape(transform.init, params)

    assert state.momentum_buffer["expert"].sharding == target_sharding


def test_grug_scale_with_muon_outputs_param_sharding():
    mesh = AbstractMesh(
        axis_sizes=(4, 4, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = {
        "dense": jax.ShapeDtypeStruct(
            (2560, 2560),
            jnp.float32,
            sharding=NamedSharding(mesh, P("data", "model")),
        ),
        "expert": jax.ShapeDtypeStruct(
            (256, 2560, 2560),
            jnp.float32,
            sharding=NamedSharding(mesh, P("expert", "data", "model")),
        ),
    }
    updates = params
    transform = _grug_scale_with_muon(momentum=0.0, nesterov=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(transform.init, params)
        next_updates, _ = jax.eval_shape(transform.update, updates, state, params)

    assert next_updates["dense"].sharding.spec == P("data", "model")
    assert next_updates["expert"].sharding.spec == P("expert", "data", "model")


def test_grug_scale_with_muon_grouped_stack_outputs_param_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = {
        "layer0": {
            "expert": jax.ShapeDtypeStruct(
                (256, 2560, 5120),
                jnp.float32,
                sharding=NamedSharding(mesh, P("expert", None, None)),
            ),
        },
        "layer1": {
            "expert": jax.ShapeDtypeStruct(
                (256, 2560, 5120),
                jnp.float32,
                sharding=NamedSharding(mesh, P("expert", None, None)),
            ),
        },
    }
    transform = _grug_scale_with_muon(momentum=0.0, nesterov=False)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(transform.init, params)
        next_updates, _ = jax.eval_shape(transform.update, params, state, params)

    assert next_updates["layer0"]["expert"].sharding.spec == P("expert", None, None)
    assert next_updates["layer1"]["expert"].sharding.spec == P("expert", None, None)


def test_grug_scale_with_muon_grouped_4d_outputs_param_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = {
        "layer0": {
            "expert": jax.ShapeDtypeStruct(
                (8, 16, 16),
                jnp.float32,
                sharding=NamedSharding(mesh, P("expert", None, None)),
            ),
        },
        "layer1": {
            "expert": jax.ShapeDtypeStruct(
                (8, 16, 16),
                jnp.float32,
                sharding=NamedSharding(mesh, P("expert", None, None)),
            ),
        },
    }
    transform = _grug_scale_with_muon(
        momentum=0.0,
        nesterov=False,
        orthogonalization_layout=STACK_BATCH_4D_SHARDED,
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(transform.init, params)
        next_updates, _ = jax.eval_shape(transform.update, params, state, params)

    assert next_updates["layer0"]["expert"].shape == (8, 16, 16)
    assert next_updates["layer1"]["expert"].shape == (8, 16, 16)
    assert next_updates["layer0"]["expert"].sharding.spec == P("expert", None, None)
    assert next_updates["layer1"]["expert"].sharding.spec == P("expert", None, None)


def test_grouped_stack_padding_slices_back_before_restoring_param_sharding():
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = {
        "layer0": {
            "expert": jax.ShapeDtypeStruct(
                (2, 8, 16),
                jnp.float32,
                sharding=NamedSharding(mesh, P(None, None, None)),
            ),
        },
        "layer1": {
            "expert": jax.ShapeDtypeStruct(
                (2, 8, 16),
                jnp.float32,
                sharding=NamedSharding(mesh, P(None, None, None)),
            ),
        },
        "layer2": {
            "expert": jax.ShapeDtypeStruct(
                (3, 8, 16),
                jnp.float32,
                sharding=NamedSharding(mesh, P(None, None, None)),
            ),
        },
    }
    transform = _grug_scale_with_muon(momentum=0.0, nesterov=False, max_grouped_stack_size=8)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(transform.init, params)
        next_updates, _ = jax.eval_shape(transform.update, params, state, params)

    assert next_updates["layer0"]["expert"].shape == (2, 8, 16)
    assert next_updates["layer1"]["expert"].shape == (2, 8, 16)
    assert next_updates["layer2"]["expert"].shape == (3, 8, 16)
    assert next_updates["layer0"]["expert"].sharding.spec == P(None, None, None)
    assert next_updates["layer1"]["expert"].sharding.spec == P(None, None, None)
    assert next_updates["layer2"]["expert"].sharding.spec == P(None, None, None)
