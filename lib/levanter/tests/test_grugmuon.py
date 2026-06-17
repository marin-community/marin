# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P, use_abstract_mesh

from levanter.optim.grugmuon import (
    GrugMuonConfig,
    STACK_BATCH_SHARDED,
    VMAP_REPLICATED,
    _batch_sharded_stack_target_pspec,
    _grug_scale_with_muon,
    _zeropower_via_newtonschulz_batched_stack_sharded,
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


def test_stack_sharded_target_preserves_param_stack_axis_sharding():
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

    assert target_pspec == P("expert", None, None)


def test_stack_sharded_target_preserves_update_replica_expert_stack_axis_sharding():
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

    assert target_pspec == P(("replica_dcn", "expert"), None, None)


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
