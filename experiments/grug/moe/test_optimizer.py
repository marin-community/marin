# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.adamh import _scale_invariant_hyperball_update as _adamh_hyperball_update
from experiments.grug.moe.launch_cw_may_d2560 import build_may_optimizer
from experiments.grug.moe.optimizer import (
    GroupedMuonHState,
    GrugMoeAdamHConfig,
    GrugMoeMuonHConfig,
    GrugMoeSgdConfig,
    _expert_momentum_sharding,
    _restore_grouped_muonh_for_split_target_layout,
    _scale_invariant_hyperball_updates,
    scale_with_grouped_expert_muonh,
)
from experiments.grug.moe.optimizer_sharding import assert_update_sharding_matches_params


def test_grug_moe_adamh_mask_routes_expert_mlp_weights_to_expert_group():
    params = {
        "blocks": {
            "0": {
                "attn": {
                    "attn_gate": jnp.ones((8, 2), dtype=jnp.float32),
                },
                "attn_gated_norm": {
                    "w_down": jnp.ones((8, 4), dtype=jnp.float32),
                },
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

    mask = GrugMoeAdamHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["attn_gated_norm"]["w_down"] == "adamh"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "adamh_expert"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh_expert"
    assert block_mask["shared"]["w_gate"] == "adamh_expert"
    assert mask["token_embed"] == "adam"


def test_grug_moe_muonh_mask_routes_may_recipe_groups():
    params = {
        "blocks": {
            "0": {
                "attn": {
                    "w_q": jnp.ones((8, 8), dtype=jnp.float32),
                    "attn_gate": jnp.ones((8, 2), dtype=jnp.float32),
                },
                "attn_gated_norm": {
                    "w_down": jnp.ones((8, 4), dtype=jnp.float32),
                },
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
        "output_proj": jnp.ones((8, 128), dtype=jnp.float32),
    }

    mask = GrugMoeMuonHConfig().create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["attn"]["w_q"] == "muonh"
    assert block_mask["attn"]["attn_gate"] == "adam"
    assert block_mask["attn_gated_norm"]["w_down"] == "muonh"
    assert block_mask["mlp"]["router"] == "adam"
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "muonh"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "muonh"
    assert block_mask["shared"]["w_gate"] == "muonh"
    assert mask["token_embed"] == "adam"
    assert mask["output_proj"] == "adamh"


def test_grug_moe_muonh_can_route_routed_expert_weights_to_adamh():
    params = {
        "blocks": {
            "0": {
                "mlp": {
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
    }

    mask = GrugMoeMuonHConfig(expert_3d_optimizer="adamh").create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "adamh"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "adamh"
    assert block_mask["shared"]["w_gate"] == "muonh"


def test_grug_moe_muonh_can_route_routed_expert_weights_to_grouped_muonh():
    params = {
        "blocks": {
            "0": {
                "mlp": {
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
    }

    mask = GrugMoeMuonHConfig(expert_3d_optimizer="grouped_muonh").create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "grouped_muonh"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "grouped_muonh"
    assert block_mask["shared"]["w_gate"] == "muonh"


def test_grug_moe_muonh_can_route_ordinary_2d_weights_away_from_muonh():
    params = {
        "blocks": {
            "0": {
                "attn": {
                    "w_q": jnp.ones((8, 8), dtype=jnp.float32),
                },
                "attn_gated_norm": {
                    "w_down": jnp.ones((8, 8), dtype=jnp.float32),
                },
                "mlp": {
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
    }

    mask = GrugMoeMuonHConfig(
        expert_3d_optimizer="grouped_muonh",
        ordinary_2d_optimizer="sgd",
    ).create_mask(params)

    block_mask = mask["blocks"]["0"]
    assert block_mask["attn"]["w_q"] == "sgd"
    assert block_mask["attn_gated_norm"]["w_down"] == "sgd"
    assert block_mask["mlp"]["expert_mlp"]["w_gate_up"] == "grouped_muonh"
    assert block_mask["mlp"]["expert_mlp"]["w_down"] == "grouped_muonh"
    assert block_mask["shared"]["w_gate"] == "sgd"


def test_grouped_expert_muonh_rejects_unknown_ns_compute_dtype():
    with pytest.raises(ValueError, match="ns_compute_dtype"):
        scale_with_grouped_expert_muonh(ns_compute_dtype="int8")


def test_may_optimizer_reads_muon_nesterov_env(monkeypatch):
    monkeypatch.setenv("MAY_OPTIMIZER", "muonh")
    monkeypatch.setenv("MAY_MUON_NESTEROV", "false")

    optimizer = build_may_optimizer(batch_size=8, seq_len=4096)

    assert isinstance(optimizer, GrugMoeMuonHConfig)
    assert optimizer.nesterov is False


def test_may_optimizer_reads_grouped_muonh_packed_entry_env(monkeypatch):
    monkeypatch.setenv("MAY_OPTIMIZER", "muonh")
    monkeypatch.setenv("MAY_EXPERT_3D_OPTIMIZER", "grouped_muonh")
    monkeypatch.setenv("MAY_EXPERT_GROUPED_MUONH_GROUP_SIZE", "4")
    monkeypatch.setenv("MAY_EXPERT_GROUPED_MUONH_PACKED_ENTRY", "true")
    monkeypatch.setenv("MAY_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE", "true")
    monkeypatch.setenv("MAY_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES", "true")

    optimizer = build_may_optimizer(batch_size=8, seq_len=4096)

    assert isinstance(optimizer, GrugMoeMuonHConfig)
    assert optimizer.expert_3d_optimizer == "grouped_muonh"
    assert optimizer.expert_grouped_muonh_group_size == 4
    assert optimizer.expert_grouped_muonh_packed_entry is True
    assert optimizer.expert_grouped_muonh_packed_bank_compute is True
    assert optimizer.expert_grouped_muonh_chunk_local_boundaries is True


def test_may_grouped_muonh_defaults_to_packed_entry(monkeypatch):
    monkeypatch.setenv("MAY_OPTIMIZER", "muonh")
    monkeypatch.setenv("MAY_EXPERT_3D_OPTIMIZER", "grouped_muonh")

    optimizer = build_may_optimizer(batch_size=8, seq_len=4096)

    assert isinstance(optimizer, GrugMoeMuonHConfig)
    assert optimizer.expert_3d_optimizer == "grouped_muonh"
    assert optimizer.expert_grouped_muonh_packed_entry is True
    assert optimizer.expert_grouped_muonh_chunk_local_boundaries is False


def test_grug_moe_sgd_update_is_stateless_and_matches_shapes():
    params = {
        "matrix": jnp.ones((4, 8), dtype=jnp.bfloat16),
        "vector": jnp.ones((8,), dtype=jnp.bfloat16),
    }
    grads = jax.tree.map(lambda x: jnp.full_like(x, 0.25), params)
    optimizer = GrugMoeSgdConfig(learning_rate=0.1, lr_schedule="constant").build(num_train_steps=8)

    opt_state = optimizer.init(params)
    updates, next_state = optimizer.update(grads, opt_state, params)

    state_leaves = jax.tree.leaves((opt_state, next_state))
    assert all(not hasattr(leaf, "shape") or leaf.shape == () for leaf in state_leaves)
    assert updates["matrix"].shape == params["matrix"].shape
    assert updates["vector"].shape == params["vector"].shape
    assert jnp.allclose(updates["matrix"], -0.025).item()
    assert jnp.allclose(updates["vector"], -0.025).item()


def test_grouped_expert_muonh_optimizer_returns_fsdp_updates_before_apply():
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    gate_sharding = NamedSharding(mesh, P("expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P("expert", "model", "data"))

    def block_specs():
        return {
            "mlp": {
                "expert_mlp": {
                    "w_gate_up": jax.ShapeDtypeStruct((8, 16, 32), jnp.bfloat16, sharding=gate_sharding),
                    "w_down": jax.ShapeDtypeStruct((8, 32, 16), jnp.bfloat16, sharding=down_sharding),
                },
            },
        }

    params = {"blocks": tuple(block_specs() for _ in range(4))}
    grads = jax.tree.map(lambda param: jax.ShapeDtypeStruct(param.shape, param.dtype, sharding=param.sharding), params)
    transform = scale_with_grouped_expert_muonh(
        learning_rate=0.02,
        steps=1,
        expert_grouped_muonh_group_size=4,
        max_grouped_stack_size=8,
    )

    def init_grouped_state(params):
        return transform.init(params)

    def transform_update_step(grads, state, params):
        return transform.update(grads, state, params)

    with use_abstract_mesh(mesh):
        grouped_state = jax.eval_shape(init_grouped_state, params)

    assert isinstance(grouped_state, GroupedMuonHState)
    assert len(grouped_state.trace_groups) == 2
    assert all(
        trace_group.sharding.spec == P(("replica_dcn", "data"), "expert", None, None)
        for trace_group in grouped_state.trace_groups
    )

    with use_abstract_mesh(mesh):
        transform_updates, next_grouped_state = jax.eval_shape(transform_update_step, grads, grouped_state, params)

    assert_update_sharding_matches_params(transform_updates, params, "grouped MuonH direct transform updates")
    assert isinstance(next_grouped_state, GroupedMuonHState)
    assert all(
        trace_group.sharding.spec == P(("replica_dcn", "data"), "expert", None, None)
        for trace_group in next_grouped_state.trace_groups
    )

    optimizer = GrugMoeMuonHConfig(
        learning_rate=0.02,
        lr_schedule="constant",
        backend_steps=1,
        expert_3d_optimizer="grouped_muonh",
        expert_grouped_muonh_group_size=4,
        expert_grouped_muonh_packed_entry=False,
        max_grouped_stack_size=8,
        max_grad_norm=None,
    ).build(num_train_steps=8)

    def update_step(params, grads, opt_state):
        updates, _ = optimizer.update(grads, opt_state, params)
        return updates

    with use_abstract_mesh(mesh):
        opt_state = jax.eval_shape(optimizer.init, params)
        updates = jax.eval_shape(update_step, params, grads, opt_state)
        update_step_jit = jax.jit(update_step)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step_jit.trace(params, grads, opt_state).lower(lowering_platforms=(platform,))

    assert_update_sharding_matches_params(updates, params, "grouped MuonH production updates")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert hlo.count("stablehlo.all_reduce") == 0
    assert hlo.count("stablehlo.reduce_scatter") == 0
    assert hlo.count("stablehlo.all_to_all") == 2
    assert hlo.count("stablehlo.all_gather") == 6


def test_grouped_expert_muonh_packs_multi_chunk_restore_boundary():
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    gate_sharding = NamedSharding(mesh, P("expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P("expert", "model", "data"))

    def block_specs():
        return {
            "mlp": {
                "expert_mlp": {
                    "w_gate_up": jax.ShapeDtypeStruct((8, 16, 32), jnp.bfloat16, sharding=gate_sharding),
                    "w_down": jax.ShapeDtypeStruct((8, 32, 16), jnp.bfloat16, sharding=down_sharding),
                },
            },
        }

    params = {"blocks": tuple(block_specs() for _ in range(5))}
    grads = jax.tree.map(lambda param: jax.ShapeDtypeStruct(param.shape, param.dtype, sharding=param.sharding), params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=0.02,
        lr_schedule="constant",
        backend_steps=1,
        expert_3d_optimizer="grouped_muonh",
        expert_grouped_muonh_group_size=4,
        expert_grouped_muonh_packed_entry=False,
        max_grouped_stack_size=8,
        max_grad_norm=None,
    ).build(num_train_steps=8)

    def update_step(params, grads, opt_state):
        updates, _ = optimizer.update(grads, opt_state, params)
        return updates

    with use_abstract_mesh(mesh):
        opt_state = jax.eval_shape(optimizer.init, params)
        updates = jax.eval_shape(update_step, params, grads, opt_state)
        update_step_jit = jax.jit(update_step)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step_jit.trace(params, grads, opt_state).lower(lowering_platforms=(platform,))

    assert_update_sharding_matches_params(updates, params, "grouped MuonH multi-chunk production updates")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert hlo.count("stablehlo.all_reduce") == 0
    assert hlo.count("stablehlo.reduce_scatter") == 0
    assert hlo.count("stablehlo.all_to_all") == 2
    assert hlo.count("stablehlo.all_gather") == 10


def test_grouped_expert_muonh_packed_entry_boundary_is_explicit():
    mesh = AbstractMesh(
        axis_sizes=(1, 4, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    gate_sharding = NamedSharding(mesh, P("expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P("expert", "model", "data"))

    def block_specs():
        return {
            "mlp": {
                "expert_mlp": {
                    "w_gate_up": jax.ShapeDtypeStruct((8, 16, 32), jnp.bfloat16, sharding=gate_sharding),
                    "w_down": jax.ShapeDtypeStruct((8, 32, 16), jnp.bfloat16, sharding=down_sharding),
                },
            },
        }

    params = {"blocks": tuple(block_specs() for _ in range(5))}
    grads = jax.tree.map(lambda param: jax.ShapeDtypeStruct(param.shape, param.dtype, sharding=param.sharding), params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=0.02,
        lr_schedule="constant",
        backend_steps=1,
        expert_3d_optimizer="grouped_muonh",
        expert_grouped_muonh_group_size=4,
        expert_grouped_muonh_packed_entry=True,
        max_grouped_stack_size=8,
        max_grad_norm=None,
    ).build(num_train_steps=8)

    def update_step(params, grads, opt_state):
        updates, _ = optimizer.update(grads, opt_state, params)
        return updates

    with use_abstract_mesh(mesh):
        opt_state = jax.eval_shape(optimizer.init, params)
        updates = jax.eval_shape(update_step, params, grads, opt_state)
        update_step_jit = jax.jit(update_step)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step_jit.trace(params, grads, opt_state).lower(lowering_platforms=(platform,))

    assert_update_sharding_matches_params(updates, params, "grouped MuonH packed-entry production updates")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert hlo.count("stablehlo.all_reduce") == 0
    assert hlo.count("stablehlo.reduce_scatter") == 0
    assert hlo.count("stablehlo.all_to_all") == 6
    assert hlo.count("stablehlo.all_gather") == 0


def test_grouped_expert_muonh_packed_entry_r4_boundary_is_explicit():
    mesh = AbstractMesh(
        axis_sizes=(4, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    gate_sharding = NamedSharding(mesh, P("expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P("expert", "model", "data"))

    def block_specs():
        return {
            "mlp": {
                "expert_mlp": {
                    "w_gate_up": jax.ShapeDtypeStruct((8, 16, 32), jnp.bfloat16, sharding=gate_sharding),
                    "w_down": jax.ShapeDtypeStruct((8, 32, 16), jnp.bfloat16, sharding=down_sharding),
                },
            },
        }

    params = {"blocks": tuple(block_specs() for _ in range(5))}
    grads = jax.tree.map(lambda param: jax.ShapeDtypeStruct(param.shape, param.dtype, sharding=param.sharding), params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=0.02,
        lr_schedule="constant",
        backend_steps=1,
        expert_3d_optimizer="grouped_muonh",
        expert_grouped_muonh_group_size=4,
        expert_grouped_muonh_packed_entry=True,
        max_grouped_stack_size=8,
        max_grad_norm=None,
    ).build(num_train_steps=8)

    def update_step(params, grads, opt_state):
        updates, _ = optimizer.update(grads, opt_state, params)
        return updates

    with use_abstract_mesh(mesh):
        opt_state = jax.eval_shape(optimizer.init, params)
        updates = jax.eval_shape(update_step, params, grads, opt_state)
        update_step_jit = jax.jit(update_step)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step_jit.trace(params, grads, opt_state).lower(lowering_platforms=(platform,))

    assert_update_sharding_matches_params(updates, params, "grouped MuonH R4 packed-entry production updates")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert hlo.count("stablehlo.all_reduce") == 0
    assert hlo.count("stablehlo.reduce_scatter") == 0
    assert hlo.count("stablehlo.all_to_all") == 0
    assert hlo.count("stablehlo.all_gather") == 2


def test_grouped_expert_muonh_packed_entry_r2_boundary_does_not_duplicate_replica_gather():
    mesh = AbstractMesh(
        axis_sizes=(2, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    gate_sharding = NamedSharding(mesh, P("expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P("expert", "model", "data"))

    def block_specs():
        return {
            "mlp": {
                "expert_mlp": {
                    "w_gate_up": jax.ShapeDtypeStruct((8, 16, 32), jnp.bfloat16, sharding=gate_sharding),
                    "w_down": jax.ShapeDtypeStruct((8, 32, 16), jnp.bfloat16, sharding=down_sharding),
                },
            },
        }

    params = {"blocks": tuple(block_specs() for _ in range(5))}
    grads = jax.tree.map(lambda param: jax.ShapeDtypeStruct(param.shape, param.dtype, sharding=param.sharding), params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=0.02,
        lr_schedule="constant",
        backend_steps=1,
        expert_3d_optimizer="grouped_muonh",
        expert_grouped_muonh_group_size=2,
        expert_grouped_muonh_packed_entry=True,
        max_grouped_stack_size=8,
        max_grad_norm=None,
    ).build(num_train_steps=8)

    def update_step(params, grads, opt_state):
        updates, _ = optimizer.update(grads, opt_state, params)
        return updates

    with use_abstract_mesh(mesh):
        opt_state = jax.eval_shape(optimizer.init, params)
        updates = jax.eval_shape(update_step, params, grads, opt_state)
        update_step_jit = jax.jit(update_step)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step_jit.trace(params, grads, opt_state).lower(lowering_platforms=(platform,))

    assert_update_sharding_matches_params(updates, params, "grouped MuonH R2 packed-entry production updates")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert hlo.count("stablehlo.all_reduce") == 0
    assert hlo.count("stablehlo.reduce_scatter") == 0
    assert hlo.count("stablehlo.all_to_all") == 0
    assert hlo.count("stablehlo.all_gather") == 2


def test_grouped_expert_muonh_packed_bank_compute_avoids_chunk_slice_boundaries():
    mesh = AbstractMesh(
        axis_sizes=(2, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    gate_sharding = NamedSharding(mesh, P("expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P("expert", "model", "data"))

    def block_specs():
        return {
            "mlp": {
                "expert_mlp": {
                    "w_gate_up": jax.ShapeDtypeStruct((8, 16, 32), jnp.bfloat16, sharding=gate_sharding),
                    "w_down": jax.ShapeDtypeStruct((8, 32, 16), jnp.bfloat16, sharding=down_sharding),
                },
            },
        }

    params = {"blocks": tuple(block_specs() for _ in range(6))}
    grads = jax.tree.map(lambda param: jax.ShapeDtypeStruct(param.shape, param.dtype, sharding=param.sharding), params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=0.02,
        lr_schedule="constant",
        backend_steps=1,
        expert_3d_optimizer="grouped_muonh",
        expert_grouped_muonh_group_size=2,
        expert_grouped_muonh_packed_entry=True,
        expert_grouped_muonh_packed_bank_compute=True,
        max_grouped_stack_size=8,
        max_grad_norm=None,
    ).build(num_train_steps=8)

    def update_step(params, grads, opt_state):
        updates, next_state = optimizer.update(grads, opt_state, params)
        return updates, next_state

    with use_abstract_mesh(mesh):
        opt_state = jax.eval_shape(optimizer.init, params)
        updates, next_state = jax.eval_shape(update_step, params, grads, opt_state)
        update_step_jit = jax.jit(update_step)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step_jit.trace(params, grads, opt_state).lower(lowering_platforms=(platform,))

    grouped_state = opt_state.inner_state.inner_states["grouped_muonh"].inner_state[0]
    next_grouped_state = next_state.inner_state.inner_states["grouped_muonh"].inner_state[0]
    assert isinstance(grouped_state, GroupedMuonHState)
    assert len(grouped_state.trace_groups) == 2
    assert isinstance(next_grouped_state, GroupedMuonHState)
    assert len(next_grouped_state.trace_groups) == 2
    assert_update_sharding_matches_params(updates, params, "packed-bank grouped MuonH production updates")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert "slice_update_chunk" not in hlo
    assert "slice_param_chunk" not in hlo
    assert "concat_chunks" not in hlo


def test_grouped_expert_muonh_chunk_local_boundary_is_explicit():
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    gate_sharding = NamedSharding(mesh, P("expert", "data", "model"))
    down_sharding = NamedSharding(mesh, P("expert", "model", "data"))

    def block_specs():
        return {
            "mlp": {
                "expert_mlp": {
                    "w_gate_up": jax.ShapeDtypeStruct((8, 16, 32), jnp.bfloat16, sharding=gate_sharding),
                    "w_down": jax.ShapeDtypeStruct((8, 32, 16), jnp.bfloat16, sharding=down_sharding),
                },
            },
        }

    params = {"blocks": tuple(block_specs() for _ in range(5))}
    grads = jax.tree.map(lambda param: jax.ShapeDtypeStruct(param.shape, param.dtype, sharding=param.sharding), params)
    optimizer = GrugMoeMuonHConfig(
        learning_rate=0.02,
        lr_schedule="constant",
        backend_steps=1,
        expert_3d_optimizer="grouped_muonh",
        expert_grouped_muonh_group_size=4,
        expert_grouped_muonh_chunk_local_boundaries=True,
        max_grouped_stack_size=8,
        max_grad_norm=None,
    ).build(num_train_steps=8)

    def update_step(params, grads, opt_state):
        updates, _ = optimizer.update(grads, opt_state, params)
        return updates

    with use_abstract_mesh(mesh):
        opt_state = jax.eval_shape(optimizer.init, params)
        updates = jax.eval_shape(update_step, params, grads, opt_state)
        update_step_jit = jax.jit(update_step)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step_jit.trace(params, grads, opt_state).lower(lowering_platforms=(platform,))

    assert_update_sharding_matches_params(updates, params, "grouped MuonH chunk-local production updates")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    assert hlo.count("stablehlo.all_reduce") == 0
    assert hlo.count("stablehlo.reduce_scatter") == 0
    assert hlo.count("stablehlo.all_to_all") == 8
    assert hlo.count("stablehlo.all_gather") == 4


def test_grouped_muonh_restore_can_reshard_grouped_stack_to_fsdp_layout():
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    param = jax.ShapeDtypeStruct(
        (8, 16, 32),
        jnp.bfloat16,
        sharding=NamedSharding(mesh, P("expert", "data", "model")),
    )
    grouped_update = jax.ShapeDtypeStruct(
        (4, 8, 16, 32),
        jnp.bfloat16,
        sharding=NamedSharding(mesh, P(("replica_dcn", "data"), "expert", None, None)),
    )
    target = grouped_update.sharding

    def restore(update):
        return _restore_grouped_muonh_for_split_target_layout(update, target, valid_size=4, sample_param=param)

    with use_abstract_mesh(mesh):
        restored = jax.eval_shape(restore, grouped_update)

    assert restored.shape == grouped_update.shape
    assert restored.sharding.spec == P(None, "expert", "data", "model")


def test_expert_momentum_sharding_uses_replica_dcn_on_expert_stack_axis():
    mesh = AbstractMesh(
        axis_sizes=(4, 4, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    w_gate_up = jax.ShapeDtypeStruct(
        (256, 2560, 5120),
        jnp.float32,
        sharding=NamedSharding(mesh, P("expert", "data", "model")),
    )
    w_down = jax.ShapeDtypeStruct(
        (256, 5120, 2560),
        jnp.float32,
        sharding=NamedSharding(mesh, P("expert", "model", "data")),
    )

    assert _expert_momentum_sharding(w_gate_up).spec == P(("replica_dcn", "expert"), "data", "model")
    assert _expert_momentum_sharding(w_down).spec == P(("replica_dcn", "expert"), "model", "data")


def test_optimizer_sharding_assertion_accepts_matching_expert_pspecs():
    mesh = AbstractMesh(
        axis_sizes=(2, 8, 8, 1),
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
    updates = {
        "expert": jax.ShapeDtypeStruct(
            (256, 2560, 5120),
            jnp.float32,
            sharding=NamedSharding(mesh, P("expert", "data", "model")),
        )
    }

    assert_update_sharding_matches_params(updates, params, "test")


def test_optimizer_sharding_assertion_rejects_lost_expert_axis():
    mesh = AbstractMesh(
        axis_sizes=(2, 8, 8, 1),
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
    updates = {
        "expert": jax.ShapeDtypeStruct(
            (256, 2560, 5120),
            jnp.float32,
            sharding=NamedSharding(mesh, P("data", None, None)),
        )
    }

    with pytest.raises(AssertionError, match="lost expert-axis sharding"):
        assert_update_sharding_matches_params(updates, params, "test")


def test_scale_invariant_hyperball_updates_matches_materialized_formula():
    params = {
        "vector": jnp.arange(6, dtype=jnp.float32) / 5 + 0.3,
        "matrix": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7 + 0.25,
        "stack": jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 11 + 0.5,
    }
    updates = {
        "vector": jnp.arange(6, dtype=jnp.float32) / 9 + 0.05,
        "matrix": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 13 + 0.1,
        "stack": jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 17 + 0.2,
    }
    learning_rate = 0.03

    def materialized_update(param, update):
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    actual = _scale_invariant_hyperball_updates(params, updates, learning_rate)
    expected = jax.tree.map(materialized_update, params, updates)

    assert jnp.allclose(actual["vector"], expected["vector"], rtol=1e-5, atol=1e-6).item()
    assert jnp.allclose(actual["matrix"], expected["matrix"], rtol=1e-5, atol=1e-6).item()
    assert jnp.allclose(actual["stack"], expected["stack"], rtol=1e-5, atol=1e-6).item()


def test_adamh_hyperball_update_matches_materialized_formula():
    params = {
        "vector": jnp.arange(6, dtype=jnp.float32) / 5 + 0.3,
        "matrix": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7 + 0.25,
        "stack": jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 11 + 0.5,
    }
    updates = {
        "vector": jnp.arange(6, dtype=jnp.float32) / 9 + 0.05,
        "matrix": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 13 + 0.1,
        "stack": jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 17 + 0.2,
    }
    learning_rate = 0.03

    def materialized_update(param, update):
        def materialized_2d(p, u):
            p_norm = jnp.linalg.norm(p)
            u_norm = jnp.linalg.norm(u)
            new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, 1e-10)
            return new_p / jnp.linalg.norm(new_p) * p_norm - p

        if param.ndim <= 2:
            return materialized_2d(param, update)
        return jax.vmap(materialized_2d)(param, update)

    actual = jax.tree.map(
        lambda param, update: _adamh_hyperball_update(param, update, learning_rate),
        params,
        updates,
    )
    expected = jax.tree.map(materialized_update, params, updates)

    assert jnp.allclose(actual["vector"], expected["vector"], rtol=1e-5, atol=1e-6).item()
    assert jnp.allclose(actual["matrix"], expected["matrix"], rtol=1e-5, atol=1e-6).item()
    assert jnp.allclose(actual["stack"], expected["stack"], rtol=1e-5, atol=1e-6).item()
