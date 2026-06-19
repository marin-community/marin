# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import jax
import jax.numpy as jnp
import optax
import pytest
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.muon_update_bench import (
    EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
    EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_APPLY_ONLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
    FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    NS4D_DATA_GROUP_APPLY_BENCH,
    ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
    ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH,
    ORDINARY_2D_GROUPED_STACK_NS_BENCH,
    ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    BenchConfig,
    _stacked_2d_target,
    assert_expert_fsdp_sharding,
    assert_ns4d_sharding,
    build_full_production_muonh_optimizer,
    build_grouped_expert_productionish_optimizer,
    build_ordinary_2d_muonh_optimizer,
    estimate_grouped_2d_muonh,
    estimate_grouping,
    estimated_full_production_muonh_ns_dot_flops,
    estimated_matrix_count,
    estimated_ns_dot_flops,
    expert_fsdp_grouped_apply_boundary_step_factory,
    expert_fsdp_grouped_muonh_optimizer_apply_step_factory,
    expert_fsdp_grouped_restore_boundary_step_factory,
    expert_fsdp_grouped_target_restore_boundary_step_factory,
    expert_fsdp_grouped_updates_muonh_apply_step_factory,
    expert_fsdp_grouped_updates_muonh_updates_step_factory,
    full_production_grouped_2d_persistent_apply_timing_step_factory,
    full_production_muonh_mask,
    full_production_muonh_optimizer_apply_step_factory,
    grouped_2d_restore_split_step_factory,
    grouped_2d_stack_ns_step_factory,
    grouped_3d_hyperball_update,
    grouped_4d_hyperball_update,
    grouped_expert_apply_boundary_step_factory,
    grouped_expert_optimizer_apply_step_factory,
    ns4d_compute_sharding,
    ns4d_grouped_apply_step_factory,
    ns4d_input_sharding,
    ns4d_result_sharding,
    ordinary_2d_grouped_persistent_apply_timing_step_factory,
    ordinary_2d_muonh_optimizer_apply_step_factory,
    persistent_grouped_2d_metadata_from_specs,
    summarize_hlo,
    summary_row,
    synthetic_fsdp_expert_specs,
    synthetic_full_production_grouped_persistent_specs,
    synthetic_full_production_muonh_specs,
    synthetic_grouped_expert_specs,
    synthetic_ns4d_specs,
    synthetic_ordinary_2d_grouped_persistent_specs,
    synthetic_ordinary_2d_muonh_specs,
    synthetic_productionish_grouped_expert_specs,
)


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def test_ns4d_grouped_apply_preserves_data_expert_sharding_through_apply_boundary():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=2,
        expert_axis=4,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    input_sharding = ns4d_input_sharding(mesh, config, NS4D_DATA_GROUP_APPLY_BENCH)
    compute_sharding = ns4d_compute_sharding(mesh, config, NS4D_DATA_GROUP_APPLY_BENCH)
    result_sharding = ns4d_result_sharding(mesh, config, NS4D_DATA_GROUP_APPLY_BENCH)
    params = synthetic_ns4d_specs(mesh, config, NS4D_DATA_GROUP_APPLY_BENCH)
    updates = synthetic_ns4d_specs(mesh, config, NS4D_DATA_GROUP_APPLY_BENCH)

    assert isinstance(input_sharding, NamedSharding)
    assert input_sharding.spec == P("data", "expert", None, None)
    assert compute_sharding.spec == P("data", "expert", None, None)
    assert result_sharding is not None
    assert result_sharding.spec == P("data", "expert", None, None)
    assert_ns4d_sharding(params, input_sharding.spec, "params")
    assert_ns4d_sharding(updates, input_sharding.spec, "updates")

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(ns4d_grouped_apply_step_factory(config), params, updates)

    assert_ns4d_sharding(result, result_sharding.spec, "grouped apply result")


def test_grouped_expert_apply_boundary_keeps_block_expert_leaves_grouped_without_collectives():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=2,
        expert_axis=4,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    input_sharding = ns4d_input_sharding(mesh, config, EXPERT_GROUPED_APPLY_BOUNDARY_BENCH)
    compute_sharding = ns4d_compute_sharding(mesh, config, EXPERT_GROUPED_APPLY_BOUNDARY_BENCH)
    result_sharding = ns4d_result_sharding(mesh, config, EXPERT_GROUPED_APPLY_BOUNDARY_BENCH)
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_APPLY_BOUNDARY_BENCH)
    updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_APPLY_BOUNDARY_BENCH)

    assert input_sharding.spec == P("data", "expert", None, None)
    assert compute_sharding.spec == P("data", "expert", None, None)
    assert result_sharding is not None
    assert result_sharding.spec == P("data", "expert", None, None)
    assert len(params["blocks"]) == 1
    assert set(params["blocks"][0]["mlp"]["expert_mlp"]) == {"w_gate_up", "w_down"}
    assert_ns4d_sharding(params, input_sharding.spec, "grouped expert params")
    assert_ns4d_sharding(updates, input_sharding.spec, "grouped expert updates")

    update_step = jax.jit(grouped_expert_apply_boundary_step_factory(config))
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, updates).lower(lowering_platforms=(platform,))

    assert_ns4d_sharding(result, result_sharding.spec, "grouped expert apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0


def test_grouped_expert_optimizer_apply_preserves_grouped_expert_sharding_without_collectives():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=2,
        expert_axis=4,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    input_sharding = ns4d_input_sharding(mesh, config, EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH)
    result_sharding = ns4d_result_sharding(mesh, config, EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH)
    params = synthetic_productionish_grouped_expert_specs(mesh, config, EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH)
    grads = synthetic_productionish_grouped_expert_specs(mesh, config, EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH)

    assert input_sharding.spec == P("data", "expert", None, None)
    assert result_sharding is not None
    assert result_sharding.spec == P("data", "expert", None, None)
    assert len(params["blocks"]) == 1
    assert len(params["ordinary_blocks"]) == 2
    assert params["ordinary_blocks"][0]["mlp"]["w_in"].ndim == 2
    assert params["ordinary_blocks"][0]["router"]["bias"].ndim == 1
    assert_ns4d_sharding(params, input_sharding.spec, "production-ish grouped expert params")
    assert_ns4d_sharding(grads, input_sharding.spec, "production-ish grouped expert grads")

    optimizer = build_grouped_expert_productionish_optimizer(config)
    update_step = jax.jit(grouped_expert_optimizer_apply_step_factory(config))
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        result, _next_state, updates = jax.eval_shape(update_step, params, grads, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state).lower(lowering_platforms=(platform,))

    assert_ns4d_sharding(updates, result_sharding.spec, "production-ish grouped expert updates")
    assert_ns4d_sharding(result, result_sharding.spec, "production-ish grouped expert apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0


@pytest.mark.parametrize(
    ("data_axis", "expert_axis", "ns4d_group_axis", "expected_spec"),
    [
        (2, 4, "data", P("data", "expert", None, None)),
        (1, 8, "none", P(None, "expert", None, None)),
    ],
)
def test_grouped_expert_muonh_optimizer_apply_preserves_grouped_expert_sharding_without_collectives(
    data_axis,
    expert_axis,
    ns4d_group_axis,
    expected_spec,
):
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis=ns4d_group_axis,
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=data_axis,
        expert_axis=expert_axis,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, data_axis, expert_axis, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    input_sharding = ns4d_input_sharding(mesh, config, EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)
    result_sharding = ns4d_result_sharding(mesh, config, EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)
    params = synthetic_productionish_grouped_expert_specs(
        mesh,
        config,
        EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    )
    grads = synthetic_productionish_grouped_expert_specs(
        mesh,
        config,
        EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    )

    assert input_sharding.spec == expected_spec
    assert result_sharding is not None
    assert result_sharding.spec == expected_spec
    assert_ns4d_sharding(params, expected_spec, "MuonH grouped expert params")
    assert_ns4d_sharding(grads, expected_spec, "MuonH grouped expert grads")

    optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=True)
    update_step = jax.jit(grouped_expert_optimizer_apply_step_factory(config, use_hyperball=True))
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        result, _next_state, updates = jax.eval_shape(update_step, params, grads, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state).lower(lowering_platforms=(platform,))

    assert_ns4d_sharding(updates, expected_spec, "MuonH grouped expert updates")
    assert_ns4d_sharding(result, expected_spec, "MuonH grouped expert apply result")
    hlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    hlo_summary = summarize_hlo(hlo_text)
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0


def test_grouped_expert_muonh_optimizer_apply_can_shard_group_over_replica_and_data_without_collectives():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=4,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    expected_spec = P(("replica_dcn", "data"), "expert", None, None)
    input_sharding = ns4d_input_sharding(mesh, config, EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)
    result_sharding = ns4d_result_sharding(mesh, config, EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)
    params = synthetic_productionish_grouped_expert_specs(
        mesh,
        config,
        EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    )
    grads = synthetic_productionish_grouped_expert_specs(
        mesh,
        config,
        EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    )

    assert input_sharding.spec == expected_spec
    assert result_sharding is not None
    assert result_sharding.spec == expected_spec
    assert_ns4d_sharding(params, expected_spec, "replica/data grouped expert params")
    assert_ns4d_sharding(grads, expected_spec, "replica/data grouped expert grads")

    optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=True)
    update_step = jax.jit(grouped_expert_optimizer_apply_step_factory(config, use_hyperball=True))
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        result, _next_state, updates = jax.eval_shape(update_step, params, grads, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state).lower(lowering_platforms=(platform,))

    assert_ns4d_sharding(updates, expected_spec, "replica/data grouped expert updates")
    assert_ns4d_sharding(result, expected_spec, "replica/data grouped expert apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0


@pytest.mark.parametrize(
    (
        "replica_axis",
        "data_axis",
        "expert_axis",
        "ns4d_group_axis",
        "layers",
        "expected_stack_size",
        "expected_spec",
    ),
    [
        (1, 1, 8, "none", 2, 2, P(None, "expert", None, None)),
        (2, 2, 8, "replica_dcn,data", 4, 4, P(("replica_dcn", "data"), "expert", None, None)),
        (2, 2, 8, "replica_dcn,data", 26, 28, P(("replica_dcn", "data"), "expert", None, None)),
    ],
)
def test_expert_only_grouped_muonh_harness_preserves_expert_stack_without_collectives(
    replica_axis,
    data_axis,
    expert_axis,
    ns4d_group_axis,
    layers,
    expected_stack_size,
    expected_spec,
):
    config = BenchConfig(
        layers=layers,
        ns4d_group_size=layers,
        ns4d_group_axis=ns4d_group_axis,
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=replica_axis,
        data_axis=data_axis,
        expert_axis=expert_axis,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(replica_axis, data_axis, expert_axis, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    input_sharding = ns4d_input_sharding(mesh, config, EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)
    result_sharding = ns4d_result_sharding(mesh, config, EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)
    grads = synthetic_grouped_expert_specs(mesh, config, EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH)

    assert input_sharding.spec == expected_spec
    assert result_sharding is not None
    assert result_sharding.spec == expected_spec
    assert params["blocks"][0]["mlp"]["expert_mlp"]["w_gate_up"].shape[0] == expected_stack_size
    assert_ns4d_sharding(params, expected_spec, "expert-only grouped params")
    assert_ns4d_sharding(grads, expected_spec, "expert-only grouped grads")

    optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=True)
    update_step = jax.jit(grouped_expert_optimizer_apply_step_factory(config, use_hyperball=True))
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        result, _next_state, updates = jax.eval_shape(update_step, params, grads, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state).lower(lowering_platforms=(platform,))

    assert_ns4d_sharding(updates, expected_spec, "expert-only grouped MuonH updates")
    assert_ns4d_sharding(result, expected_spec, "expert-only grouped MuonH result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0
    assert estimated_matrix_count(config, EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH) == expected_stack_size * 16


@pytest.mark.parametrize(
    ("replica_axis", "data_axis", "expert_axis", "ns4d_group_axis", "layers", "expected_compute_spec"),
    [
        (1, 1, 8, "none", 2, P(None, "expert", None, None)),
        (2, 2, 2, "replica_dcn,data", 4, P(("replica_dcn", "data"), "expert", None, None)),
    ],
)
def test_expert_fsdp_grouped_muonh_restores_ordinary_expert_updates_before_apply(
    replica_axis,
    data_axis,
    expert_axis,
    ns4d_group_axis,
    layers,
    expected_compute_spec,
):
    config = BenchConfig(
        layers=layers,
        ns4d_group_size=layers,
        ns4d_group_axis=ns4d_group_axis,
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=replica_axis,
        data_axis=data_axis,
        expert_axis=expert_axis,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(replica_axis, data_axis, expert_axis, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    optimizer = optax.trace(0.95, nesterov=True)
    update_step = jax.jit(expert_fsdp_grouped_muonh_optimizer_apply_step_factory(mesh, config))

    assert ns4d_compute_sharding(mesh, config, EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH).spec == (
        expected_compute_spec
    )
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        result, _next_state, updates = jax.eval_shape(update_step, params, grads, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "restored expert FSDP updates")
    assert_expert_fsdp_sharding(result, "expert FSDP apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_gather + hlo_summary.reduce_scatter <= 8
    assert estimated_matrix_count(config, EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH) == layers * 16


def test_expert_fsdp_grouped_apply_boundary_restores_ordinary_expert_updates_before_apply():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_apply_boundary_step_factory(mesh, config))

    assert ns4d_compute_sharding(mesh, config, EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH).spec == (
        P(("replica_dcn", "data"), "expert", None, None)
    )
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result, updates = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "restored expert FSDP updates")
    assert_expert_fsdp_sharding(result, "expert FSDP apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_restore_boundary_returns_updates_without_apply():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_restore_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        updates = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "restored expert FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_target_restore_boundary_returns_updates_without_apply():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_target_restore_boundary_step_factory(config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        updates = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "target restored expert FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_updates_muonh_restores_ordinary_expert_updates_before_apply():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_updates_muonh_apply_step_factory(mesh, config))

    assert ns4d_compute_sharding(mesh, config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH).spec == (
        P(("replica_dcn", "data"), "expert", None, None)
    )
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result, updates = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "restored expert FSDP updates")
    assert_expert_fsdp_sharding(result, "expert FSDP apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH) == config.layers * 16


def test_expert_fsdp_grouped_updates_muonh_can_return_updates_without_apply():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_updates_muonh_updates_step_factory(mesh, config))

    assert ns4d_compute_sharding(mesh, config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH).spec == (
        P(("replica_dcn", "data"), "expert", None, None)
    )
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        updates = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "restored expert FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH) == config.layers * 16


def test_full_production_muonh_optimizer_apply_covers_2d_and_grouped_expert_leaves():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=8,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    input_sharding = ns4d_input_sharding(mesh, config, FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH)
    result_sharding = ns4d_result_sharding(mesh, config, FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH)
    params = synthetic_full_production_muonh_specs(mesh, config, FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH)
    grads = synthetic_full_production_muonh_specs(mesh, config, FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH)
    mask = full_production_muonh_mask(params)

    assert input_sharding.spec == P(None, "expert", None, None)
    assert result_sharding is not None
    assert result_sharding.spec == P(None, "expert", None, None)
    assert_ns4d_sharding(params, input_sharding.spec, "full production grouped expert params")
    assert mask["blocks"][0]["mlp"]["expert_mlp"]["w_gate_up"] == "grouped_muonh"
    assert mask["ordinary_blocks"][0]["attn"]["w_q"] == "muonh"
    assert mask["ordinary_blocks"][0]["attn_gated_norm"]["w_down"] == "muonh"
    assert mask["ordinary_blocks"][0]["shared"]["w_down"] == "muonh"
    assert mask["ordinary_blocks"][0]["mlp"]["router"] == "ordinary"
    assert mask["ordinary_blocks"][0]["attn"]["attn_gate"] == "ordinary"
    assert mask["output_proj"] == "ordinary"

    optimizer = build_full_production_muonh_optimizer(config)
    update_step = jax.jit(full_production_muonh_optimizer_apply_step_factory(config))
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        result, _next_state, updates = jax.eval_shape(update_step, params, grads, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state).lower(lowering_platforms=(platform,))

    assert_ns4d_sharding(updates, result_sharding.spec, "full production grouped expert updates")
    assert_ns4d_sharding(result, result_sharding.spec, "full production grouped expert apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general >= 6
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_full_production_muonh_ns_dot_flops(config) > 2 * 8 * 8 * 16 * 8


def test_full_production_grouped_2d_muonh_optimizer_apply_groups_2d_leaves_without_collectives():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=16,
        replica_axis=1,
        data_axis=1,
        expert_axis=8,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_full_production_muonh_specs(mesh, config, FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH)
    grads = synthetic_full_production_muonh_specs(mesh, config, FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH)
    baseline_optimizer = build_full_production_muonh_optimizer(config)
    grouped_optimizer = build_full_production_muonh_optimizer(config, group_2d_muonh=True)
    baseline_step = jax.jit(full_production_muonh_optimizer_apply_step_factory(config))
    grouped_step = jax.jit(full_production_muonh_optimizer_apply_step_factory(config, group_2d_muonh=True))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        baseline_state = jax.eval_shape(baseline_optimizer.init, params)
        grouped_state = jax.eval_shape(grouped_optimizer.init, params)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        baseline_lowered = baseline_step.trace(params, grads, baseline_state).lower(lowering_platforms=(platform,))
        grouped_lowered = grouped_step.trace(params, grads, grouped_state).lower(lowering_platforms=(platform,))

    baseline_hlo = summarize_hlo(str(baseline_lowered.compiler_ir(dialect="stablehlo")))
    grouped_hlo = summarize_hlo(str(grouped_lowered.compiler_ir(dialect="stablehlo")))
    assert grouped_hlo.dot_general < baseline_hlo.dot_general
    assert grouped_hlo.batched_stack_dot_general >= 3
    assert grouped_hlo.two_batch_axis_dot_general >= 6
    assert grouped_hlo.all_gather == 0
    assert grouped_hlo.all_reduce == 0
    assert grouped_hlo.reduce_scatter == 0

    grouped_bench_specs = synthetic_full_production_muonh_specs(
        mesh,
        config,
        FULL_PRODUCTION_GROUPED_2D_MUONH_OPTIMIZER_APPLY_BENCH,
    )
    assert grouped_bench_specs["ordinary_blocks"][0]["attn"]["w_q"].shape == (16, 16)
    grouped_2d_estimates = estimate_grouped_2d_muonh(mesh, config, full_production_tree=True)
    assert sum(estimate.leaves for estimate in grouped_2d_estimates) == 22
    assert sorted(estimate.chunks for estimate in grouped_2d_estimates) == [[4], [4], [14]]


@pytest.mark.parametrize(
    ("bench_kind", "spec_factory", "step_factory", "expected_matrix_count", "expected_two_batch_dots"),
    [
        (
            ORDINARY_2D_GROUPED_PERSISTENT_APPLY_BENCH,
            lambda mesh, config: synthetic_ordinary_2d_grouped_persistent_specs(mesh, config),
            ordinary_2d_grouped_persistent_apply_timing_step_factory,
            22,
            0,
        ),
        (
            FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
            synthetic_full_production_grouped_persistent_specs,
            full_production_grouped_2d_persistent_apply_timing_step_factory,
            54,
            6,
        ),
    ],
)
def test_grouped_2d_persistent_apply_removes_apply_boundary_collectives(
    bench_kind,
    spec_factory,
    step_factory,
    expected_matrix_count,
    expected_two_batch_dots,
):
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="replica_dcn",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=16,
        replica_axis=2,
        data_axis=1,
        expert_axis=4,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 1, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    specs = (
        spec_factory(mesh, config, bench_kind)
        if bench_kind.startswith("full_production")
        else spec_factory(mesh, config)
    )
    update_step = jax.jit(step_factory(config))
    optimizer = optax.trace(0.95, nesterov=True)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, specs)
        result, _next_state = jax.eval_shape(update_step, specs, specs, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(specs, specs, state).lower(lowering_platforms=(platform,))

    assert result["ordinary_2d_groups"][0].ndim == 3
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.batched_stack_dot_general == 9
    assert hlo_summary.two_batch_axis_dot_general == expected_two_batch_dots
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, bench_kind) == expected_matrix_count


def test_full_production_grouped_persistent_specs_use_explicit_sharding_for_remainder_chunks():
    config = BenchConfig(
        layers=26,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=104,
        replica_axis=4,
        data_axis=1,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(4, 1, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )

    specs = synthetic_full_production_grouped_persistent_specs(
        mesh,
        config,
        FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
    )

    group_specs = specs["ordinary_2d_groups"]
    assert group_specs
    assert all(isinstance(getattr(spec, "sharding", None), NamedSharding) for spec in group_specs)
    assert any(spec.shape[0] == 80 and spec.sharding.spec == P("replica_dcn", None, None) for spec in group_specs)
    assert all(spec.shape[0] % config.replica_axis == 0 for spec in group_specs)


def test_persistent_grouped_2d_metadata_tracks_source_paths_and_padding():
    config = BenchConfig(
        layers=26,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=104,
        replica_axis=4,
        data_axis=1,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(4, 1, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    specs = synthetic_full_production_muonh_specs(
        mesh,
        config,
        FULL_PRODUCTION_GROUPED_2D_PERSISTENT_APPLY_BENCH,
    )

    metadata = persistent_grouped_2d_metadata_from_specs(
        specs,
        full_production_tree=True,
        max_grouped_stack_size=config.max_grouped_stack_size,
    )

    assert metadata
    assert all(group.source_paths for group in metadata)
    assert all(group.padded_stack_length >= group.valid_stack_length for group in metadata)
    assert all(group.padded_stack_length % config.replica_axis == 0 for group in metadata)
    assert any(
        group.valid_stack_length == 78
        and group.padded_stack_length == 80
        and "replica_dcn" in group.target_sharding_spec
        for group in metadata
    )
    assert any("ordinary_blocks.0.attn.w_q" in group.source_paths for group in metadata)


def test_ordinary_2d_grouped_decomposition_lowers_stack_ns_and_restore_split_separately():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=16,
        replica_axis=1,
        data_axis=1,
        expert_axis=8,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_ordinary_2d_muonh_specs(mesh, config)
    updates = synthetic_ordinary_2d_muonh_specs(mesh, config)
    stack_ns_step = jax.jit(grouped_2d_stack_ns_step_factory(config))
    restore_split_step = jax.jit(grouped_2d_restore_split_step_factory(config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        stack_ns_result = jax.eval_shape(stack_ns_step, params, updates)
        restore_split_result = jax.eval_shape(restore_split_step, params, updates)
        stack_ns_lowered = stack_ns_step.trace(params, updates).lower(lowering_platforms=(platform,))
        restore_split_lowered = restore_split_step.trace(params, updates).lower(lowering_platforms=(platform,))

    assert len(stack_ns_result) == 3
    assert restore_split_result["ordinary_blocks"][0]["attn"]["w_q"].shape == (16, 16)
    stack_ns_hlo = summarize_hlo(str(stack_ns_lowered.compiler_ir(dialect="stablehlo")))
    restore_split_hlo = summarize_hlo(str(restore_split_lowered.compiler_ir(dialect="stablehlo")))
    assert stack_ns_hlo.batched_stack_dot_general == 9
    assert restore_split_hlo.dot_general == 0
    assert restore_split_hlo.all_gather == 0
    assert restore_split_hlo.all_reduce == 0
    assert restore_split_hlo.reduce_scatter == 0


def test_ordinary_2d_grouped_optimizer_apply_reduces_dot_count_and_reports_grouping():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=16,
        replica_axis=1,
        data_axis=1,
        expert_axis=8,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_ordinary_2d_muonh_specs(mesh, config)
    grads = synthetic_ordinary_2d_muonh_specs(mesh, config)
    baseline_optimizer = build_ordinary_2d_muonh_optimizer(config)
    grouped_optimizer = build_ordinary_2d_muonh_optimizer(config, group_2d_muonh=True)
    baseline_step = jax.jit(ordinary_2d_muonh_optimizer_apply_step_factory(config))
    grouped_step = jax.jit(ordinary_2d_muonh_optimizer_apply_step_factory(config, group_2d_muonh=True))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        baseline_state = jax.eval_shape(baseline_optimizer.init, params)
        grouped_state = jax.eval_shape(grouped_optimizer.init, params)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        baseline_lowered = baseline_step.trace(params, grads, baseline_state).lower(lowering_platforms=(platform,))
        grouped_lowered = grouped_step.trace(params, grads, grouped_state).lower(lowering_platforms=(platform,))

    baseline_hlo = summarize_hlo(str(baseline_lowered.compiler_ir(dialect="stablehlo")))
    grouped_hlo = summarize_hlo(str(grouped_lowered.compiler_ir(dialect="stablehlo")))
    grouped_2d_estimates = estimate_grouped_2d_muonh(mesh, config, full_production_tree=False)

    assert grouped_hlo.dot_general < baseline_hlo.dot_general
    assert grouped_hlo.batched_stack_dot_general == 9
    assert sum(estimate.leaves for estimate in grouped_2d_estimates) == 22
    assert sorted(estimate.chunks for estimate in grouped_2d_estimates) == [[4], [4], [14]]
    assert estimated_ns_dot_flops(config, ORDINARY_2D_GROUPED_RESTORE_SPLIT_BENCH) == 0
    assert estimated_ns_dot_flops(config, FULL_PRODUCTION_APPLY_ONLY_BENCH) == 0
    assert estimated_matrix_count(config, ORDINARY_2D_MUONH_OPTIMIZER_APPLY_BENCH) == 22
    assert estimated_matrix_count(config, ORDINARY_2D_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH) == 22
    assert estimated_matrix_count(config, ORDINARY_2D_GROUPED_STACK_NS_BENCH) == 22


def test_grouped_2d_stack_target_shards_stack_axis_over_replica_and_data():
    mesh = AbstractMesh(
        axis_sizes=(4, 2, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    sample = jax.ShapeDtypeStruct((16, 16), jnp.float32, sharding=NamedSharding(mesh, P(None, None)))

    target = _stacked_2d_target((8, 16, 16), sample)

    assert isinstance(target, NamedSharding)
    assert target.spec == P(("replica_dcn", "data"), None, None)


def test_summary_row_reports_matrix_count_and_stack_estimates():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=3,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=8,
        model_axis=1,
        learning_rate=0.02,
    )
    group_estimates = [asdict(estimate) for estimate in estimate_grouping(config)]
    result = {
        "metadata": {
            "label": "full_production_muonh_optimizer_apply_h3",
            "bench_kind": FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
            "config": asdict(config),
            "devices": 8,
            "ns4d_group_size": 2,
            "ns4d_padded_group_size": 2,
            "ns4d_input_sharding_spec": "PartitionSpec(None, 'expert', None, None)",
            "ns4d_compute_sharding_spec": "PartitionSpec(None, 'expert', None, None)",
            "ns4d_result_sharding_spec": "PartitionSpec(None, 'expert', None, None)",
            "ns4d_boundary_status": "full_production_muonh_optimizer_updates_apply",
            "boundary_collectives_allowed": False,
            "grouped_expert_group_count": 1,
            "group_estimates": group_estimates,
        }
    }

    row = summary_row(result)

    assert row["estimated_matrix_count"] == estimated_matrix_count(
        config,
        FULL_PRODUCTION_MUONH_OPTIMIZER_APPLY_BENCH,
    )
    assert row["estimated_matrix_count"] == 54
    assert row["grouped_expert_group_count"] == 1
    assert row["group_estimates"] == group_estimates


def test_grouped_4d_hyperball_projects_each_group_expert_matrix_independently():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="data",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=3,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    param = jnp.arange(2 * 3 * 2 * 2, dtype=jnp.float32).reshape(2, 3, 2, 2) + 1.0
    direction = jnp.flip(param, axis=-1) * 0.1 + 0.25

    grouped = grouped_4d_hyperball_update(param, direction, config)

    expected_rows = []
    for group_index in range(param.shape[0]):
        expert_rows = []
        for expert_index in range(param.shape[1]):
            matrix_param = param[group_index, expert_index]
            matrix_direction = direction[group_index, expert_index]
            param_norm = jnp.linalg.norm(matrix_param)
            update_norm = jnp.linalg.norm(matrix_direction)
            step_scale = config.learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
            dot = jnp.sum(matrix_param * matrix_direction)
            new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
            new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
            rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
            expert_rows.append((rescale - 1) * matrix_param - rescale * step_scale * matrix_direction)
        expected_rows.append(jnp.stack(expert_rows))
    expected = jnp.stack(expected_rows)

    assert jnp.allclose(grouped, expected, rtol=1e-6, atol=1e-6)


def test_grouped_3d_hyperball_projects_each_stacked_matrix_independently():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="data",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=3,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    param = jnp.arange(3 * 2 * 2, dtype=jnp.float32).reshape(3, 2, 2) + 1.0
    direction = jnp.flip(param, axis=-1) * 0.1 + 0.25

    grouped = grouped_3d_hyperball_update(param, direction, config)

    expected_rows = []
    for matrix_index in range(param.shape[0]):
        matrix_param = param[matrix_index]
        matrix_direction = direction[matrix_index]
        param_norm = jnp.linalg.norm(matrix_param)
        update_norm = jnp.linalg.norm(matrix_direction)
        step_scale = config.learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
        dot = jnp.sum(matrix_param * matrix_direction)
        new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
        new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
        rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
        expected_rows.append((rescale - 1) * matrix_param - rescale * step_scale * matrix_direction)
    expected = jnp.stack(expected_rows)

    assert jnp.allclose(grouped, expected, rtol=1e-6, atol=1e-6)
