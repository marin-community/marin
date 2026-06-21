# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import optax
import pytest
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe import launch_cw_muon_update_bench as launcher
from experiments.grug.moe.launch_cw_muon_update_bench import (
    _sync_global_devices_if_multihost,
    _wandb_metric_row,
    build_step,
)
from experiments.grug.moe.muon_update_bench import (
    EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_BENCH,
    EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH,
    EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_CHECKSUM_BENCH,
    EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_UPDATE_BENCH,
    EXPERT_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
    EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
    EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_CHUNKS_BENCH,
    EXPERT_FSDP_GRADS_TO_GROUPED_CHUNKS_BENCH,
    EXPERT_FSDP_GRADS_TO_PACKED_GROUPED_CHUNKS_BENCH,
    EXPERT_FSDP_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_CUSTOM_PARTITION_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_A2A_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_A2A_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_PPERMUTE_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_DATA_PPERMUTE_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_DATA_PPERMUTE_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_SLICE_FIRST_GATHER_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_EXPLICIT_TUPLE_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_PACKED_A2A_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_PACKED_DATA_FIRST_PPERMUTE_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_PACKED_DATA_PPERMUTE_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_PACKED_SLICE_FIRST_GATHER_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_APPLY_CHUNKED_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_APPLY_CHUNKED_FSDP_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TARGET_RESTORE_BOUNDARY_BENCH,
    EXPERT_FSDP_GROUPED_TRACE_MUONH_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_DIRECT_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_EXPLICIT_A2A_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_EXPLICIT_APPLY_BENCH,
    EXPERT_FSDP_GROUPED_UPDATES_MUONH_UPDATES_BENCH,
    EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_PACKED_BANK_DIRECTION_APPLY_BENCH,
    EXPERT_FSDP_PACKED_BANK_MUONH_APPLY_BENCH,
    EXPERT_FSDP_PACKED_BANK_MUONH_DIRECT_APPLY_BENCH,
    EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH,
    EXPERT_FSDP_PACKED_BANK_SLICE_FIRST_APPLY_BOUNDARY_BENCH,
    EXPERT_FSDP_PACKED_BANK_SLICE_FIRST_DIRECT_APPLY_BOUNDARY_BENCH,
    EXPERT_GROUPED_APPLY_BOUNDARY_BENCH,
    EXPERT_GROUPED_BANK_CONSUMER_BENCH,
    EXPERT_GROUPED_LAYER_SLICE_BENCH,
    EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
    EXPERT_GROUPED_MUONH_BANK_CONSUMER_BENCH,
    EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH,
    EXPERT_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_OPTIMIZER_APPLY_BENCH,
    EXPERT_GROUPED_SCAN_BANK_CONSUMER_BENCH,
    EXPERT_GROUPED_SEQUENTIAL_BANK_CONSUMER_BENCH,
    EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_BLOCK_GROUP_VALUE_GRAD_MUONH_UPDATE_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_UPDATE_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_LAYERWISE_VALUE_GRAD_MUONH_UPDATE_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_VALUE_GRAD_MUONH_UPDATE_BENCH,
    EXPERT_LAYER_CHUNKED_PACKED_MASTER_GROUPED_GRAD_MUONH_UPDATE_BENCH,
    EXPERT_ONLY_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    EXPERT_PACKED_MASTER_FSDP_BULK_CONSUMER_BENCH,
    EXPERT_PACKED_MASTER_FSDP_BULK_GRAD_BENCH,
    EXPERT_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH,
    EXPERT_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_BENCH,
    EXPERT_PACKED_MASTER_FSDP_SLAB_CONSUMER_BENCH,
    EXPERT_PACKED_MASTER_FSDP_SLAB_GRAD_BENCH,
    EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    EXPERT_PACKED_MASTER_MUONH_FSDP_BULK_CONSUMER_BENCH,
    EXPERT_PACKED_MASTER_MUONH_FSDP_SEQUENTIAL_CONSUMER_BENCH,
    EXPERT_PACKED_MASTER_MUONH_FSDP_SLAB_CONSUMER_BENCH,
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
    REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH,
    REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH,
    BenchConfig,
    HloSummary,
    MuonExpertState,
    MuonMasterBank,
    _stacked_2d_target,
    assert_chunked_packed_grouped_expert_bank_sharding,
    assert_expert_ep_sharding,
    assert_expert_fsdp_sharding,
    assert_grouped_expert_sharding,
    assert_grouped_expert_target_fsdp_sharding,
    assert_grouped_moe_consumer_sharding,
    assert_ns4d_sharding,
    assert_packed_grouped_expert_bank_sharding,
    bench_skip_reason,
    boundary_correctness_skipped_reason,
    build_full_production_muonh_optimizer,
    build_grouped_expert_productionish_optimizer,
    build_ordinary_2d_muonh_optimizer,
    build_real_expert_fsdp_grouped_muonh_optimizer,
    bytes_to_gib,
    chunked_packed_master_bank_to_grouped_expert_tree,
    chunked_packed_master_chunk_to_fsdp_expert_layer,
    chunked_packed_master_valid_group_sizes_for_bench,
    create_abstract_mesh,
    create_mesh,
    estimate_grouped_2d_muonh,
    estimate_grouping,
    estimated_boundary_byte_estimates,
    estimated_boundary_phase_estimates,
    estimated_full_production_muonh_ns_dot_flops,
    estimated_matrix_count,
    estimated_ns_dot_flops,
    expert_chunked_packed_master_fsdp_block_group_chunk_consumer_loss,
    expert_chunked_packed_master_fsdp_block_group_consumer_loss,
    expert_chunked_packed_master_fsdp_chunk_consumer_loss,
    expert_chunked_packed_master_fsdp_layer_grad_muonh_update_step_factory,
    expert_chunked_packed_master_fsdp_layer_grad_step_factory,
    expert_chunked_packed_master_fsdp_sequential_consumer_loss,
    expert_chunked_packed_master_fsdp_sequential_grad_muonh_checksum_step_factory,
    expert_chunked_packed_master_fsdp_sequential_grad_muonh_update_step_factory,
    expert_chunked_packed_master_fsdp_streaming_grad_muonh_checksum_step_factory,
    expert_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss,
    expert_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss_step_factory,
    expert_chunked_packed_master_fsdp_streaming_grad_muonh_update_step_factory,
    expert_chunked_packed_master_fsdp_streaming_value_grad_muonh_update_step_factory,
    expert_chunked_packed_master_grad_muonh_update_step_factory_for_bench,
    expert_chunked_packed_master_grouped_consumer_grad_step_factory,
    expert_chunked_packed_master_grouped_consumer_loss,
    expert_chunked_packed_master_grouped_grad_muonh_update_step_factory,
    expert_fsdp_grouped_apply_boundary_step_factory,
    expert_fsdp_grouped_custom_partition_slice_first_gather_restore_boundary_step_factory,
    expert_fsdp_grouped_explicit_a2a_apply_boundary_step_factory,
    expert_fsdp_grouped_explicit_apply_boundary_step_factory,
    expert_fsdp_grouped_explicit_data_first_a2a_apply_boundary_step_factory,
    expert_fsdp_grouped_explicit_data_first_a2a_restore_boundary_step_factory,
    expert_fsdp_grouped_explicit_data_first_ppermute_restore_boundary_step_factory,
    expert_fsdp_grouped_explicit_data_ppermute_apply_boundary_step_factory,
    expert_fsdp_grouped_explicit_data_ppermute_restore_boundary_step_factory,
    expert_fsdp_grouped_explicit_restore_boundary_step_factory,
    expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary_step_factory,
    expert_fsdp_grouped_explicit_slice_first_gather_restore_boundary_step_factory,
    expert_fsdp_grouped_explicit_tuple_slice_first_gather_restore_boundary_step_factory,
    expert_fsdp_grouped_muonh_optimizer_apply_step_factory,
    expert_fsdp_grouped_packed_a2a_apply_boundary_step_factory,
    expert_fsdp_grouped_packed_data_first_ppermute_apply_boundary_step_factory,
    expert_fsdp_grouped_packed_data_ppermute_apply_boundary_step_factory,
    expert_fsdp_grouped_packed_slice_first_gather_apply_boundary_step_factory,
    expert_fsdp_grouped_persistent_muonh_apply_step_factory,
    expert_fsdp_grouped_restore_boundary_step_factory,
    expert_fsdp_grouped_target_apply_boundary_step_factory,
    expert_fsdp_grouped_target_apply_chunked_boundary_step_factory,
    expert_fsdp_grouped_target_apply_chunked_fsdp_boundary_step_factory,
    expert_fsdp_grouped_target_restore_boundary_step_factory,
    expert_fsdp_grouped_trace_muonh_apply_step_factory,
    expert_fsdp_grouped_updates_muonh_apply_step_factory,
    expert_fsdp_grouped_updates_muonh_direct_apply_step_factory,
    expert_fsdp_grouped_updates_muonh_explicit_a2a_apply_step_factory,
    expert_fsdp_grouped_updates_muonh_explicit_apply_step_factory,
    expert_fsdp_grouped_updates_muonh_updates_step_factory,
    expert_fsdp_layer_consumer_input_specs,
    expert_fsdp_layer_consumer_loss,
    expert_fsdp_packed_bank_a2a_apply_boundary_step_factory,
    expert_fsdp_packed_bank_direct_apply_boundary_step_factory,
    expert_fsdp_packed_bank_direction_apply_step_factory,
    expert_fsdp_packed_bank_muonh_apply_step_factory,
    expert_fsdp_packed_bank_muonh_direct_apply_step_factory,
    expert_fsdp_packed_bank_muonh_update_only_step_factory,
    expert_fsdp_packed_bank_muonh_update_only_timing_step_factory,
    expert_fsdp_packed_bank_muonh_updates_step_factory,
    expert_fsdp_packed_bank_slice_first_apply_boundary_step_factory,
    expert_fsdp_packed_bank_slice_first_direct_apply_boundary_step_factory,
    expert_grouped_layer_slice_step_factory,
    expert_grouped_single_layer_slice_step_factory,
    expert_layer_chunked_packed_master_fsdp_sequential_consumer_loss,
    expert_layer_chunked_packed_master_fsdp_sequential_consumer_step_factory,
    expert_muon_master_bank_block_group_train_step_factory,
    expert_packed_master_consumer_grad_step_factory,
    expert_packed_master_consumer_loss,
    expert_packed_master_fsdp_bulk_consumer_loss,
    expert_packed_master_fsdp_bulk_consumer_step_factory,
    expert_packed_master_fsdp_grad_outputs,
    expert_packed_master_fsdp_grad_step_factory,
    expert_packed_master_fsdp_layer_consumer_step_factory,
    expert_packed_master_fsdp_sequential_consumer_loss,
    expert_packed_master_fsdp_sequential_consumer_step_factory,
    expert_packed_master_fsdp_slab_consumer_loss,
    expert_packed_master_fsdp_slab_consumer_step_factory,
    expert_packed_master_muonh_consumer_step_factory,
    expert_packed_master_muonh_fsdp_bulk_consumer_outputs,
    expert_packed_master_muonh_fsdp_bulk_consumer_step_factory,
    expert_packed_master_muonh_fsdp_sequential_consumer_outputs,
    expert_packed_master_muonh_fsdp_sequential_consumer_step_factory,
    expert_packed_master_muonh_fsdp_slab_consumer_outputs,
    expert_packed_master_muonh_fsdp_slab_consumer_step_factory,
    expert_packed_master_muonh_update_outputs,
    fsdp_grads_to_explicit_packed_grouped_bank_step_factory,
    fsdp_grads_to_explicit_packed_grouped_chunks_step_factory,
    fsdp_grads_to_grouped_chunks_step_factory,
    fsdp_grads_to_grouped_chunks_timing_step_factory_for_bench,
    fsdp_grads_to_packed_grouped_chunks_step_factory,
    fsdp_grouped_boundary_correctness_max_error,
    full_production_grouped_2d_persistent_apply_timing_step_factory,
    full_production_muonh_mask,
    full_production_muonh_optimizer_apply_step_factory,
    grouped_2d_restore_split_step_factory,
    grouped_2d_stack_ns_step_factory,
    grouped_3d_hyperball_update,
    grouped_4d_hyperball_update,
    grouped_apply_boundary_collectives,
    grouped_expert_apply_boundary_step_factory,
    grouped_expert_bank_consumer_flops,
    grouped_expert_bank_consumer_outputs,
    grouped_expert_bank_consumer_step_factory,
    grouped_expert_group_sizes_for_bench,
    grouped_expert_muonh_bank_consumer_step_factory,
    grouped_expert_muonh_moe_mlp_consumer_step_factory,
    grouped_expert_optimizer_apply_step_factory,
    grouped_expert_scan_bank_consumer_step_factory,
    grouped_expert_sequential_bank_consumer_step_factory,
    grouped_moe_consumer_chunk_tokens,
    grouped_moe_mlp_consumer_step_factory,
    is_expert_fsdp_grouped_bench,
    make_array_tree,
    make_chunked_packed_grouped_expert_master_bank_tree,
    make_grouped_expert_array_tree,
    make_grouped_expert_consumer_input_tree,
    make_packed_grouped_expert_bank_tree,
    make_packed_grouped_expert_master_bank_tree,
    materialize_expert_block_group_from_muon_master_bank,
    materialize_expert_layer_from_muon_master_bank,
    muon_master_bank_metadata,
    ns4d_boundary_status,
    ns4d_compute_sharding,
    ns4d_grouped_apply_step_factory,
    ns4d_input_sharding,
    ns4d_result_sharding,
    ns_logical_matrix_shapes,
    ordinary_2d_grouped_persistent_apply_timing_step_factory,
    ordinary_2d_muonh_optimizer_apply_step_factory,
    output_path_for_config,
    packed_master_bank_to_fsdp_expert_layer,
    packed_master_bank_to_grouped_expert_tree,
    persistent_grouped_2d_metadata_from_specs,
    real_expert_fsdp_grouped_muonh_optimizer_apply_step_factory,
    real_expert_fsdp_grouped_muonh_optimizer_update_step_factory,
    rebuild_expert_tree_from_muon_master_bank,
    should_check_grouped_apply_boundary_collectives,
    summarize_hlo,
    summary_row,
    synthetic_chunked_packed_grouped_expert_master_bank_specs,
    synthetic_fsdp_expert_shardings,
    synthetic_fsdp_expert_specs,
    synthetic_full_production_grouped_persistent_specs,
    synthetic_full_production_muonh_specs,
    synthetic_grouped_expert_consumer_input_specs,
    synthetic_grouped_expert_sequential_consumer_input_specs,
    synthetic_grouped_expert_specs,
    synthetic_grouped_moe_mlp_consumer_input_specs,
    synthetic_ns4d_specs,
    synthetic_ordinary_2d_grouped_persistent_specs,
    synthetic_ordinary_2d_muonh_specs,
    synthetic_packed_grouped_expert_bank_specs,
    synthetic_packed_grouped_expert_master_bank_specs,
    synthetic_productionish_grouped_expert_specs,
    synthetic_shapes,
    time_ns4d,
    zeropower_via_newtonschulz_4d_for_config,
)


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def test_cw_muon_update_bench_launcher_reads_nesterov_env(monkeypatch):
    monkeypatch.setenv("RUN_ID", "muon-update-bench-test")
    monkeypatch.setenv("MUON_BENCH_NESTEROV", "false")

    step = build_step()

    assert step.config.nesterov is False


def test_cw_muon_update_bench_launcher_can_write_compiled_hlo(monkeypatch):
    monkeypatch.setenv("RUN_ID", "muon-update-bench-test")
    monkeypatch.setenv("MUON_BENCH_WRITE_COMPILED_HLO", "true")

    step = build_step()

    assert step.config.write_compiled_hlo is True


def test_cw_muon_update_bench_launcher_reads_strict_boundary_gate_env(monkeypatch):
    monkeypatch.setenv("RUN_ID", "muon-update-bench-test")
    monkeypatch.setenv("MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES", "true")

    step = build_step()

    assert step.config.require_no_boundary_collectives is True


def test_cw_muon_update_bench_launcher_reads_grouped_muonh_boundary_env(monkeypatch):
    monkeypatch.setenv("RUN_ID", "muon-update-bench-test")
    monkeypatch.setenv("MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY", "true")
    monkeypatch.setenv("MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE", "true")
    monkeypatch.setenv("MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES", "true")

    step = build_step()

    assert step.config.expert_grouped_muonh_packed_entry is True
    assert step.config.expert_grouped_muonh_packed_bank_compute is True
    assert step.config.expert_grouped_muonh_chunk_local_boundaries is True


def test_cw_muon_update_bench_launcher_reads_wandb_env(monkeypatch):
    monkeypatch.setenv("RUN_ID", "muon-update-bench-test")
    monkeypatch.setenv("MUON_BENCH_TRACKER", "wandb")
    monkeypatch.setenv("MUON_BENCH_WANDB_PROJECT", "marin_moe_test")
    monkeypatch.setenv("MUON_BENCH_WANDB_GROUP", "muon-test-group")

    step = build_step()

    assert step.config.wandb is True
    assert step.config.wandb_project == "marin_moe_test"
    assert step.config.wandb_group == "muon-test-group"


def test_create_abstract_mesh_includes_current_device_kind_when_device_count_matches():
    config = BenchConfig(
        layers=1,
        ns4d_group_size=1,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=1,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=1,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )

    abstract_mesh = create_abstract_mesh(config)

    assert abstract_mesh.abstract_device is not None


def test_wandb_metric_row_keeps_scalar_topline_fields():
    row = {
        "label": "candidate",
        "median_seconds": 0.3,
        "median_h100_bf16_peak_pct": 50.0,
        "compiled_hlo_all_gather": 26,
        "chunks": [[26]],
    }

    metrics = _wandb_metric_row(row, 7)

    assert metrics["bench/row_index"] == 7
    assert metrics["bench/label"] == "candidate"
    assert metrics["bench/median_seconds"] == 0.3
    assert metrics["bench/median_h100_bf16_peak_pct"] == 50.0
    assert metrics["bench/compiled_hlo_all_gather"] == 26
    assert "bench/chunks" not in metrics


def test_sync_global_devices_if_multihost_skips_single_process(monkeypatch):
    sync_calls = []
    monkeypatch.setattr(launcher.jax, "process_count", lambda: 1)
    monkeypatch.setattr(launcher, "sync_global_devices", sync_calls.append)

    _sync_global_devices_if_multihost("before_wandb")

    assert sync_calls == []


def test_sync_global_devices_if_multihost_uses_stable_bench_prefix(monkeypatch):
    sync_calls = []
    monkeypatch.setattr(launcher.jax, "process_count", lambda: 2)
    monkeypatch.setattr(launcher, "sync_global_devices", sync_calls.append)

    _sync_global_devices_if_multihost("after_wandb")

    assert sync_calls == ["muon_update_bench_after_wandb"]


def test_output_path_for_config_preserves_remote_uri():
    output = output_path_for_config("s3://bucket/prefix/compiled_hlo.txt", "label", 2)

    assert output == "s3://bucket/prefix/compiled_hlo_label.txt"


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


def test_summarize_hlo_counts_compiled_collective_instructions_not_metadata_mentions():
    hlo_text = "\n".join(
        (
            'ROOT %a2a = bf16[1,2] all-to-all(%param_0), metadata={op_name="foo/all-to-all"}',
            "%all-to-all-start.29 = ((bf16[1,2]), bf16[1,2]) async-start(%bitcast), "
            'calls=%async_computation.29, metadata={op_name="foo/all-to-all"}',
            '%all-to-all-done.29 = bf16[1,2] async-done(%all-to-all-start.29), metadata={op_name="foo"}',
            "%ag-start.10 = (bf16[1], bf16[4]) all-gather-start(%slice), metadata={op_name='foo'}",
            "%ag-done.10 = bf16[4,32] all-gather-done(%ag-start.10), metadata={op_name='foo/all_gather'}",
        )
    )
    hlo_summary = summarize_hlo(hlo_text)

    assert hlo_summary.all_to_all == 1
    assert hlo_summary.all_gather == 1


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


def test_unfused_expert_gate_up_uses_three_uniform_logical_ns_shapes():
    fused_config = BenchConfig(
        layers=26,
        ns4d_group_size=26,
        ns4d_group_axis="none",
        hidden_dim=2560,
        intermediate_dim=1280,
        num_experts=256,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=3,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=512,
        replica_axis=1,
        data_axis=1,
        expert_axis=8,
        model_axis=1,
        learning_rate=0.02,
    )
    unfused_config = replace(fused_config, unfused_expert_gate_up=True)

    assert synthetic_shapes(unfused_config) == {
        "w_gate": (256, 2560, 1280),
        "w_up": (256, 2560, 1280),
        "w_down": (256, 1280, 2560),
    }
    assert ns_logical_matrix_shapes(unfused_config) == {
        "w_gate": (1280, 2560),
        "w_up": (1280, 2560),
        "w_down": (1280, 2560),
    }

    fused_flops = estimated_ns_dot_flops(fused_config, EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH)
    unfused_flops = estimated_ns_dot_flops(unfused_config, EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH)
    assert fused_flops == 2_428_804_005_888_000
    assert unfused_flops == 1_256_277_934_080_000
    assert unfused_flops / fused_flops == pytest.approx(15 / 29)
    assert estimated_matrix_count(unfused_config, EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH) == 26 * 3 * 256


def test_unfused_expert_gate_up_packed_bank_update_only_returns_three_banks():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=4,
        data_axis=1,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        unfused_expert_gate_up=True,
    )
    mesh = AbstractMesh(
        axis_sizes=(4, 1, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(expert_fsdp_packed_bank_muonh_update_only_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "unfused packed-bank MuonH params")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        packed_updates = jax.eval_shape(update_step, params, params)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, params).lower(lowering_platforms=(platform,))

    assert set(packed_updates["packed"]) == {"w_gate", "w_up", "w_down"}
    assert packed_updates["packed"]["w_gate"].shape == (4, 8, 16, 8)
    assert packed_updates["packed"]["w_up"].shape == (4, 8, 16, 8)
    assert packed_updates["packed"]["w_down"].shape == (4, 8, 8, 16)
    assert_packed_grouped_expert_bank_sharding(
        packed_updates,
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH,
        "unfused packed-bank MuonH updates",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_to_all == 0


def test_grouped_expert_layer_slice_boundary_returns_ep_consumable_leaves():
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
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_LAYER_SLICE_BENCH)
    update_step = jax.jit(expert_grouped_layer_slice_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params).lower(lowering_platforms=(platform,))

    assert params["blocks"][0]["mlp"]["expert_mlp"]["w_gate_up"].shape == (4, 8, 16, 16)
    assert_expert_ep_sharding(result, "grouped expert layer slices")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_grouped_expert_single_layer_slice_boundary_returns_one_ep_consumable_layer():
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
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH)
    update_step = jax.jit(expert_grouped_single_layer_slice_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params).lower(lowering_platforms=(platform,))

    assert set(result) == {"mlp"}
    assert result["mlp"]["expert_mlp"]["w_gate_up"].shape == (8, 16, 16)
    assert result["mlp"]["expert_mlp"]["w_down"].shape == (8, 8, 16)
    assert_expert_ep_sharding(result, "grouped expert single layer slice")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_grouped_expert_bank_consumer_preserves_grouped_bank_without_collectives():
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
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    expected_spec = P(("replica_dcn", "data"), "expert", None, None)
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_BANK_CONSUMER_BENCH)
    activations = synthetic_grouped_expert_consumer_input_specs(mesh, config, EXPERT_GROUPED_BANK_CONSUMER_BENCH)
    update_step = jax.jit(grouped_expert_bank_consumer_step_factory(config))

    assert_ns4d_sharding(params, expected_spec, "grouped expert bank consumer params")
    assert_ns4d_sharding(activations, expected_spec, "grouped expert bank consumer inputs")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, activations).lower(lowering_platforms=(platform,))

    assert_ns4d_sharding(result, expected_spec, "grouped expert bank consumer result")
    assert result["blocks"][0]["x"].shape == (4, 8, 3, 16)
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 2
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0
    assert grouped_expert_bank_consumer_flops(config) == 4 * 8 * 3 * (2 * 16 * 16 + 2 * 8 * 16)


def test_grouped_expert_sequential_bank_consumer_skips_sharded_group_axis():
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
        grouped_expert_consumer_tokens_per_expert=3,
    )

    reason = bench_skip_reason(config, EXPERT_GROUPED_SEQUENTIAL_BANK_CONSUMER_BENCH)

    assert reason is not None
    assert "cannot directly slice a grouped layer axis" in reason


def test_grouped_expert_sequential_bank_consumer_returns_expert_local_outputs():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="none",
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
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_SEQUENTIAL_BANK_CONSUMER_BENCH)
    activations = synthetic_grouped_expert_sequential_consumer_input_specs(mesh, config)
    update_step = jax.jit(grouped_expert_sequential_bank_consumer_step_factory(config))

    assert_grouped_expert_sharding(
        params,
        mesh,
        config,
        EXPERT_GROUPED_SEQUENTIAL_BANK_CONSUMER_BENCH,
        "sequential grouped expert params",
    )
    assert_expert_ep_sharding(activations, "sequential grouped expert inputs")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, activations).lower(lowering_platforms=(platform,))

    assert_expert_ep_sharding(result, "sequential grouped expert result")
    assert result["blocks"][0]["x"].shape == (8, 3, 16)
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 2 * config.layers
    assert grouped_expert_bank_consumer_flops(config) == 4 * 8 * 3 * (2 * 16 * 16 + 2 * 8 * 16)


def test_grouped_expert_scan_bank_consumer_skips_sharded_group_axis():
    config = BenchConfig(
        layers=6,
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
        grouped_expert_consumer_tokens_per_expert=3,
    )

    reason = bench_skip_reason(config, EXPERT_GROUPED_SCAN_BANK_CONSUMER_BENCH)

    assert reason is not None
    assert "lax.scan requires the scanned xs dimension to be replicated" in reason


def test_grouped_expert_scan_bank_consumer_returns_expert_local_outputs_without_sharded_group_axis():
    config = BenchConfig(
        layers=6,
        ns4d_group_size=4,
        ns4d_group_axis="none",
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
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_SCAN_BANK_CONSUMER_BENCH)
    activations = synthetic_grouped_expert_sequential_consumer_input_specs(mesh, config)
    update_step = jax.jit(grouped_expert_scan_bank_consumer_step_factory(config))

    assert bench_skip_reason(config, EXPERT_GROUPED_SCAN_BANK_CONSUMER_BENCH) is None
    assert_grouped_expert_sharding(
        params,
        mesh,
        config,
        EXPERT_GROUPED_SCAN_BANK_CONSUMER_BENCH,
        "scan grouped expert params",
    )
    assert_expert_ep_sharding(activations, "scan grouped expert inputs")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, activations).lower(lowering_platforms=(platform,))

    assert_expert_ep_sharding(result, "scan grouped expert result")
    assert result["blocks"][0]["x"].shape == (8, 3, 16)
    assert result["blocks"][1]["x"].shape == (8, 3, 16)
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0
    assert grouped_expert_bank_consumer_flops(config) == 6 * 8 * 3 * (2 * 16 * 16 + 2 * 8 * 16)


def test_grouped_moe_mlp_consumer_preserves_grouped_bank_and_routed_activation_sharding():
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
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH)
    routed_inputs = synthetic_grouped_moe_mlp_consumer_input_specs(
        mesh,
        config,
        EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
    )
    update_step = jax.jit(grouped_moe_mlp_consumer_step_factory(mesh, config))

    assert_grouped_expert_sharding(params, mesh, config, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH, "params")
    assert_grouped_moe_consumer_sharding(
        routed_inputs,
        mesh,
        config,
        EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
        "routed inputs",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, routed_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, routed_inputs).lower(lowering_platforms=(platform,))

    assert_grouped_moe_consumer_sharding(result, mesh, config, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH, "result")
    assert result["blocks"][0]["x"].shape == (4, 24, 16)
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 2
    assert hlo_summary.all_gather > 0
    assert hlo_summary.reduce_scatter > 0
    assert grouped_expert_bank_consumer_flops(config) == 4 * 8 * 3 * (2 * 16 * 16 + 2 * 8 * 16)


def test_grouped_muonh_bank_consumer_updates_then_consumes_grouped_bank():
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
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    expected_spec = P(("replica_dcn", "data"), "expert", None, None)
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_MUONH_BANK_CONSUMER_BENCH)
    grads = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_MUONH_BANK_CONSUMER_BENCH)
    activations = synthetic_grouped_expert_consumer_input_specs(
        mesh,
        config,
        EXPERT_GROUPED_MUONH_BANK_CONSUMER_BENCH,
    )
    optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=True)
    update_step = jax.jit(grouped_expert_muonh_bank_consumer_step_factory(config))

    assert_ns4d_sharding(params, expected_spec, "grouped MuonH bank params")
    assert_ns4d_sharding(grads, expected_spec, "grouped MuonH bank grads")
    assert_ns4d_sharding(activations, expected_spec, "grouped MuonH bank consumer inputs")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        next_params, _next_state, outputs = jax.eval_shape(update_step, params, grads, state, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state, activations).lower(lowering_platforms=(platform,))

    assert_grouped_expert_sharding(
        next_params,
        mesh,
        config,
        EXPERT_GROUPED_MUONH_BANK_CONSUMER_BENCH,
        "updated grouped MuonH bank params",
    )
    assert_ns4d_sharding(outputs, expected_spec, "grouped MuonH bank consumer outputs")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0
    assert estimated_ns_dot_flops(config, EXPERT_GROUPED_MUONH_BANK_CONSUMER_BENCH) > grouped_expert_bank_consumer_flops(
        config
    )


def test_grouped_muonh_moe_mlp_consumer_updates_then_uses_public_grouped_moe():
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
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH)
    grads = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH)
    routed_inputs = synthetic_grouped_moe_mlp_consumer_input_specs(
        mesh,
        config,
        EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH,
    )
    optimizer = build_grouped_expert_productionish_optimizer(config, use_hyperball=True)
    update_step = jax.jit(grouped_expert_muonh_moe_mlp_consumer_step_factory(mesh, config))

    assert_grouped_expert_sharding(params, mesh, config, EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH, "params")
    assert_grouped_expert_sharding(grads, mesh, config, EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH, "grads")
    assert_grouped_moe_consumer_sharding(
        routed_inputs,
        mesh,
        config,
        EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH,
        "routed inputs",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        next_params, _next_state, outputs = jax.eval_shape(update_step, params, grads, state, routed_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads, state, routed_inputs).lower(lowering_platforms=(platform,))

    assert_grouped_expert_sharding(
        next_params,
        mesh,
        config,
        EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH,
        "updated grouped MuonH MoE params",
    )
    assert_grouped_moe_consumer_sharding(
        outputs,
        mesh,
        config,
        EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH,
        "grouped MuonH MoE outputs",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.reduce_scatter > 0
    assert hlo_summary.all_to_all == 0
    assert estimated_ns_dot_flops(
        config, EXPERT_GROUPED_MUONH_MOE_MLP_CONSUMER_BENCH
    ) > grouped_expert_bank_consumer_flops(config)


def test_grouped_moe_mlp_consumer_can_chunk_routed_tokens():
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
        grouped_expert_consumer_tokens_per_expert=3,
        grouped_expert_consumer_chunk_tokens_per_expert=1,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_grouped_expert_specs(mesh, config, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH)
    routed_inputs = synthetic_grouped_moe_mlp_consumer_input_specs(
        mesh,
        config,
        EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH,
    )
    update_step = jax.jit(grouped_moe_mlp_consumer_step_factory(mesh, config))

    assert grouped_moe_consumer_chunk_tokens(config) == 8
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, routed_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, routed_inputs).lower(lowering_platforms=(platform,))

    assert_grouped_moe_consumer_sharding(result, mesh, config, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH, "result")
    assert result["blocks"][0]["x"].shape == (4, 24, 16)
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general >= 2
    assert hlo_summary.all_gather > 0
    assert hlo_summary.reduce_scatter > 0
    assert grouped_expert_bank_consumer_flops(config) == 4 * 8 * 3 * (2 * 16 * 16 + 2 * 8 * 16)


def test_grouped_moe_mlp_consumer_skips_without_expert_parallel_axis():
    config = BenchConfig(
        layers=1,
        ns4d_group_size=1,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        packed_master_layer_chunk_size=4,
    )

    assert bench_skip_reason(config, EXPERT_GROUPED_MOE_MLP_CONSUMER_BENCH) is not None


def test_hlo_summary_counts_gpu_custom_calls():
    hlo = """
    %custom = "stablehlo.custom_call"(%arg0, %arg1) {
      call_target_name = "__cublas$gemm"
    } : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
    """

    summary = summarize_hlo(hlo)

    assert summary.custom_call == 1
    assert summary.gpu_gemm_custom_call == 1


def test_grouped_expert_layer_slice_boundary_times_compile_only():
    config = BenchConfig(
        layers=1,
        ns4d_group_size=1,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=1,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        packed_master_layer_chunk_size=4,
    )
    mesh = create_mesh(replica_axis=1, data_axis=1, expert_axis=1, model_axis=1)

    timing = time_ns4d(
        mesh,
        config,
        EXPERT_GROUPED_LAYER_SLICE_BENCH,
        warmup=0,
        iters=0,
        compile_only=True,
        compiled_hlo_output=None,
        abstract_mesh_enabled=False,
        allow_boundary_collectives=True,
        require_no_boundary_collectives=False,
        profile_dir=None,
        boundary_correctness_max_global_bytes=1 << 30,
        force_boundary_correctness=False,
    )

    assert timing.compiled_hlo.all_reduce == 0
    assert timing.compiled_hlo.all_to_all == 0


def test_grouped_expert_single_layer_slice_boundary_times_compile_only():
    config = BenchConfig(
        layers=1,
        ns4d_group_size=1,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=1,
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
    mesh = create_mesh(replica_axis=1, data_axis=1, expert_axis=1, model_axis=1)

    timing = time_ns4d(
        mesh,
        config,
        EXPERT_GROUPED_SINGLE_LAYER_SLICE_BENCH,
        warmup=0,
        iters=0,
        compile_only=True,
        compiled_hlo_output=None,
        abstract_mesh_enabled=False,
        allow_boundary_collectives=True,
        require_no_boundary_collectives=False,
        profile_dir=None,
        boundary_correctness_max_global_bytes=1 << 30,
        force_boundary_correctness=False,
    )

    assert timing.compiled_hlo.all_reduce == 0
    assert timing.compiled_hlo.all_to_all == 0


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


@pytest.mark.parametrize(
    ("packed_entry", "chunk_local_boundaries", "expected_all_gather", "expected_all_to_all"),
    [(False, False, 6, 0), (True, False, 2, 4), (False, True, 2, 4)],
)
def test_real_expert_fsdp_grouped_muonh_optimizer_uses_fsdp_params_and_outputs(
    packed_entry: bool,
    chunk_local_boundaries: bool,
    expected_all_gather: int,
    expected_all_to_all: int,
):
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
        expert_grouped_muonh_packed_entry=packed_entry,
        expert_grouped_muonh_chunk_local_boundaries=chunk_local_boundaries,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    optimizer = build_real_expert_fsdp_grouped_muonh_optimizer(config)
    apply_step = jax.jit(real_expert_fsdp_grouped_muonh_optimizer_apply_step_factory(config))
    update_step = jax.jit(real_expert_fsdp_grouped_muonh_optimizer_update_step_factory(config))

    assert ns4d_compute_sharding(mesh, config, REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH).spec == P(
        ("replica_dcn", "data"),
        "expert",
        None,
        None,
    )
    assert ns4d_compute_sharding(mesh, config, REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH).spec == P(
        ("replica_dcn", "data"),
        "expert",
        None,
        None,
    )
    assert_expert_fsdp_sharding(params, "real grouped MuonH expert FSDP params")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        state = jax.eval_shape(optimizer.init, params)
        result, _next_state = jax.eval_shape(apply_step, params, grads, state)
        updates, _next_update_state = jax.eval_shape(update_step, params, grads, state)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = apply_step.trace(params, grads, state).lower(lowering_platforms=(platform,))
        lowered_update = update_step.trace(params, grads, state).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "real grouped MuonH expert FSDP apply result")
    assert_expert_fsdp_sharding(updates, "real grouped MuonH expert FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    update_hlo_summary = summarize_hlo(str(lowered_update.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert update_hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_gather == expected_all_gather
    assert update_hlo_summary.all_gather == expected_all_gather
    assert hlo_summary.all_to_all == expected_all_to_all
    assert update_hlo_summary.all_to_all == expected_all_to_all
    assert hlo_summary.all_reduce == 0
    assert update_hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert update_hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_APPLY_BENCH) == 4 * 16
    assert estimated_matrix_count(config, REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH) == 4 * 16


def test_expert_fsdp_grouped_trace_muonh_keeps_trace_grouped_and_params_fsdp():
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
    grouped_grads = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_TRACE_MUONH_APPLY_BENCH)
    grouped_trace = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_TRACE_MUONH_APPLY_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_trace_muonh_apply_step_factory(mesh, config))

    assert ns4d_compute_sharding(mesh, config, EXPERT_FSDP_GROUPED_TRACE_MUONH_APPLY_BENCH).spec == P(
        ("replica_dcn", "data"),
        "expert",
        None,
        None,
    )
    assert_expert_fsdp_sharding(params, "grouped-trace MuonH FSDP params")
    assert_grouped_expert_sharding(
        grouped_trace,
        mesh,
        config,
        EXPERT_FSDP_GROUPED_TRACE_MUONH_APPLY_BENCH,
        "grouped-trace MuonH trace",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result, next_trace, restored_updates = jax.eval_shape(update_step, params, grouped_grads, grouped_trace)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_grads, grouped_trace).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(restored_updates, "grouped-trace MuonH restored updates")
    assert_expert_fsdp_sharding(result, "grouped-trace MuonH FSDP apply result")
    assert_grouped_expert_sharding(
        next_trace,
        mesh,
        config,
        EXPERT_FSDP_GROUPED_TRACE_MUONH_APPLY_BENCH,
        "grouped-trace MuonH next trace",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, EXPERT_FSDP_GROUPED_TRACE_MUONH_APPLY_BENCH) == 4 * 16


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
    assert is_expert_fsdp_grouped_bench(EXPERT_FSDP_GROUPED_EXPLICIT_TUPLE_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH)
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


def test_expert_fsdp_grouped_target_apply_boundary_keeps_grouped_fsdp_layout():
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
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_target_apply_boundary_step_factory(config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_grouped_expert_target_fsdp_sharding(result, "target apply grouped FSDP result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_TARGET_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_target_apply_chunked_boundary_keeps_grouped_fsdp_layout_without_collectives():
    config = BenchConfig(
        layers=6,
        ns4d_group_size=6,
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_TARGET_APPLY_CHUNKED_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_target_apply_chunked_boundary_step_factory(config))

    assert grouped_updates["blocks"][0]["mlp"]["expert_mlp"]["w_gate_up"].shape[0] == 8
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert len(result["blocks"]) == 2
    assert result["blocks"][0]["mlp"]["expert_mlp"]["w_gate_up"].shape[0] == 4
    assert result["blocks"][1]["mlp"]["expert_mlp"]["w_gate_up"].shape[0] == 4
    assert_grouped_expert_target_fsdp_sharding(result, "chunked target apply grouped FSDP result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_TARGET_APPLY_CHUNKED_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_target_apply_chunked_fsdp_boundary_returns_fsdp_layout_without_collectives():
    config = BenchConfig(
        layers=6,
        ns4d_group_size=6,
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_TARGET_APPLY_CHUNKED_FSDP_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_target_apply_chunked_fsdp_boundary_step_factory(config))

    assert grouped_updates["blocks"][0]["mlp"]["expert_mlp"]["w_gate_up"].shape[0] == 8
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert len(result["layers"]) == 6
    assert_expert_fsdp_sharding(result, "chunked target apply FSDP result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_TARGET_APPLY_CHUNKED_FSDP_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_restore_boundary_returns_fsdp_updates():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_restore_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit restored FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_RESTORE_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_apply_boundary_returns_fsdp_params():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_apply_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH) == 0


def test_fsdp_grads_to_grouped_chunks_returns_grouped_updates():
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
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(fsdp_grads_to_grouped_chunks_step_factory(mesh, config))

    assert_expert_fsdp_sharding(grads, "expert FSDP grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(grads).lower(lowering_platforms=(platform,))

    assert_grouped_expert_sharding(
        result,
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_GROUPED_CHUNKS_BENCH,
        "FSDP grads-to-grouped chunks result",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GRADS_TO_GROUPED_CHUNKS_BENCH) == 0


def test_fsdp_grads_to_packed_grouped_chunks_returns_grouped_updates():
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
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(fsdp_grads_to_packed_grouped_chunks_step_factory(mesh, config))

    assert_expert_fsdp_sharding(grads, "expert FSDP grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(grads).lower(lowering_platforms=(platform,))

    assert_grouped_expert_sharding(
        result,
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_PACKED_GROUPED_CHUNKS_BENCH,
        "packed FSDP grads-to-grouped chunks result",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GRADS_TO_PACKED_GROUPED_CHUNKS_BENCH) == 0


def test_fsdp_grads_to_explicit_packed_grouped_chunks_returns_grouped_updates():
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
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(fsdp_grads_to_explicit_packed_grouped_chunks_step_factory(mesh, config))

    assert_expert_fsdp_sharding(grads, "expert FSDP grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(grads).lower(lowering_platforms=(platform,))

    assert_grouped_expert_sharding(
        result,
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_CHUNKS_BENCH,
        "explicit packed FSDP grads-to-grouped chunks result",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_CHUNKS_BENCH) == 0


def test_fsdp_grads_to_explicit_packed_grouped_bank_returns_packed_bank():
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
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(fsdp_grads_to_explicit_packed_grouped_bank_step_factory(mesh, config))

    assert_expert_fsdp_sharding(grads, "expert FSDP grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(grads).lower(lowering_platforms=(platform,))

    assert_packed_grouped_expert_bank_sharding(
        result,
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
        "explicit packed FSDP grads-to-grouped bank result",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH) == 0


def test_fsdp_grads_to_explicit_packed_grouped_bank_pads_for_r2d2_group_axis():
    config = BenchConfig(
        layers=5,
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
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(fsdp_grads_to_explicit_packed_grouped_bank_step_factory(mesh, config))

    assert grouped_expert_group_sizes_for_bench(
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
    ) == (4, 4)
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(grads).lower(lowering_platforms=(platform,))

    assert result["packed"]["w_gate_up"].shape[0] == 8
    assert_packed_grouped_expert_bank_sharding(
        result,
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
        "explicit packed FSDP grads-to-grouped bank R2D2 result",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.collective_permute == 0
    assert hlo_summary.reduce_scatter == 0


def test_expert_fsdp_grouped_boundary_correctness_max_error_is_zero_for_reference_apply():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
    grouped_updates = make_grouped_expert_array_tree(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
        seed=1,
    )

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
        params,
        grouped_updates,
    )

    assert max_error == 0.0


def test_fsdp_grads_to_grouped_chunks_correctness_max_error_is_zero_for_reference():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    grads = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=1)

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_GROUPED_CHUNKS_BENCH,
        None,
        grads,
    )

    assert max_error == 0.0


def test_fsdp_grads_to_packed_grouped_chunks_correctness_max_error_is_zero_for_reference():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    grads = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=1)

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_PACKED_GROUPED_CHUNKS_BENCH,
        None,
        grads,
    )

    assert max_error == 0.0


def test_fsdp_grads_to_explicit_packed_grouped_chunks_correctness_max_error_is_zero_for_reference():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    grads = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=1)

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_CHUNKS_BENCH,
        None,
        grads,
    )

    assert max_error == 0.0


def test_fsdp_grads_to_explicit_packed_grouped_bank_correctness_max_error_is_zero_for_reference():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    grads = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=1)

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
        None,
        grads,
    )

    assert max_error == 0.0


def test_boundary_correctness_gate_reports_size_skip_reason():
    config = BenchConfig(
        layers=26,
        ns4d_group_size=2,
        ns4d_group_axis="data",
        hidden_dim=2560,
        intermediate_dim=1280,
        num_experts=256,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=3,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=512,
        replica_axis=1,
        data_axis=2,
        expert_axis=8,
        model_axis=1,
        learning_rate=0.02,
    )

    skip_reason = boundary_correctness_skipped_reason(
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
        max_global_bytes=1 << 30,
        force=False,
    )
    forced_skip_reason = boundary_correctness_skipped_reason(
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
        max_global_bytes=1 << 30,
        force=True,
    )

    assert skip_reason is not None
    assert "exceed correctness cap" in skip_reason
    assert forced_skip_reason is None


def test_fsdp_grads_to_explicit_packed_grouped_bank_timing_returns_scalar_checksum():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    grads = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=1)
    timing_step = jax.jit(
        fsdp_grads_to_grouped_chunks_timing_step_factory_for_bench(
            mesh,
            config,
            EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
        )
    )

    checksum = timing_step(grads)

    assert checksum.shape == ()


def test_expert_fsdp_packed_bank_a2a_apply_boundary_correctness_max_error_is_zero_for_reference():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
    packed_updates = make_packed_grouped_expert_bank_tree(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
        seed=1,
    )

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
        params,
        packed_updates,
    )

    assert max_error == 0.0


def test_expert_fsdp_packed_bank_direct_apply_boundary_correctness_max_error_is_zero_for_reference():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
    packed_updates = make_packed_grouped_expert_bank_tree(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH,
        seed=1,
    )

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH,
        params,
        packed_updates,
    )

    assert max_error == 0.0


def test_expert_fsdp_packed_bank_slice_first_direct_apply_boundary_correctness_max_error_is_zero_for_reference():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=2,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    params = make_array_tree(config, synthetic_fsdp_expert_shardings(mesh, config), seed=0)
    packed_updates = make_packed_grouped_expert_bank_tree(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_SLICE_FIRST_DIRECT_APPLY_BOUNDARY_BENCH,
        seed=1,
    )

    max_error = fsdp_grouped_boundary_correctness_max_error(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_SLICE_FIRST_DIRECT_APPLY_BOUNDARY_BENCH,
        params,
        packed_updates,
    )

    assert max_error == 0.0


def test_expert_fsdp_packed_bank_a2a_apply_boundary_returns_fsdp_params():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
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
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    packed_updates = synthetic_packed_grouped_expert_bank_specs(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_packed_bank_a2a_apply_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_packed_grouped_expert_bank_sharding(
        packed_updates,
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
        "packed grouped updates",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, packed_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, packed_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "packed-bank a2a apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_packed_bank_direct_apply_boundary_returns_fsdp_params():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
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
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    packed_updates = synthetic_packed_grouped_expert_bank_specs(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_packed_bank_direct_apply_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_packed_grouped_expert_bank_sharding(
        packed_updates,
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH,
        "direct packed grouped updates",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, packed_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, packed_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "packed-bank direct apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH) == 0


@pytest.mark.parametrize(
    ("bench_kind", "step_factory"),
    [
        (
            EXPERT_FSDP_PACKED_BANK_SLICE_FIRST_APPLY_BOUNDARY_BENCH,
            expert_fsdp_packed_bank_slice_first_apply_boundary_step_factory,
        ),
        (
            EXPERT_FSDP_PACKED_BANK_SLICE_FIRST_DIRECT_APPLY_BOUNDARY_BENCH,
            expert_fsdp_packed_bank_slice_first_direct_apply_boundary_step_factory,
        ),
    ],
)
def test_expert_fsdp_packed_bank_slice_first_apply_boundaries_return_fsdp_params_without_a2a(
    bench_kind,
    step_factory,
):
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
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
    params = synthetic_fsdp_expert_specs(mesh, config)
    packed_updates = synthetic_packed_grouped_expert_bank_specs(mesh, config, bench_kind)
    update_step = jax.jit(step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "slice-first packed-bank FSDP params")
    assert_packed_grouped_expert_bank_sharding(
        packed_updates,
        mesh,
        config,
        bench_kind,
        "slice-first packed grouped updates",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, packed_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, packed_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "slice-first packed-bank apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, bench_kind) == 0


def test_expert_fsdp_packed_bank_a2a_apply_boundary_pads_for_r2d2_group_axis():
    config = BenchConfig(
        layers=5,
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
    packed_updates = synthetic_packed_grouped_expert_bank_specs(
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_packed_bank_a2a_apply_boundary_step_factory(mesh, config))

    assert grouped_expert_group_sizes_for_bench(
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
    ) == (4, 4)
    assert packed_updates["packed"]["w_gate_up"].shape[0] == 8
    assert_packed_grouped_expert_bank_sharding(
        packed_updates,
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
        "packed grouped R2D2 updates",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, packed_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, packed_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "packed-bank R2D2 a2a apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.collective_permute == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0


def test_expert_fsdp_packed_bank_muonh_apply_returns_fsdp_params():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
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
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(expert_fsdp_packed_bank_muonh_apply_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "packed-bank MuonH params")
    assert_expert_fsdp_sharding(grads, "packed-bank MuonH grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "packed-bank MuonH FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_to_all > 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_PACKED_BANK_MUONH_APPLY_BENCH) > 0


def test_expert_fsdp_packed_bank_muonh_direct_apply_returns_fsdp_params():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
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
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(expert_fsdp_packed_bank_muonh_direct_apply_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "packed-bank MuonH direct-apply params")
    assert_expert_fsdp_sharding(grads, "packed-bank MuonH direct-apply grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "packed-bank MuonH direct-apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_to_all > 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_PACKED_BANK_MUONH_DIRECT_APPLY_BENCH) > 0
    assert estimated_matrix_count(config, EXPERT_FSDP_PACKED_BANK_MUONH_DIRECT_APPLY_BENCH) > 0


def test_expert_fsdp_packed_bank_muonh_restores_fsdp_updates_before_apply():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
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
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(expert_fsdp_packed_bank_muonh_updates_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "packed-bank MuonH params")
    assert_expert_fsdp_sharding(grads, "packed-bank MuonH grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        updates = jax.eval_shape(update_step, params, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "packed-bank MuonH restored FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0


def test_expert_fsdp_packed_bank_muonh_update_only_returns_packed_bank():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
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
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(expert_fsdp_packed_bank_muonh_update_only_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "packed-bank MuonH update-only params")
    assert_expert_fsdp_sharding(grads, "packed-bank MuonH update-only grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads).lower(lowering_platforms=(platform,))

    assert_packed_grouped_expert_bank_sharding(
        result,
        mesh,
        config,
        EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH,
        "packed-bank MuonH update-only result",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_to_all > 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH) > 0


def test_expert_fsdp_packed_bank_muonh_update_only_timing_returns_scalar_checksum():
    config = BenchConfig(
        layers=2,
        ns4d_group_size=2,
        ns4d_group_axis="data",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=4,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=4,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 1, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(expert_fsdp_packed_bank_muonh_update_only_timing_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grads)

    assert result.shape == ()
    assert result.dtype == jnp.float32


def test_packed_master_rebuild_tree_matches_packed_slices_numerically():
    config = BenchConfig(
        layers=3,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        seed=0,
    )
    metadata = muon_master_bank_metadata(config, EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH, mesh)
    grouped_view = packed_master_bank_to_grouped_expert_tree(
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), master_bank),
    )

    assert metadata.group_sizes == (2, 1)
    assert metadata.leaves[0].packed_path == "packed.w_gate_up[0:2]"
    assert metadata.leaves[0].model_tree_path == "blocks[0].mlp.expert_mlp.w_gate_up"
    assert metadata.leaves[0].target_fsdp_sharding_spec == "P('expert', 'data', 'model')"

    layer_index = 0
    for block in grouped_view["blocks"]:
        block_size = block["mlp"]["expert_mlp"]["w_gate_up"].shape[0]
        for local_index in range(block_size):
            for name in synthetic_shapes(config):
                rebuilt = block["mlp"]["expert_mlp"][name][local_index]
                expected = master_bank["packed"][name][layer_index].astype(jnp.bfloat16)
                assert jnp.array_equal(rebuilt, expected)
            layer_index += 1


def test_chunked_packed_master_rebuild_tree_matches_physical_chunk_slices_numerically():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        packed_master_layer_chunk_size=4,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH
    chunked_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=3)
    grouped_view = rebuild_expert_tree_from_muon_master_bank(config, bench_kind, chunked_bank)

    assert isinstance(chunked_bank, MuonMasterBank)
    assert chunked_packed_master_valid_group_sizes_for_bench(config, bench_kind) == (4, 1)
    assert len(grouped_view["blocks"]) == 2
    assert grouped_view["blocks"][0]["mlp"]["expert_mlp"]["w_gate_up"].shape[0] == 4
    assert grouped_view["blocks"][1]["mlp"]["expert_mlp"]["w_gate_up"].shape[0] == 4

    layer_index = 0
    valid_group_sizes = chunked_packed_master_valid_group_sizes_for_bench(config, bench_kind)
    for chunk_index, (block, valid_group_size) in enumerate(zip(grouped_view["blocks"], valid_group_sizes, strict=True)):
        block_size = block["mlp"]["expert_mlp"]["w_gate_up"].shape[0]
        for local_index in range(block_size):
            for name in synthetic_shapes(config):
                rebuilt = block["mlp"]["expert_mlp"][name][local_index]
                expected = chunked_bank["chunks"][chunk_index]["packed"][name][local_index].astype(jnp.bfloat16)
                assert jnp.array_equal(rebuilt, expected), (
                    f"chunked rebuilt layer {layer_index} {name} did not match "
                    f"chunk {chunk_index} local layer {local_index}"
                )
            if local_index < valid_group_size:
                layer_index += 1
    assert layer_index == config.layers


def test_layer_chunked_packed_master_grouped_grad_muonh_update_allows_padded_r4_tail():
    config = BenchConfig(
        layers=6,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=4,
        data_axis=1,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(4, 1, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_GROUPED_GRAD_MUONH_UPDATE_BENCH
    assert chunked_packed_master_valid_group_sizes_for_bench(config, bench_kind) == (4, 2)
    assert grouped_expert_group_sizes_for_bench(config, bench_kind) == (4, 4)

    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    activations = synthetic_grouped_expert_consumer_input_specs(mesh, config, bench_kind)
    update_step = jax.jit(expert_chunked_packed_master_grouped_grad_muonh_update_step_factory(config, bench_kind))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, activations).lower(lowering_platforms=(platform,))

    assert len(next_master["chunks"]) == 2
    assert len(next_momentum["chunks"]) == 2
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        bench_kind,
        "R4 padded-tail grouped-view grad MuonH next master",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_grouped_consumer_grad_matches_rebuilt_grouped_view():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        packed_master_layer_chunk_size=4,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH
    chunked_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=5)
    activations = make_grouped_expert_consumer_input_tree(mesh, config, bench_kind, seed=6)

    def direct_grouped_loss(bank):
        grouped_view = chunked_packed_master_bank_to_grouped_expert_tree(
            config,
            bench_kind,
            jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), bank),
        )
        outputs = grouped_expert_bank_consumer_outputs(config, grouped_view, activations)
        return sum(jnp.sum(leaf.astype(jnp.float32)) for leaf in jax.tree.leaves(outputs))

    actual = jax.grad(
        lambda bank: expert_chunked_packed_master_grouped_consumer_loss(config, bench_kind, bank, activations)
    )(chunked_bank)
    expected = jax.grad(direct_grouped_loss)(chunked_bank)

    assert_chunked_packed_grouped_expert_bank_sharding(
        actual,
        mesh,
        config,
        bench_kind,
        "chunked packed master grouped consumer grad",
    )
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert jnp.array_equal(actual_leaf, expected_leaf)


def test_chunked_packed_master_grouped_consumer_grad_lowers_with_packed_sharding():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    activations = synthetic_grouped_expert_consumer_input_specs(mesh, config, bench_kind)
    grad_step = jax.jit(expert_chunked_packed_master_grouped_consumer_grad_step_factory(config, bench_kind))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        grad_specs = jax.eval_shape(grad_step, master_bank, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = grad_step.trace(master_bank, activations).lower(lowering_platforms=(platform,))

    assert_chunked_packed_grouped_expert_bank_sharding(
        grad_specs,
        mesh,
        config,
        bench_kind,
        "chunked packed master grouped consumer grad output",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_layer_chunked_packed_master_grouped_grad_muonh_update_preserves_packed_state():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_GROUPED_GRAD_MUONH_UPDATE_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    activations = synthetic_grouped_expert_consumer_input_specs(mesh, config, bench_kind)
    update_step = jax.jit(expert_chunked_packed_master_grouped_grad_muonh_update_step_factory(config, bench_kind))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, activations).lower(lowering_platforms=(platform,))

    assert len(next_master["chunks"]) == 1
    assert len(next_momentum["chunks"]) == 1
    for chunk in next_master["chunks"]:
        for leaf in chunk["packed"].values():
            assert leaf.dtype == jnp.float32
    for chunk in next_momentum["chunks"]:
        for leaf in chunk["packed"].values():
            assert leaf.dtype == jnp.float32
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        bench_kind,
        "grouped-view grad MuonH next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        bench_kind,
        "grouped-view grad MuonH next momentum",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_fsdp_layer_matches_chunked_bank_slices_numerically():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH
    chunked_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=4)
    group_sizes = grouped_expert_group_sizes_for_bench(config, bench_kind)

    def chunk_index_and_local_layer(layer_index: int) -> tuple[int, int]:
        layer_offset = 0
        for group_index, group_size in enumerate(group_sizes):
            if layer_index < layer_offset + group_size:
                return group_index, layer_index - layer_offset
            layer_offset += group_size
        raise AssertionError(f"layer_index={layer_index} was outside grouped packed chunks")

    for layer_index in (0, 2, 4):
        materialized = materialize_expert_layer_from_muon_master_bank(
            mesh,
            config,
            bench_kind,
            chunked_bank,
            layer_index=layer_index,
        )
        group_index, local_layer_index = chunk_index_and_local_layer(layer_index)
        for name in synthetic_shapes(config):
            rebuilt = materialized["mlp"]["expert_mlp"][name]
            expected = chunked_bank["chunks"][group_index]["packed"][name][local_layer_index].astype(jnp.bfloat16)
            assert jnp.array_equal(rebuilt, expected), (
                f"chunked packed layer {layer_index} {name} did not match "
                f"chunk {group_index} local layer {local_layer_index}"
            )


def test_muon_master_bank_block_group_materialization_matches_chunked_bank_slices_numerically():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        packed_master_layer_chunk_size=4,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_VALUE_GRAD_MUONH_UPDATE_BENCH
    master_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=21)

    slabs = materialize_expert_block_group_from_muon_master_bank(
        mesh,
        config,
        bench_kind,
        master_bank,
        group_index=0,
    )

    assert isinstance(master_bank, MuonMasterBank)
    assert len(slabs) == 4
    for local_layer_index, slab in enumerate(slabs):
        for name in synthetic_shapes(config):
            rebuilt = slab["mlp"]["expert_mlp"][name][0]
            expected = master_bank["chunks"][0]["packed"][name][local_layer_index].astype(jnp.bfloat16)
            assert jnp.array_equal(
                rebuilt, expected
            ), f"block-group slab for {name} local layer {local_layer_index} did not match packed master"


def test_muon_master_bank_block_group_materialization_lowers_with_fsdp_slab_sharding():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_VALUE_GRAD_MUONH_UPDATE_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)

    def materialize(master):
        return materialize_expert_block_group_from_muon_master_bank(
            mesh,
            config,
            bench_kind,
            master,
            group_index=0,
        )

    materialize_step = jax.jit(materialize)
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        slab_specs = jax.eval_shape(materialize_step, master_bank)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = materialize_step.trace(master_bank).lower(lowering_platforms=(platform,))

    assert len(slab_specs) == 1
    slab = slab_specs[0]["mlp"]["expert_mlp"]
    assert slab["w_gate_up"].sharding.spec == P(None, "expert", "data", "model")
    assert slab["w_down"].sharding.spec == P(None, "expert", "model", "data")
    assert slab["w_gate_up"].shape == (4, config.num_experts, config.hidden_dim, 2 * config.intermediate_dim)
    assert slab["w_down"].shape == (4, config.num_experts, config.intermediate_dim, config.hidden_dim)
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_muon_master_bank_block_group_consumer_matches_chunk_consumer_numerically():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=2,
        packed_master_layer_chunk_size=4,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_BLOCK_GROUP_VALUE_GRAD_MUONH_UPDATE_BENCH
    master_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=23)
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.grouped_expert_consumer_tokens_per_expert * config.hidden_dim)
        .reshape(config.num_experts, config.grouped_expert_consumer_tokens_per_expert, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None, None)),
    )

    block_group_loss = expert_chunked_packed_master_fsdp_block_group_consumer_loss(
        mesh,
        config,
        master_bank,
        expert_inputs,
        group_index=0,
        bench_kind=bench_kind,
    )
    chunk_block_group_loss = expert_chunked_packed_master_fsdp_block_group_chunk_consumer_loss(
        mesh,
        config,
        master_bank["chunks"][0],
        expert_inputs,
        group_index=0,
        bench_kind=bench_kind,
    )
    chunk_loss = expert_chunked_packed_master_fsdp_chunk_consumer_loss(
        mesh,
        config,
        master_bank["chunks"][0],
        expert_inputs,
        chunk_index=0,
        bench_kind=bench_kind,
    )

    assert jnp.allclose(block_group_loss, chunk_loss, atol=0, rtol=0)
    assert jnp.allclose(chunk_block_group_loss, chunk_loss, atol=0, rtol=0)


def test_layer_chunked_packed_master_fsdp_streaming_block_group_value_grad_muonh_update_lowers_with_packed_state():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_BLOCK_GROUP_VALUE_GRAD_MUONH_UPDATE_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(
        expert_chunked_packed_master_grad_muonh_update_step_factory_for_bench(
            mesh,
            config,
            bench_kind,
        )
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        loss, next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        bench_kind,
        "block-group value+grad streaming MuonH next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        bench_kind,
        "block-group value+grad streaming MuonH next momentum",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_muon_expert_state_block_group_train_step_keeps_authoritative_packed_state():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_BLOCK_GROUP_VALUE_GRAD_MUONH_UPDATE_BENCH
    state = MuonExpertState(
        master=synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind),
        momentum=synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind),
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    train_step = jax.jit(expert_muon_master_bank_block_group_train_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        loss, next_state = jax.eval_shape(train_step, state, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = train_step.trace(state, expert_inputs).lower(lowering_platforms=(platform,))

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert isinstance(next_state, MuonExpertState)
    assert isinstance(next_state.master, MuonMasterBank)
    assert isinstance(next_state.momentum, MuonMasterBank)
    assert next_state.master.metadata == state.master.metadata
    assert next_state.momentum.metadata == state.momentum.metadata
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_state.master,
        mesh,
        config,
        bench_kind,
        "MuonExpertState block-group train step next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_state.momentum,
        mesh,
        config,
        bench_kind,
        "MuonExpertState block-group train step next momentum",
    )
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(next_state.master))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(next_state.momentum))
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_summary_row_reports_packed_master_bank_dtypes_for_chunked_streaming_bench():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=2,
        ns4d_group_axis="replica_dcn",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH
    result = {
        "metadata": {
            "label": "layer_chunked_packed_master_streaming_grad_muonh_checksum",
            "bench_kind": bench_kind,
            "config": asdict(config),
            "devices": 2,
            "local_devices": 2,
            "process_count": 1,
            "ns4d_group_size": 2,
            "ns4d_padded_group_size": 2,
            "ns4d_input_sharding_spec": "P('expert', None, None)",
            "ns4d_compute_sharding_spec": "P('replica_dcn', 'expert', None, None)",
            "ns4d_result_sharding_spec": "P('replica_dcn', 'expert', None, None)",
            "ns4d_boundary_status": ns4d_boundary_status(config, bench_kind),
            "boundary_collectives_allowed": True,
            "boundary_collectives_required_absent": False,
            "grouped_expert_group_count": None,
            "grouped_expert_packed_bank_count": None,
            "muon_master_bank_metadata": asdict(muon_master_bank_metadata(config, bench_kind)),
            "group_estimates": [asdict(estimate) for estimate in estimate_grouping(config, bench_kind)],
            "grouped_2d_estimates": [],
        },
    }

    row = summary_row(result)

    assert row["muon_master_bank_master_dtype"] == str(jnp.dtype(jnp.float32))
    assert row["muon_master_bank_momentum_dtype"] == str(jnp.dtype(jnp.float32))
    assert row["muon_master_bank_consumer_dtype"] == str(jnp.dtype(jnp.bfloat16))
    assert row["muon_master_bank_group_axis"] == "replica_dcn"
    assert row["muon_master_bank_group_sizes"] == (2, 2, 2)
    assert row["muon_master_bank_leaf_count"] == 6


def test_layer_chunked_packed_master_chunk_size_override_controls_group_count():
    config = BenchConfig(
        layers=10,
        ns4d_group_size=8,
        ns4d_group_axis="replica_dcn",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH

    assert chunked_packed_master_valid_group_sizes_for_bench(config, bench_kind) == (2, 2, 2, 2, 2)
    assert grouped_expert_group_sizes_for_bench(config, bench_kind) == (2, 2, 2, 2, 2)

    chunk4_config = replace(config, packed_master_layer_chunk_size=4)
    assert chunked_packed_master_valid_group_sizes_for_bench(chunk4_config, bench_kind) == (4, 4, 2)
    assert grouped_expert_group_sizes_for_bench(chunk4_config, bench_kind) == (4, 4, 4)

    with pytest.raises(ValueError, match="divisible by the active NS sharding axis size"):
        grouped_expert_group_sizes_for_bench(replace(config, packed_master_layer_chunk_size=3), bench_kind)


def test_chunked_packed_master_fsdp_layer_grad_matches_direct_layer_grad_numerically():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=2,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH
    chunked_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=5)
    chunked_bank = jax.tree.map(lambda leaf: jnp.asarray(jax.device_get(leaf)), chunked_bank)
    assert isinstance(chunked_bank, MuonMasterBank)
    # With layers=5 and group_size=2, this is logical layer 2: first layer in the second chunk.
    group_index = 1
    local_layer_index = 0
    direct_layer = {
        "mlp": {
            "expert_mlp": {
                name: chunked_bank["chunks"][group_index]["packed"][name][local_layer_index]
                for name in synthetic_shapes(config)
            }
        }
    }
    x = jnp.full(
        (config.num_experts, config.grouped_expert_consumer_tokens_per_expert, config.hidden_dim),
        0.125,
        dtype=jnp.bfloat16,
    )

    def simple_expert_loss(layer):
        expert_mlp = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), layer["mlp"]["expert_mlp"])
        gate_up = jnp.einsum("etd,edi->eti", x, expert_mlp["w_gate_up"])
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = jax.nn.silu(gate) * up
        out = jnp.einsum("eti,eid->etd", hidden, expert_mlp["w_down"])
        return jnp.sum(out.astype(jnp.float32))

    def chunked_loss(bank):
        layer = {
            "mlp": {
                "expert_mlp": {
                    name: bank["chunks"][group_index]["packed"][name][local_layer_index].astype(jnp.bfloat16)
                    for name in synthetic_shapes(config)
                }
            }
        }
        return simple_expert_loss(layer)

    def direct_loss(layer):
        return simple_expert_loss(layer)

    chunked_grads = jax.grad(chunked_loss)(chunked_bank)
    direct_grads = jax.grad(direct_loss)(direct_layer)

    assert isinstance(chunked_grads, MuonMasterBank)
    assert chunked_grads.metadata == chunked_bank.metadata
    for candidate_group_index, chunk in enumerate(chunked_grads.chunks):
        for name in synthetic_shapes(config):
            for candidate_local_layer_index in range(chunk["packed"][name].shape[0]):
                chunked_grad = chunk["packed"][name][candidate_local_layer_index]
                if candidate_group_index == group_index and candidate_local_layer_index == local_layer_index:
                    expected = direct_grads["mlp"]["expert_mlp"][name]
                else:
                    expected = jnp.zeros_like(chunked_grad)
                assert jnp.array_equal(chunked_grad, expected), (
                    f"chunked packed grad for {name} chunk {candidate_group_index} "
                    f"local layer {candidate_local_layer_index} did not match direct layer grad"
                )


def test_chunked_packed_master_fsdp_sequential_grad_reaches_all_layer_chunks_numerically():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=2,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_UPDATE_BENCH
    chunked_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=6)
    chunked_bank = jax.tree.map(lambda leaf: jnp.asarray(jax.device_get(leaf)), chunked_bank)
    expert_inputs = jnp.full(
        (config.num_experts, config.grouped_expert_consumer_tokens_per_expert, config.hidden_dim),
        0.125,
        dtype=jnp.bfloat16,
    )

    def layer_loss(bank):
        expert_mlp = jax.tree.map(
            lambda leaf: leaf.astype(jnp.bfloat16),
            {name: bank["chunks"][0]["packed"][name][0] for name in synthetic_shapes(config)},
        )
        gate_up = jnp.einsum("etd,edi->eti", expert_inputs, expert_mlp["w_gate_up"])
        hidden = gate_up[:, :, : config.intermediate_dim]
        out = jnp.einsum("eti,eid->etd", hidden, expert_mlp["w_down"])
        return jnp.sum(out.astype(jnp.float32))

    def sequential_loss(bank):
        loss = jnp.asarray(0, dtype=jnp.float32)
        remaining_layers = config.layers
        for chunk in bank["chunks"]:
            chunk_layers = min(remaining_layers, chunk["packed"]["w_down"].shape[0])
            for local_layer_index in range(chunk_layers):
                expert_mlp = jax.tree.map(
                    lambda leaf: leaf.astype(jnp.bfloat16),
                    {name: chunk["packed"][name][local_layer_index] for name in synthetic_shapes(config)},
                )
                gate_up = jnp.einsum("etd,edi->eti", expert_inputs, expert_mlp["w_gate_up"])
                hidden = gate_up[:, :, : config.intermediate_dim]
                out = jnp.einsum("eti,eid->etd", hidden, expert_mlp["w_down"])
                loss = loss + jnp.sum(out.astype(jnp.float32))
            remaining_layers -= chunk_layers
        return loss

    layer_grad = jax.grad(layer_loss)(chunked_bank)
    sequential_grad = jax.grad(sequential_loss)(chunked_bank)

    for chunk_index, chunk in enumerate(sequential_grad["chunks"]):
        for name in synthetic_shapes(config):
            for local_layer_index in range(chunk["packed"][name].shape[0]):
                grad_sum = jnp.sum(jnp.abs(chunk["packed"][name][local_layer_index].astype(jnp.float32)))
                assert grad_sum > 0, f"sequential grad missed {name} chunk={chunk_index} local={local_layer_index}"

    for chunk_index, chunk in enumerate(layer_grad["chunks"]):
        for name in synthetic_shapes(config):
            for local_layer_index in range(chunk["packed"][name].shape[0]):
                grad_sum = jnp.sum(jnp.abs(chunk["packed"][name][local_layer_index].astype(jnp.float32)))
                if chunk_index == 0 and local_layer_index == 0:
                    assert grad_sum > 0
                else:
                    assert grad_sum == 0, f"one-layer grad unexpectedly touched {name} chunk={chunk_index}"


def test_expert_packed_master_muonh_consumer_keeps_packed_master_and_grouped_outputs():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    )
    grad_bank = synthetic_packed_grouped_expert_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    )
    momentum_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    )
    activations = synthetic_grouped_expert_consumer_input_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    )
    update_step = jax.jit(expert_packed_master_muonh_consumer_step_factory(config))

    assert_packed_grouped_expert_bank_sharding(
        master_bank,
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        "packed-master MuonH master input",
    )
    assert_packed_grouped_expert_bank_sharding(
        grad_bank,
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        "packed-master MuonH grad input",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_master, next_momentum, outputs = jax.eval_shape(
            update_step,
            master_bank,
            grad_bank,
            momentum_bank,
            activations,
        )
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, grad_bank, momentum_bank, activations).lower(
            lowering_platforms=(platform,)
        )

    assert next_master["packed"]["w_gate_up"].dtype == jnp.float32
    assert next_momentum["packed"]["w_down"].dtype == jnp.float32
    assert_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        "packed-master MuonH next master",
    )
    assert_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        "packed-master MuonH next momentum",
    )
    assert_ns4d_sharding(
        outputs,
        P(("replica_dcn", "data"), "expert", None, None),
        "packed-master MuonH grouped consumer outputs",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_expert_packed_master_consumer_grad_returns_packed_master_grads_without_collectives():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    )
    activations = synthetic_grouped_expert_consumer_input_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    )
    grad_step = jax.jit(expert_packed_master_consumer_grad_step_factory(config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        packed_grads = jax.eval_shape(grad_step, master_bank, activations)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = grad_step.trace(master_bank, activations).lower(lowering_platforms=(platform,))

    assert packed_grads["packed"]["w_gate_up"].dtype == jnp.float32
    assert packed_grads["packed"]["w_down"].dtype == jnp.float32
    assert_packed_grouped_expert_bank_sharding(
        packed_grads,
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        "packed-master consumer transposed grads",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_fsdp_layer_grad_returns_chunked_bank_grads_without_lowered_collectives():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    grad_step = jax.jit(expert_chunked_packed_master_fsdp_layer_grad_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        chunked_grads = jax.eval_shape(grad_step, master_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = grad_step.trace(master_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert_chunked_packed_grouped_expert_bank_sharding(
        chunked_grads,
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_BENCH,
        "chunked packed-master FSDP layer transposed grads",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_fsdp_layer_grad_muonh_update_preserves_chunked_banks():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH,
    )
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(expert_chunked_packed_master_fsdp_layer_grad_muonh_update_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH,
        "chunked packed-master layer-grad MuonH next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH,
        "chunked packed-master layer-grad MuonH next momentum",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_fsdp_sequential_grad_muonh_update_preserves_chunked_banks():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_UPDATE_BENCH,
    )
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_UPDATE_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(expert_chunked_packed_master_fsdp_sequential_grad_muonh_update_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_UPDATE_BENCH,
        "chunked packed-master sequential-grad MuonH next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_UPDATE_BENCH,
        "chunked packed-master sequential-grad MuonH next momentum",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_fsdp_sequential_grad_muonh_checksum_is_scalar():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_CHECKSUM_BENCH,
    )
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_CHECKSUM_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(expert_chunked_packed_master_fsdp_sequential_grad_muonh_checksum_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        checksum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert checksum.shape == ()
    assert checksum.dtype == jnp.float32
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_fsdp_streaming_grad_muonh_checksum_is_scalar():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
    )
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(
        expert_chunked_packed_master_fsdp_streaming_grad_muonh_checksum_step_factory(
            mesh,
            config,
            EXPERT_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
        )
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        checksum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert checksum.shape == ()
    assert checksum.dtype == jnp.float32
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_layer_chunked_packed_master_fsdp_streaming_grad_muonh_checksum_is_scalar():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
    )
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(
        expert_chunked_packed_master_fsdp_streaming_grad_muonh_checksum_step_factory(
            mesh,
            config,
            EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH,
        )
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        checksum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert checksum.shape == ()
    assert checksum.dtype == jnp.float32
    assert len(master_bank["chunks"]) == 1
    assert next(iter(master_bank["chunks"][0]["packed"].values())).shape[0] == 4
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_layer_chunked_packed_master_fsdp_streaming_grad_muonh_update_preserves_packed_state():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_UPDATE_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(
        expert_chunked_packed_master_fsdp_streaming_grad_muonh_update_step_factory(
            mesh,
            config,
            bench_kind,
        )
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert len(next_master["chunks"]) == 1
    assert len(next_momentum["chunks"]) == 1
    for chunk in next_master["chunks"]:
        for leaf in chunk["packed"].values():
            assert leaf.dtype == jnp.float32
    for chunk in next_momentum["chunks"]:
        for leaf in chunk["packed"].values():
            assert leaf.dtype == jnp.float32
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        bench_kind,
        "state-returning streaming MuonH next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        bench_kind,
        "state-returning streaming MuonH next momentum",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_layer_chunked_packed_master_fsdp_streaming_value_grad_muonh_update_returns_current_loss():
    config = BenchConfig(
        layers=3,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=2,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_VALUE_GRAD_MUONH_UPDATE_BENCH
    master_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=18)
    momentum_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=19)
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.grouped_expert_consumer_tokens_per_expert * config.hidden_dim)
        .reshape(config.num_experts, config.grouped_expert_consumer_tokens_per_expert, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None, None)),
    )
    update_step = jax.jit(
        expert_chunked_packed_master_fsdp_streaming_value_grad_muonh_update_step_factory(
            mesh,
            config,
            bench_kind,
        )
    )
    old_loss_jit = jax.jit(
        lambda master, inputs: expert_layer_chunked_packed_master_fsdp_sequential_consumer_loss(
            mesh,
            config,
            master,
            inputs,
        )
    )

    with mesh:
        loss, next_master, next_momentum = update_step(master_bank, momentum_bank, expert_inputs)
        old_loss = old_loss_jit(master_bank, expert_inputs)

    assert isinstance(master_bank, MuonMasterBank)
    assert isinstance(momentum_bank, MuonMasterBank)
    assert isinstance(next_master, MuonMasterBank)
    assert isinstance(next_momentum, MuonMasterBank)
    assert next_master.metadata == master_bank.metadata
    assert next_momentum.metadata == momentum_bank.metadata
    assert loss.shape == ()
    assert jnp.allclose(loss, old_loss, atol=0, rtol=0)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(next_master))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(next_momentum))
    assert any(
        not jnp.allclose(old_leaf, new_leaf, atol=0, rtol=0)
        for old_leaf, new_leaf in zip(jax.tree.leaves(master_bank), jax.tree.leaves(next_master), strict=True)
    )


def test_layer_chunked_packed_master_fsdp_streaming_value_grad_muonh_update_lowers_with_packed_state():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_VALUE_GRAD_MUONH_UPDATE_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(
        expert_chunked_packed_master_fsdp_streaming_value_grad_muonh_update_step_factory(
            mesh,
            config,
            bench_kind,
        )
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        loss_specs, next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert isinstance(master_bank, MuonMasterBank)
    assert isinstance(momentum_bank, MuonMasterBank)
    assert isinstance(next_master, MuonMasterBank)
    assert isinstance(next_momentum, MuonMasterBank)
    assert next_master.metadata == master_bank.metadata
    assert next_momentum.metadata == momentum_bank.metadata
    assert loss_specs.shape == ()
    assert loss_specs.dtype == jnp.float32
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        bench_kind,
        "value+grad streaming MuonH next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        bench_kind,
        "value+grad streaming MuonH next momentum",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_layer_chunked_packed_master_fsdp_streaming_layerwise_value_grad_muonh_update_lowers_with_packed_state():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_LAYERWISE_VALUE_GRAD_MUONH_UPDATE_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(
        expert_chunked_packed_master_fsdp_streaming_value_grad_muonh_update_step_factory(
            mesh,
            config,
            bench_kind,
        )
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        loss_specs, next_master, next_momentum = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert isinstance(next_master, MuonMasterBank)
    assert isinstance(next_momentum, MuonMasterBank)
    assert next_master.metadata == master_bank.metadata
    assert next_momentum.metadata == momentum_bank.metadata
    assert loss_specs.shape == ()
    assert loss_specs.dtype == jnp.float32
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        bench_kind,
        "layerwise value+grad streaming MuonH next master",
    )
    assert_chunked_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        bench_kind,
        "layerwise value+grad streaming MuonH next momentum",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_layer_chunked_packed_master_fsdp_sequential_consumer_lowers_as_scalar():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(mesh, config, bench_kind)
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    consume_step = jax.jit(expert_layer_chunked_packed_master_fsdp_sequential_consumer_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result_specs = jax.eval_shape(consume_step, master_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = consume_step.trace(master_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert result_specs.shape == ()
    assert result_specs.dtype == jnp.float32
    assert_chunked_packed_grouped_expert_bank_sharding(
        master_bank,
        mesh,
        config,
        bench_kind,
        "layer-chunked packed master FSDP consumer input",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_layer_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss_updates_before_consume():
    config = BenchConfig(
        layers=3,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=2,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_chunked_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH,
        seed=14,
    )
    momentum_bank = make_chunked_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH,
        seed=15,
    )
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.grouped_expert_consumer_tokens_per_expert * config.hidden_dim)
        .reshape(config.num_experts, config.grouped_expert_consumer_tokens_per_expert, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None, None)),
    )

    def actual_step(master, momentum, inputs):
        return expert_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss(
            mesh,
            config,
            master,
            momentum,
            inputs,
            bench_kind=EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH,
        )

    platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
    actual_jit = jax.jit(actual_step)
    old_loss_jit = jax.jit(
        lambda master, inputs: expert_chunked_packed_master_fsdp_sequential_consumer_loss(
            mesh,
            config,
            master,
            inputs,
        )
    )
    with mesh:
        actual_compiled = (
            actual_jit.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,)).compile()
        )
        old_loss_compiled = (
            old_loss_jit.trace(master_bank, expert_inputs).lower(lowering_platforms=(platform,)).compile()
        )
        actual = actual_compiled(master_bank, momentum_bank, expert_inputs)
        old_loss = old_loss_compiled(master_bank, expert_inputs)

    assert actual.shape == ()
    assert old_loss.shape == ()
    assert not jnp.allclose(actual, old_loss, atol=0, rtol=0)


def test_layer_chunked_packed_master_fsdp_chunk_consumer_matches_per_layer_materialization():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=2,
    )
    mesh = create_mesh(1, 1, 1, 1)
    bench_kind = EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH
    chunked_bank = make_chunked_packed_grouped_expert_master_bank_tree(mesh, config, bench_kind, seed=16)
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.grouped_expert_consumer_tokens_per_expert * config.hidden_dim)
        .reshape(config.num_experts, config.grouped_expert_consumer_tokens_per_expert, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None, None)),
    )
    chunk = chunked_bank["chunks"][0]

    slab_loss = expert_chunked_packed_master_fsdp_chunk_consumer_loss(
        mesh,
        config,
        chunk,
        expert_inputs,
        chunk_index=0,
        bench_kind=bench_kind,
    )

    per_layer_loss = jnp.asarray(0, dtype=jnp.float32)
    valid_group_size = len(chunk["packed"]["w_gate_up"])
    for local_layer_index in range(valid_group_size):
        layer = chunked_packed_master_chunk_to_fsdp_expert_layer(
            mesh,
            config,
            chunk,
            chunk_index=0,
            local_layer_index=local_layer_index,
            bench_kind=bench_kind,
        )
        per_layer_loss = per_layer_loss + expert_fsdp_layer_consumer_loss(mesh, config, layer, expert_inputs)

    assert jnp.allclose(slab_loss, per_layer_loss, atol=0, rtol=0)


def test_layer_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss_lowers_as_scalar():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=3,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    master_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH,
    )
    momentum_bank = synthetic_chunked_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(
        expert_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss_step_factory(
            mesh,
            config,
            EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH,
        )
    )

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_loss = jax.eval_shape(update_step, master_bank, momentum_bank, expert_inputs)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(master_bank, momentum_bank, expert_inputs).lower(lowering_platforms=(platform,))

    assert next_loss.shape == ()
    assert next_loss.dtype == jnp.float32
    assert len(master_bank["chunks"]) == 1
    assert next(iter(master_bank["chunks"][0]["packed"].values())).shape[0] == 4
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert hlo_summary.all_to_all == 0


def test_chunked_packed_master_grad_muonh_update_reports_specific_boundary_status():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=2,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )

    assert (
        ns4d_boundary_status(config, EXPERT_CHUNKED_PACKED_MASTER_FSDP_LAYER_GRAD_MUONH_UPDATE_BENCH)
        == "chunked_packed_master_one_fsdp_layer_grad_then_muonh_update"
    )
    assert (
        ns4d_boundary_status(config, EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_UPDATE_BENCH)
        == "chunked_packed_master_sequential_fsdp_layer_grads_then_muonh_update"
    )
    assert (
        ns4d_boundary_status(config, EXPERT_CHUNKED_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_MUONH_CHECKSUM_BENCH)
        == "chunked_packed_master_sequential_fsdp_layer_grads_then_muonh_checksum"
    )
    assert (
        ns4d_boundary_status(config, EXPERT_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH)
        == "chunked_packed_master_streaming_chunk_fsdp_layer_grads_then_muonh_checksum"
    )
    assert (
        ns4d_boundary_status(config, EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_CHECKSUM_BENCH)
        == "replica_aligned_chunked_packed_master_streaming_fsdp_grad_then_muonh_checksum"
    )
    assert (
        ns4d_boundary_status(config, EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_UPDATE_BENCH)
        == "replica_aligned_chunked_packed_master_streaming_fsdp_grad_then_muonh_update"
    )
    assert (
        ns4d_boundary_status(config, EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_GRAD_MUONH_NEXT_LOSS_BENCH)
        == "replica_aligned_chunked_packed_master_streaming_fsdp_grad_then_muonh_next_loss"
    )
    assert (
        ns4d_boundary_status(
            config,
            EXPERT_LAYER_CHUNKED_PACKED_MASTER_FSDP_STREAMING_BLOCK_GROUP_VALUE_GRAD_MUONH_UPDATE_BENCH,
        )
        == "replica_aligned_chunked_packed_master_streaming_block_group_fsdp_value_grad_then_muonh_update"
    )
    assert (
        ns4d_boundary_status(config, EXPERT_LAYER_CHUNKED_PACKED_MASTER_GROUPED_GRAD_MUONH_UPDATE_BENCH)
        == "replica_aligned_chunked_packed_master_grouped_view_grad_then_muonh_update"
    )


def test_expert_packed_master_consumer_grad_matches_grouped_tree_grad_numerically():
    config = BenchConfig(
        layers=3,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        grouped_expert_consumer_tokens_per_expert=2,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        seed=2,
    )
    grouped_master = packed_master_bank_to_grouped_expert_tree(
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        master_bank,
    )
    activations = make_grouped_expert_consumer_input_tree(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
        seed=3,
    )

    def grouped_loss(grouped_params):
        bf16_grouped_params = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), grouped_params)
        outputs = grouped_expert_bank_consumer_outputs(config, bf16_grouped_params, activations)
        return sum(jnp.sum(leaf.astype(jnp.float32)) for leaf in jax.tree.leaves(outputs))

    packed_grads = jax.grad(lambda bank: expert_packed_master_consumer_loss(config, bank, activations))(master_bank)
    grouped_grads = jax.grad(grouped_loss)(grouped_master)

    layer_index = 0
    for block_index, block in enumerate(grouped_grads["blocks"]):
        block_size = block["mlp"]["expert_mlp"]["w_gate_up"].shape[0]
        for local_index in range(block_size):
            for name in synthetic_shapes(config):
                rebuilt_grad = block["mlp"]["expert_mlp"][name][local_index]
                packed_grad = packed_grads["packed"][name][layer_index]
                assert jnp.allclose(packed_grad, rebuilt_grad, atol=0, rtol=0), (
                    f"packed grad for {name} layer {layer_index} did not match "
                    f"grouped block {block_index} local layer {local_index}"
                )
            layer_index += 1


def test_packed_master_fsdp_layer_consumer_materializes_one_use_site_layer():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
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
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_MUONH_CONSUMER_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)

    def materialize_layer(master):
        return packed_master_bank_to_fsdp_expert_layer(mesh, config, master, layer_index=1)

    materialize_step = jax.jit(materialize_layer)
    consume_step = jax.jit(expert_packed_master_fsdp_layer_consumer_step_factory(mesh, config, layer_index=1))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        layer = jax.eval_shape(materialize_step, master_bank)
        layer_lowered = materialize_step.trace(master_bank).lower(
            lowering_platforms=(jax.devices()[0].platform if jax.devices() else jax.default_backend(),)
        )
        loss = jax.eval_shape(consume_step, master_bank, expert_inputs)
        consume_lowered = consume_step.trace(master_bank, expert_inputs).lower(
            lowering_platforms=(jax.devices()[0].platform if jax.devices() else jax.default_backend(),)
        )

    assert_expert_fsdp_sharding(layer, "packed-master materialized FSDP layer")
    assert layer["mlp"]["expert_mlp"]["w_gate_up"].dtype == jnp.bfloat16
    assert loss.shape == ()
    assert loss.dtype == jnp.float32

    layer_hlo = summarize_hlo(str(layer_lowered.compiler_ir(dialect="stablehlo")))
    assert layer_hlo.all_reduce == 0
    assert layer_hlo.reduce_scatter == 0
    assert layer_hlo.all_to_all == 0

    consume_hlo = summarize_hlo(str(consume_lowered.compiler_ir(dialect="stablehlo")))
    assert consume_hlo.dot_general > 0
    assert consume_hlo.all_reduce == 0
    assert consume_hlo.reduce_scatter == 0
    assert consume_hlo.all_to_all == 0


def test_packed_master_fsdp_sequential_consumer_matches_layer_sum():
    config = BenchConfig(
        layers=3,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        EXPERT_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH,
        seed=4,
    )
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.hidden_dim, dtype=jnp.float32)
        .reshape(config.num_experts, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None)),
    )

    with mesh:
        sequential_loss = expert_packed_master_fsdp_sequential_consumer_loss(
            mesh,
            config,
            master_bank,
            expert_inputs,
        )
        expected_loss = sum(
            expert_packed_master_fsdp_layer_consumer_step_factory(mesh, config, layer_index)(
                master_bank,
                expert_inputs,
            )
            for layer_index in range(config.layers)
        )

    assert jnp.allclose(sequential_loss, expected_loss, atol=0, rtol=0)


def test_packed_master_fsdp_slab_consumer_matches_sequential_consumer():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        EXPERT_PACKED_MASTER_FSDP_SLAB_CONSUMER_BENCH,
        seed=5,
    )
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.hidden_dim, dtype=jnp.float32)
        .reshape(config.num_experts, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None)),
    )

    with mesh:
        slab_loss = expert_packed_master_fsdp_slab_consumer_loss(mesh, config, master_bank, expert_inputs)
        sequential_loss = expert_packed_master_fsdp_sequential_consumer_loss(mesh, config, master_bank, expert_inputs)

    assert jnp.allclose(slab_loss, sequential_loss, atol=0, rtol=0)


def test_packed_master_fsdp_bulk_consumer_matches_sequential_consumer():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        EXPERT_PACKED_MASTER_FSDP_BULK_CONSUMER_BENCH,
        seed=6,
    )
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.hidden_dim, dtype=jnp.float32)
        .reshape(config.num_experts, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None)),
    )

    with mesh:
        bulk_loss = expert_packed_master_fsdp_bulk_consumer_loss(mesh, config, master_bank, expert_inputs)
        sequential_loss = expert_packed_master_fsdp_sequential_consumer_loss(mesh, config, master_bank, expert_inputs)

    assert jnp.allclose(bulk_loss, sequential_loss, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("bench_kind", "outputs_fn", "consumer_loss_fn"),
    [
        (
            EXPERT_PACKED_MASTER_MUONH_FSDP_SEQUENTIAL_CONSUMER_BENCH,
            expert_packed_master_muonh_fsdp_sequential_consumer_outputs,
            expert_packed_master_fsdp_sequential_consumer_loss,
        ),
        (
            EXPERT_PACKED_MASTER_MUONH_FSDP_SLAB_CONSUMER_BENCH,
            expert_packed_master_muonh_fsdp_slab_consumer_outputs,
            expert_packed_master_fsdp_slab_consumer_loss,
        ),
        (
            EXPERT_PACKED_MASTER_MUONH_FSDP_BULK_CONSUMER_BENCH,
            expert_packed_master_muonh_fsdp_bulk_consumer_outputs,
            expert_packed_master_fsdp_bulk_consumer_loss,
        ),
    ],
)
def test_packed_master_muonh_fsdp_consumer_matches_separate_update_then_consume(
    bench_kind,
    outputs_fn,
    consumer_loss_fn,
):
    config = BenchConfig(
        layers=3,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        bench_kind,
        seed=7,
    )
    grad_bank = make_packed_grouped_expert_bank_tree(
        mesh,
        config,
        bench_kind,
        seed=8,
    )
    momentum_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        bench_kind,
        seed=9,
    )
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.hidden_dim, dtype=jnp.float32)
        .reshape(config.num_experts, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None)),
    )

    with mesh:
        actual_master, actual_momentum, actual_loss = outputs_fn(
            mesh,
            config,
            master_bank,
            grad_bank,
            momentum_bank,
            expert_inputs,
        )
        expected_master, expected_momentum = expert_packed_master_muonh_update_outputs(
            config,
            master_bank,
            grad_bank,
            momentum_bank,
        )
        expected_loss = consumer_loss_fn(mesh, config, expected_master, expert_inputs)

    for actual, expected in zip(jax.tree.leaves(actual_master), jax.tree.leaves(expected_master), strict=True):
        assert jnp.allclose(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(jax.tree.leaves(actual_momentum), jax.tree.leaves(expected_momentum), strict=True):
        assert jnp.allclose(actual, expected, atol=0, rtol=0)
    assert jnp.allclose(actual_loss, expected_loss, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("bench_kind", "consumer_loss_fn"),
    [
        (
            EXPERT_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_BENCH,
            expert_packed_master_fsdp_sequential_consumer_loss,
        ),
        (
            EXPERT_PACKED_MASTER_FSDP_SLAB_GRAD_BENCH,
            expert_packed_master_fsdp_slab_consumer_loss,
        ),
        (
            EXPERT_PACKED_MASTER_FSDP_BULK_GRAD_BENCH,
            expert_packed_master_fsdp_bulk_consumer_loss,
        ),
    ],
)
def test_packed_master_fsdp_grad_matches_direct_consumer_grad(bench_kind, consumer_loss_fn):
    config = BenchConfig(
        layers=3,
        ns4d_group_size=2,
        ns4d_group_axis="none",
        hidden_dim=8,
        intermediate_dim=4,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.bfloat16)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = create_mesh(1, 1, 1, 1)
    master_bank = make_packed_grouped_expert_master_bank_tree(
        mesh,
        config,
        bench_kind,
        seed=10,
    )
    expert_inputs = jax.device_put(
        jnp.arange(config.num_experts * config.hidden_dim, dtype=jnp.float32)
        .reshape(config.num_experts, config.hidden_dim)
        .astype(jnp.bfloat16),
        NamedSharding(mesh, P("expert", None)),
    )

    with mesh:
        actual_grad = expert_packed_master_fsdp_grad_outputs(
            mesh,
            config,
            bench_kind,
            master_bank,
            expert_inputs,
        )
        expected_grad = jax.grad(lambda bank: consumer_loss_fn(mesh, config, bank, expert_inputs))(master_bank)

    for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True):
        assert jnp.allclose(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "bench_kind",
    [
        EXPERT_PACKED_MASTER_FSDP_SEQUENTIAL_GRAD_BENCH,
        EXPERT_PACKED_MASTER_FSDP_SLAB_GRAD_BENCH,
        EXPERT_PACKED_MASTER_FSDP_BULK_GRAD_BENCH,
    ],
)
def test_packed_master_fsdp_grad_lowers_with_packed_grad_sharding(bench_kind):
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
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
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        bench_kind,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    grad_step = jax.jit(expert_packed_master_fsdp_grad_step_factory(mesh, config, bench_kind))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        grad_specs = jax.eval_shape(grad_step, master_bank, expert_inputs)
        lowered = grad_step.trace(master_bank, expert_inputs).lower(
            lowering_platforms=(jax.devices()[0].platform if jax.devices() else jax.default_backend(),)
        )

    assert_packed_grouped_expert_bank_sharding(
        grad_specs,
        mesh,
        config,
        bench_kind,
        "packed-master FSDP consumer grad",
    )
    hlo = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo.dot_general > 0
    if bench_kind != EXPERT_PACKED_MASTER_FSDP_BULK_GRAD_BENCH:
        assert hlo.all_reduce == 0
        assert hlo.reduce_scatter == 0


def test_packed_master_fsdp_sequential_consumer_lowers_without_reduction_collectives():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
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
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_FSDP_SEQUENTIAL_CONSUMER_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    consume_step = jax.jit(expert_packed_master_fsdp_sequential_consumer_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        loss = jax.eval_shape(consume_step, master_bank, expert_inputs)
        lowered = consume_step.trace(master_bank, expert_inputs).lower(
            lowering_platforms=(jax.devices()[0].platform if jax.devices() else jax.default_backend(),)
        )

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    hlo = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo.dot_general >= 2 * config.layers
    assert hlo.all_reduce == 0
    assert hlo.reduce_scatter == 0
    assert hlo.all_to_all == 0


def test_packed_master_fsdp_slab_consumer_lowers_without_reduction_collectives():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
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
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_FSDP_SLAB_CONSUMER_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    consume_step = jax.jit(expert_packed_master_fsdp_slab_consumer_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        loss = jax.eval_shape(consume_step, master_bank, expert_inputs)
        lowered = consume_step.trace(master_bank, expert_inputs).lower(
            lowering_platforms=(jax.devices()[0].platform if jax.devices() else jax.default_backend(),)
        )

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    hlo = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo.dot_general >= 2 * config.layers
    assert hlo.all_reduce == 0
    assert hlo.reduce_scatter == 0
    assert hlo.all_to_all == 0


def test_packed_master_fsdp_bulk_consumer_lowers_without_reduction_collectives():
    config = BenchConfig(
        layers=5,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
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
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        EXPERT_PACKED_MASTER_FSDP_BULK_CONSUMER_BENCH,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    consume_step = jax.jit(expert_packed_master_fsdp_bulk_consumer_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        loss = jax.eval_shape(consume_step, master_bank, expert_inputs)
        lowered = consume_step.trace(master_bank, expert_inputs).lower(
            lowering_platforms=(jax.devices()[0].platform if jax.devices() else jax.default_backend(),)
        )

    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    hlo = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo.dot_general >= 2 * config.layers
    assert hlo.all_reduce == 0
    assert hlo.reduce_scatter == 0
    assert hlo.all_to_all == 0


@pytest.mark.parametrize(
    ("bench_kind", "step_factory"),
    [
        (
            EXPERT_PACKED_MASTER_MUONH_FSDP_SEQUENTIAL_CONSUMER_BENCH,
            expert_packed_master_muonh_fsdp_sequential_consumer_step_factory,
        ),
        (
            EXPERT_PACKED_MASTER_MUONH_FSDP_SLAB_CONSUMER_BENCH,
            expert_packed_master_muonh_fsdp_slab_consumer_step_factory,
        ),
        (
            EXPERT_PACKED_MASTER_MUONH_FSDP_BULK_CONSUMER_BENCH,
            expert_packed_master_muonh_fsdp_bulk_consumer_step_factory,
        ),
    ],
)
def test_packed_master_muonh_fsdp_consumer_lowers_with_packed_state_and_scalar_loss(bench_kind, step_factory):
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
        ns4d_group_axis="replica_dcn,data",
        hidden_dim=16,
        intermediate_dim=8,
        num_experts=8,
        dtype=str(jnp.dtype(jnp.bfloat16)),
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
    master_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        bench_kind,
    )
    grad_bank = synthetic_packed_grouped_expert_bank_specs(
        mesh,
        config,
        bench_kind,
    )
    momentum_bank = synthetic_packed_grouped_expert_master_bank_specs(
        mesh,
        config,
        bench_kind,
    )
    expert_inputs = expert_fsdp_layer_consumer_input_specs(mesh, config)
    update_step = jax.jit(step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        next_master, next_momentum, loss = jax.eval_shape(
            update_step, master_bank, grad_bank, momentum_bank, expert_inputs
        )
        lowered = update_step.trace(master_bank, grad_bank, momentum_bank, expert_inputs).lower(
            lowering_platforms=(jax.devices()[0].platform if jax.devices() else jax.default_backend(),)
        )

    assert_packed_grouped_expert_bank_sharding(
        next_master,
        mesh,
        config,
        bench_kind,
        "packed-master MuonH/FSDP next master",
    )
    assert_packed_grouped_expert_bank_sharding(
        next_momentum,
        mesh,
        config,
        bench_kind,
        "packed-master MuonH/FSDP next momentum",
    )
    assert next_master["packed"]["w_gate_up"].dtype == jnp.float32
    assert next_momentum["packed"]["w_down"].dtype == jnp.float32
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    hlo = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo.dot_general > 0
    assert hlo.all_reduce == 0
    assert hlo.reduce_scatter == 0
    assert hlo.all_to_all == 0


def test_expert_fsdp_packed_bank_direction_apply_returns_fsdp_params():
    config = BenchConfig(
        layers=4,
        ns4d_group_size=4,
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
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(1, 2, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    grads = synthetic_fsdp_expert_specs(mesh, config)
    update_step = jax.jit(expert_fsdp_packed_bank_direction_apply_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "packed-bank direction-apply params")
    assert_expert_fsdp_sharding(grads, "packed-bank direction-apply grads")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grads)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grads).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "packed-bank direction-apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general > 0
    assert hlo_summary.all_to_all > 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_PACKED_BANK_DIRECTION_APPLY_BENCH) > 0


def test_expert_fsdp_grouped_explicit_a2a_apply_boundary_returns_fsdp_params():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_a2a_apply_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit a2a apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_data_first_a2a_apply_boundary_returns_fsdp_params():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_A2A_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_data_first_a2a_apply_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit data-first a2a apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_A2A_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_data_first_a2a_restore_boundary_returns_fsdp_updates():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_A2A_RESTORE_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_data_first_a2a_restore_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit data-first a2a restore FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_A2A_RESTORE_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_data_first_ppermute_restore_boundary_returns_fsdp_updates():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_PPERMUTE_RESTORE_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_data_first_ppermute_restore_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit data-first ppermute restore FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_to_all > 0
    assert hlo_summary.collective_permute > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_PPERMUTE_RESTORE_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_data_ppermute_boundary_returns_fsdp_updates_without_a2a():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_DATA_PPERMUTE_RESTORE_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_data_ppermute_restore_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit data ppermute restore FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.collective_permute > 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_DATA_PPERMUTE_RESTORE_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_data_ppermute_apply_boundary_returns_fsdp_params_without_a2a():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_DATA_PPERMUTE_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_data_ppermute_apply_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit data ppermute apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.collective_permute > 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_DATA_PPERMUTE_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_slice_first_gather_boundary_returns_fsdp_updates_without_a2a():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_slice_first_gather_restore_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit slice-first gather restore FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_explicit_tuple_slice_first_gather_boundary_returns_fsdp_updates_without_a2a():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_TUPLE_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
    )
    update_step = jax.jit(
        expert_fsdp_grouped_explicit_tuple_slice_first_gather_restore_boundary_step_factory(mesh, config)
    )

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit tuple slice-first gather restore FSDP updates")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert (
        estimated_ns_dot_flops(
            config,
            EXPERT_FSDP_GROUPED_EXPLICIT_TUPLE_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
        )
        == 0
    )


def test_expert_fsdp_grouped_custom_partition_slice_first_gather_boundary_returns_fsdp_updates():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_CUSTOM_PARTITION_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
    )
    update_step = jax.jit(
        expert_fsdp_grouped_custom_partition_slice_first_gather_restore_boundary_step_factory(mesh, config)
    )

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)

    assert_expert_fsdp_sharding(result, "custom-partition slice-first gather restore FSDP updates")
    assert (
        estimated_ns_dot_flops(
            config,
            EXPERT_FSDP_GROUPED_CUSTOM_PARTITION_SLICE_FIRST_GATHER_RESTORE_BOUNDARY_BENCH,
        )
        == 0
    )


def test_expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary_returns_fsdp_params_without_a2a():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_SLICE_FIRST_GATHER_APPLY_BOUNDARY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary_step_factory(mesh, config))

    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "explicit slice-first gather apply FSDP params")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.dot_general == 0
    assert hlo_summary.all_gather > 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_EXPLICIT_SLICE_FIRST_GATHER_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_packed_a2a_apply_boundary_packs_group_restores():
    config = BenchConfig(
        layers=8,
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
    explicit_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_A2A_APPLY_BOUNDARY_BENCH,
    )
    packed_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_PACKED_A2A_APPLY_BOUNDARY_BENCH,
    )
    explicit_step = jax.jit(expert_fsdp_grouped_explicit_a2a_apply_boundary_step_factory(mesh, config))
    packed_step = jax.jit(expert_fsdp_grouped_packed_a2a_apply_boundary_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        packed_result = jax.eval_shape(packed_step, params, packed_grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        explicit_lowered = explicit_step.trace(params, explicit_grouped_updates).lower(lowering_platforms=(platform,))
        packed_lowered = packed_step.trace(params, packed_grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(packed_result, "packed a2a apply FSDP params")
    explicit_hlo_summary = summarize_hlo(str(explicit_lowered.compiler_ir(dialect="stablehlo")))
    packed_hlo_summary = summarize_hlo(str(packed_lowered.compiler_ir(dialect="stablehlo")))
    assert explicit_hlo_summary.all_to_all > 0
    assert 0 < packed_hlo_summary.all_to_all < explicit_hlo_summary.all_to_all
    assert packed_hlo_summary.all_reduce == 0
    assert packed_hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_PACKED_A2A_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_packed_slice_first_gather_apply_boundary_packs_group_restores():
    config = BenchConfig(
        layers=8,
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
        data_axis=1,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    mesh = AbstractMesh(
        axis_sizes=(2, 1, 2, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    params = synthetic_fsdp_expert_specs(mesh, config)
    explicit_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_SLICE_FIRST_GATHER_APPLY_BOUNDARY_BENCH,
    )
    packed_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_PACKED_SLICE_FIRST_GATHER_APPLY_BOUNDARY_BENCH,
    )
    explicit_step = jax.jit(expert_fsdp_grouped_explicit_slice_first_gather_apply_boundary_step_factory(mesh, config))
    packed_step = jax.jit(expert_fsdp_grouped_packed_slice_first_gather_apply_boundary_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        packed_result = jax.eval_shape(packed_step, params, packed_grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        explicit_lowered = explicit_step.trace(params, explicit_grouped_updates).lower(lowering_platforms=(platform,))
        packed_lowered = packed_step.trace(params, packed_grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(packed_result, "packed slice-first gather apply FSDP params")
    explicit_hlo_summary = summarize_hlo(str(explicit_lowered.compiler_ir(dialect="stablehlo")))
    packed_hlo_summary = summarize_hlo(str(packed_lowered.compiler_ir(dialect="stablehlo")))
    assert 0 < packed_hlo_summary.all_gather < explicit_hlo_summary.all_gather
    assert packed_hlo_summary.all_to_all == 0
    assert packed_hlo_summary.all_reduce == 0
    assert packed_hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_PACKED_SLICE_FIRST_GATHER_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_packed_data_first_ppermute_apply_boundary_packs_group_restores():
    config = BenchConfig(
        layers=8,
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
    explicit_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_DATA_FIRST_PPERMUTE_RESTORE_BOUNDARY_BENCH,
    )
    packed_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_PACKED_DATA_FIRST_PPERMUTE_APPLY_BOUNDARY_BENCH,
    )
    explicit_step = jax.jit(expert_fsdp_grouped_explicit_data_first_ppermute_restore_boundary_step_factory(mesh, config))
    packed_step = jax.jit(expert_fsdp_grouped_packed_data_first_ppermute_apply_boundary_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        packed_result = jax.eval_shape(packed_step, params, packed_grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        explicit_lowered = explicit_step.trace(params, explicit_grouped_updates).lower(lowering_platforms=(platform,))
        packed_lowered = packed_step.trace(params, packed_grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(packed_result, "packed data-first ppermute apply FSDP params")
    explicit_hlo_summary = summarize_hlo(str(explicit_lowered.compiler_ir(dialect="stablehlo")))
    packed_hlo_summary = summarize_hlo(str(packed_lowered.compiler_ir(dialect="stablehlo")))
    assert explicit_hlo_summary.all_to_all > 0
    assert 0 < packed_hlo_summary.all_to_all < explicit_hlo_summary.all_to_all
    assert packed_hlo_summary.collective_permute > 0
    assert packed_hlo_summary.all_gather == 0
    assert packed_hlo_summary.all_reduce == 0
    assert packed_hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_PACKED_DATA_FIRST_PPERMUTE_APPLY_BOUNDARY_BENCH) == 0


def test_expert_fsdp_grouped_packed_data_ppermute_apply_boundary_packs_group_restores():
    config = BenchConfig(
        layers=8,
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
    explicit_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_EXPLICIT_DATA_PPERMUTE_RESTORE_BOUNDARY_BENCH,
    )
    packed_grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_PACKED_DATA_PPERMUTE_APPLY_BOUNDARY_BENCH,
    )
    explicit_step = jax.jit(expert_fsdp_grouped_explicit_data_ppermute_restore_boundary_step_factory(mesh, config))
    packed_step = jax.jit(expert_fsdp_grouped_packed_data_ppermute_apply_boundary_step_factory(mesh, config))

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        packed_result = jax.eval_shape(packed_step, params, packed_grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        explicit_lowered = explicit_step.trace(params, explicit_grouped_updates).lower(lowering_platforms=(platform,))
        packed_lowered = packed_step.trace(params, packed_grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(packed_result, "packed data-ppermute apply FSDP params")
    explicit_hlo_summary = summarize_hlo(str(explicit_lowered.compiler_ir(dialect="stablehlo")))
    packed_hlo_summary = summarize_hlo(str(packed_lowered.compiler_ir(dialect="stablehlo")))
    assert explicit_hlo_summary.all_to_all == 0
    assert packed_hlo_summary.all_to_all == 0
    assert 0 < packed_hlo_summary.all_gather < explicit_hlo_summary.all_gather
    assert 0 < packed_hlo_summary.collective_permute < explicit_hlo_summary.collective_permute
    assert packed_hlo_summary.all_reduce == 0
    assert packed_hlo_summary.reduce_scatter == 0
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_PACKED_DATA_PPERMUTE_APPLY_BOUNDARY_BENCH) == 0


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


def test_expert_fsdp_grouped_updates_muonh_direct_apply_restores_slices_before_apply():
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
    grouped_updates = synthetic_grouped_expert_specs(
        mesh,
        config,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_DIRECT_APPLY_BENCH,
    )
    update_step = jax.jit(expert_fsdp_grouped_updates_muonh_direct_apply_step_factory(mesh, config))

    assert ns4d_compute_sharding(mesh, config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_DIRECT_APPLY_BENCH).spec == P(
        ("replica_dcn", "data"), "expert", None, None
    )
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(result, "direct grouped-updates MuonH FSDP apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_DIRECT_APPLY_BENCH) == config.layers * 16
    assert estimated_ns_dot_flops(
        config,
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_DIRECT_APPLY_BENCH,
    ) == estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_UPDATES_MUONH_APPLY_BENCH)


def test_expert_fsdp_grouped_persistent_muonh_keeps_grouped_layout_without_boundary_collectives():
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
    grouped_params = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH)
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH)
    update_step = jax.jit(expert_fsdp_grouped_persistent_muonh_apply_step_factory(config))

    assert ns4d_compute_sharding(mesh, config, EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH).spec == (
        P(("replica_dcn", "data"), "expert", None, None)
    )
    assert_grouped_expert_sharding(
        grouped_params,
        mesh,
        config,
        EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH,
        "persistent grouped params",
    )
    assert_grouped_expert_sharding(
        grouped_updates,
        mesh,
        config,
        EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH,
        "persistent grouped updates",
    )
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result = jax.eval_shape(update_step, grouped_params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(grouped_params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_grouped_expert_sharding(
        result,
        mesh,
        config,
        EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH,
        "persistent grouped MuonH result",
    )
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    assert hlo_summary.all_gather == 0
    assert hlo_summary.all_to_all == 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH) == config.layers * 16
    assert estimated_ns_dot_flops(config, EXPERT_FSDP_GROUPED_PERSISTENT_MUONH_APPLY_BENCH) > 0


@pytest.mark.parametrize(
    ("bench_kind", "step_factory", "expected_collective"),
    [
        (
            EXPERT_FSDP_GROUPED_UPDATES_MUONH_EXPLICIT_APPLY_BENCH,
            expert_fsdp_grouped_updates_muonh_explicit_apply_step_factory,
            "all_gather",
        ),
        (
            EXPERT_FSDP_GROUPED_UPDATES_MUONH_EXPLICIT_A2A_APPLY_BENCH,
            expert_fsdp_grouped_updates_muonh_explicit_a2a_apply_step_factory,
            "all_to_all",
        ),
    ],
)
def test_expert_fsdp_grouped_updates_muonh_explicit_restore_then_apply_returns_fsdp_params(
    bench_kind,
    step_factory,
    expected_collective,
):
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
    grouped_updates = synthetic_grouped_expert_specs(mesh, config, bench_kind)
    update_step = jax.jit(step_factory(mesh, config))

    assert ns4d_compute_sharding(mesh, config, bench_kind).spec == P(("replica_dcn", "data"), "expert", None, None)
    assert_expert_fsdp_sharding(params, "expert FSDP params")
    assert_ns4d_sharding(grouped_updates, P(("replica_dcn", "data"), "expert", None, None), "grouped updates")
    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        result, updates = jax.eval_shape(update_step, params, grouped_updates)
        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = update_step.trace(params, grouped_updates).lower(lowering_platforms=(platform,))

    assert_expert_fsdp_sharding(updates, "explicit restored expert FSDP updates")
    assert_expert_fsdp_sharding(result, "explicit restored expert FSDP apply result")
    hlo_summary = summarize_hlo(str(lowered.compiler_ir(dialect="stablehlo")))
    assert hlo_summary.two_batch_axis_dot_general == 6
    if expected_collective == "all_gather":
        assert hlo_summary.all_gather > 0
    else:
        assert hlo_summary.all_to_all > 0
    assert hlo_summary.all_reduce == 0
    assert hlo_summary.reduce_scatter == 0
    assert estimated_matrix_count(config, bench_kind) == config.layers * 16
    assert estimated_ns_dot_flops(config, bench_kind) > 0


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


def test_ns_compute_dtype_can_cast_fp32_inputs_to_bf16_for_harness_ns():
    config = BenchConfig(
        layers=1,
        ns4d_group_size=1,
        ns4d_group_axis="none",
        hidden_dim=4,
        intermediate_dim=2,
        num_experts=2,
        dtype=str(jnp.dtype(jnp.float32)),
        backend_steps=1,
        orthogonalization_layout="stack_batch_4d_sharded",
        max_grouped_stack_size=8,
        replica_axis=1,
        data_axis=1,
        expert_axis=1,
        model_axis=1,
        learning_rate=0.02,
        ns_compute_dtype=str(jnp.dtype(jnp.bfloat16)),
    )
    x = jnp.ones((1, 2, 3, 4), dtype=jnp.float32)
    step = jax.jit(lambda update: zeropower_via_newtonschulz_4d_for_config(update, config))

    result = step(x)
    lowered = step.lower(x)
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))

    assert result.dtype == jnp.float32
    assert "tensor<1x2x3x4xbf16>" in hlo
    assert "stablehlo.dot_general" in hlo


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
            "boundary_collectives_required_absent": False,
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


def test_summary_row_reports_boundary_byte_estimates():
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
    estimates = estimated_boundary_byte_estimates(config, EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH)

    assert estimates is not None
    assert estimates["fsdp_output_to_grouped_input_ratio"] == 2.0
    assert estimates["all_gather_slice_peak_to_grouped_input_ratio"] == 4.0
    assert estimates["replica_fanout_factor"] == 2.0
    assert estimates["requires_replica_fanout"] is True
    assert (
        estimates["replica_fanout_min_extra_per_device_bytes"]
        == estimates["fsdp_output_per_device_bytes"] - estimates["grouped_input_per_device_bytes"]
    )
    assert estimates["replica_fanout_min_total_receive_bytes"] == estimates["global_update_bytes"]

    group_estimates = [asdict(estimate) for estimate in estimate_grouping(config)]
    result = {
        "metadata": {
            "label": "expert_fsdp_grouped_explicit_apply_boundary_h1",
            "bench_kind": EXPERT_FSDP_GROUPED_EXPLICIT_APPLY_BOUNDARY_BENCH,
            "config": asdict(config),
            "devices": 8,
            "ns4d_group_size": 4,
            "ns4d_padded_group_size": 4,
            "ns4d_input_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_compute_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_result_sharding_spec": None,
            "ns4d_boundary_status": "expert_fsdp_params_grouped_updates_explicit_apply",
            "boundary_collectives_allowed": True,
            "boundary_collectives_required_absent": False,
            "grouped_expert_group_count": None,
            "group_estimates": group_estimates,
        }
    }

    row = summary_row(result)

    assert row["boundary_primitive"] == "grouped_updates_apply_direct"
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_global_update_gib"] == bytes_to_gib(estimates["global_update_bytes"])
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_gib"] == bytes_to_gib(
        estimates["grouped_input_per_device_bytes"]
    )
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_gib"] == bytes_to_gib(
        estimates["fsdp_output_per_device_bytes"]
    )
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )
    assert row["estimated_boundary_all_gather_slice_peak_per_device_gib"] == bytes_to_gib(
        estimates["all_gather_slice_peak_per_device_bytes"]
    )
    assert row["estimated_boundary_peak_per_device_bytes"] == estimates["estimated_peak_per_device_bytes"]
    assert row["estimated_boundary_peak_per_device_gib"] == bytes_to_gib(estimates["estimated_peak_per_device_bytes"])
    assert row["estimated_boundary_fsdp_output_to_grouped_input_ratio"] == 2.0
    assert row["estimated_boundary_all_gather_slice_peak_to_grouped_input_ratio"] == 4.0
    assert row["estimated_boundary_replica_fanout_factor"] == 2.0
    assert row["estimated_boundary_requires_replica_fanout"] is True
    assert (
        row["estimated_boundary_replica_fanout_min_extra_per_device_bytes"]
        == estimates["replica_fanout_min_extra_per_device_bytes"]
    )
    assert row["estimated_boundary_replica_fanout_min_extra_per_device_gib"] == bytes_to_gib(
        estimates["replica_fanout_min_extra_per_device_bytes"]
    )
    assert (
        row["estimated_boundary_replica_fanout_min_total_receive_bytes"]
        == estimates["replica_fanout_min_total_receive_bytes"]
    )
    assert row["estimated_boundary_replica_fanout_min_total_receive_gib"] == bytes_to_gib(
        estimates["replica_fanout_min_total_receive_bytes"]
    )

    result["lowered"] = {
        "hlo": {
            "dot_general": 0,
            "batched_stack_dot_general": 0,
            "two_batch_axis_dot_general": 0,
            "custom_call": 0,
            "gpu_gemm_custom_call": 0,
            "all_gather": 4,
            "all_reduce": 0,
            "reduce_scatter": 0,
            "all_to_all": 2,
            "collective_permute": 0,
        },
        "lower_seconds": 1.0,
    }
    row = summary_row(result)

    assert (
        row["estimated_boundary_lowered_all_gather_slice_peak_per_all_gather_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"] / 4
    )
    assert (
        row["estimated_boundary_lowered_fsdp_output_per_all_to_all_bytes"]
        == estimates["fsdp_output_per_device_bytes"] / 2
    )
    assert row["estimated_boundary_lowered_global_update_per_collective_bytes"] == estimates["global_update_bytes"] / 6
    assert row["estimated_boundary_lowered_collective_count"] == 6.0
    assert row["estimated_boundary_lowered_ideal_collective_count"] == 2.0
    assert row["estimated_boundary_lowered_fragmentation_factor"] == 3.0

    real_grouped_estimates = estimated_boundary_byte_estimates(
        config,
        REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH,
    )
    assert real_grouped_estimates == estimates

    result["metadata"]["label"] = "real_expert_fsdp_grouped_muonh_optimizer_update_h1"
    result["metadata"]["bench_kind"] = REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH
    row = summary_row(result)

    assert (
        row["boundary_primitive"]
        == "fsdp_grads_to_grouped_chunks+grouped_muon_update+grouped_updates_to_fsdp_update_tree"
    )
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )

    grads_to_grouped_estimates = estimated_boundary_byte_estimates(
        config,
        EXPERT_FSDP_GRADS_TO_GROUPED_CHUNKS_BENCH,
    )
    assert grads_to_grouped_estimates == estimates

    result["metadata"]["label"] = "expert_fsdp_grads_to_grouped_chunks_h1"
    result["metadata"]["bench_kind"] = EXPERT_FSDP_GRADS_TO_GROUPED_CHUNKS_BENCH
    row = summary_row(result)

    assert row["boundary_primitive"] == "fsdp_grads_to_grouped_chunks"
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )

    packed_grads_to_grouped_estimates = estimated_boundary_byte_estimates(
        config,
        EXPERT_FSDP_GRADS_TO_PACKED_GROUPED_CHUNKS_BENCH,
    )
    assert packed_grads_to_grouped_estimates == estimates

    result["metadata"]["label"] = "expert_fsdp_grads_to_packed_grouped_chunks_h1"
    result["metadata"]["bench_kind"] = EXPERT_FSDP_GRADS_TO_PACKED_GROUPED_CHUNKS_BENCH
    row = summary_row(result)

    assert row["boundary_primitive"] == "fsdp_grads_to_packed_grouped_chunks"
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )

    explicit_packed_grads_to_grouped_estimates = estimated_boundary_byte_estimates(
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_CHUNKS_BENCH,
    )
    assert explicit_packed_grads_to_grouped_estimates == estimates

    result["metadata"]["label"] = "expert_fsdp_grads_to_explicit_packed_grouped_chunks_h1"
    result["metadata"]["bench_kind"] = EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_CHUNKS_BENCH
    row = summary_row(result)

    assert row["boundary_primitive"] == "fsdp_grads_to_explicit_packed_grouped_chunks"
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )

    explicit_packed_bank_estimates = estimated_boundary_byte_estimates(
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
    )
    assert explicit_packed_bank_estimates == estimates

    result["metadata"]["label"] = "expert_fsdp_grads_to_explicit_packed_grouped_bank_h1"
    result["metadata"]["bench_kind"] = EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH
    row = summary_row(result)

    assert row["boundary_primitive"] == "fsdp_grads_to_explicit_packed_grouped_bank"
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )

    packed_bank_apply_estimates = estimated_boundary_byte_estimates(
        config,
        EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH,
    )
    assert packed_bank_apply_estimates == estimates

    result["metadata"]["label"] = "expert_fsdp_packed_bank_a2a_apply_boundary_h1"
    result["metadata"]["bench_kind"] = EXPERT_FSDP_PACKED_BANK_A2A_APPLY_BOUNDARY_BENCH
    row = summary_row(result)

    assert row["boundary_primitive"] == "packed_grouped_updates_to_fsdp_apply"
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )

    packed_bank_direct_apply_estimates = estimated_boundary_byte_estimates(
        config,
        EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH,
    )
    assert packed_bank_direct_apply_estimates == estimates

    result["metadata"]["label"] = "expert_fsdp_packed_bank_direct_apply_boundary_h1"
    result["metadata"]["bench_kind"] = EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH
    row = summary_row(result)

    assert row["boundary_primitive"] == "packed_grouped_updates_to_fsdp_direct_apply"
    assert row["estimated_boundary_global_update_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_grouped_input_per_device_bytes"] == estimates["grouped_input_per_device_bytes"]
    assert row["estimated_boundary_fsdp_output_per_device_bytes"] == estimates["fsdp_output_per_device_bytes"]
    assert (
        row["estimated_boundary_all_gather_slice_peak_per_device_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"]
    )

    result["timing"] = {
        "timing": {
            "compile_seconds": 1.0,
            "compiled_hlo": {
                "dot_general": 0,
                "batched_stack_dot_general": 0,
                "two_batch_axis_dot_general": 0,
                "custom_call": 0,
                "gpu_gemm_custom_call": 0,
                "all_gather": 8,
                "all_reduce": 0,
                "reduce_scatter": 0,
                "all_to_all": 0,
                "collective_permute": 0,
            },
            "compiled_memory": {
                "argument_bytes": 100,
                "output_bytes": 40,
                "alias_bytes": 24,
                "temp_bytes": 16,
                "generated_code_bytes": 8,
                "host_argument_bytes": 4,
                "host_output_bytes": 2,
                "host_alias_bytes": 1,
                "host_temp_bytes": 3,
                "hbm_peak_bytes": 132,
            },
            "median_seconds": 1.0,
            "mean_seconds": 1.0,
            "min_seconds": 1.0,
            "correctness_max_error": 0.0,
        }
    }
    row = summary_row(result)

    assert (
        row["estimated_boundary_compiled_all_gather_slice_peak_per_all_gather_bytes"]
        == estimates["all_gather_slice_peak_per_device_bytes"] / 8
    )
    assert row["estimated_boundary_compiled_fsdp_output_per_all_to_all_bytes"] is None
    assert row["estimated_boundary_compiled_global_update_per_collective_bytes"] == estimates["global_update_bytes"] / 8
    assert row["estimated_boundary_compiled_collective_count"] == 8.0
    assert row["estimated_boundary_compiled_ideal_collective_count"] == 2.0
    assert row["estimated_boundary_compiled_fragmentation_factor"] == 4.0
    assert row["compiled_memory_argument_bytes"] == 100
    assert row["compiled_memory_argument_gib"] == bytes_to_gib(100)
    assert row["compiled_memory_output_bytes"] == 40
    assert row["compiled_memory_output_gib"] == bytes_to_gib(40)
    assert row["compiled_memory_alias_bytes"] == 24
    assert row["compiled_memory_alias_gib"] == bytes_to_gib(24)
    assert row["compiled_memory_temp_bytes"] == 16
    assert row["compiled_memory_temp_gib"] == bytes_to_gib(16)
    assert row["compiled_memory_generated_code_bytes"] == 8
    assert row["compiled_memory_generated_code_gib"] == bytes_to_gib(8)
    assert row["compiled_memory_host_argument_bytes"] == 4
    assert row["compiled_memory_host_argument_gib"] == bytes_to_gib(4)
    assert row["compiled_memory_host_output_bytes"] == 2
    assert row["compiled_memory_host_output_gib"] == bytes_to_gib(2)
    assert row["compiled_memory_host_alias_bytes"] == 1
    assert row["compiled_memory_host_alias_gib"] == bytes_to_gib(1)
    assert row["compiled_memory_host_temp_bytes"] == 3
    assert row["compiled_memory_host_temp_gib"] == bytes_to_gib(3)
    assert row["compiled_memory_hbm_peak_bytes"] == 132
    assert row["compiled_memory_hbm_peak_gib"] == bytes_to_gib(132)
    assert row["mean_estimated_boundary_global_gbps"] == estimates["global_update_bytes"] / 1e9
    assert row["median_estimated_boundary_global_gbps"] == estimates["global_update_bytes"] / 1e9
    assert row["boundary_correctness_max_error"] == 0.0
    assert (
        row["mean_estimated_boundary_grouped_input_per_device_gbps"] == estimates["grouped_input_per_device_bytes"] / 1e9
    )
    assert row["mean_estimated_boundary_fsdp_output_per_device_gbps"] == estimates["fsdp_output_per_device_bytes"] / 1e9


def test_summary_row_reports_packed_bank_boundary_phase_estimates():
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
        replica_axis=1,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    estimates = estimated_boundary_byte_estimates(config, EXPERT_FSDP_PACKED_BANK_MUONH_APPLY_BENCH)
    phases = estimated_boundary_phase_estimates(config, EXPERT_FSDP_PACKED_BANK_MUONH_APPLY_BENCH)

    assert estimates is not None
    assert [phase["name"] for phase in phases] == [
        "fsdp_grads_to_packed_grouped_bank",
        "fsdp_params_to_packed_grouped_bank",
        "packed_grouped_updates_to_fsdp_apply",
    ]
    assert [phase["expected_collective_type"] for phase in phases] == ["all_to_all", "all_to_all", "all_gather"]
    assert sum(phase["ideal_collective_count"] for phase in phases) == 6.0
    assert sum(phase["global_bytes"] for phase in phases) == 3 * estimates["global_update_bytes"]
    assert [phase["global_bytes_per_ideal_collective"] for phase in phases] == [
        estimates["global_update_bytes"] / 2,
        estimates["global_update_bytes"] / 2,
        estimates["global_update_bytes"] / 2,
    ]
    assert [phase["grouped_input_per_device_bytes_per_ideal_collective"] for phase in phases] == [
        estimates["grouped_input_per_device_bytes"] / 2,
        estimates["grouped_input_per_device_bytes"] / 2,
        estimates["grouped_input_per_device_bytes"] / 2,
    ]

    group_estimates = [asdict(estimate) for estimate in estimate_grouping(config)]
    result = {
        "metadata": {
            "label": "expert_fsdp_packed_bank_muonh_apply_h1",
            "bench_kind": EXPERT_FSDP_PACKED_BANK_MUONH_APPLY_BENCH,
            "config": asdict(config),
            "devices": 16,
            "ns4d_group_size": 4,
            "ns4d_padded_group_size": 4,
            "ns4d_input_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_compute_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_result_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_boundary_status": "packed_bank_muonh_apply",
            "boundary_collectives_allowed": True,
            "boundary_collectives_required_absent": False,
            "grouped_expert_group_count": None,
            "group_estimates": group_estimates,
            "local_devices": 8,
            "process_count": 2,
        },
        "lowered": {
            "hlo": {
                "dot_general": 18,
                "batched_stack_dot_general": 0,
                "two_batch_axis_dot_general": 18,
                "custom_call": 0,
                "gpu_gemm_custom_call": 0,
                "all_gather": 2,
                "all_reduce": 0,
                "reduce_scatter": 0,
                "all_to_all": 4,
                "collective_permute": 0,
            },
            "lower_seconds": 1.0,
        },
        "timing": {
            "timing": {
                "compile_seconds": 1.0,
                "compiled_hlo": {
                    "dot_general": 0,
                    "batched_stack_dot_general": 0,
                    "two_batch_axis_dot_general": 0,
                    "custom_call": 90,
                    "gpu_gemm_custom_call": 41,
                    "all_gather": 2,
                    "all_reduce": 0,
                    "reduce_scatter": 0,
                    "all_to_all": 4,
                    "collective_permute": 0,
                },
                "median_seconds": 2.0,
                "mean_seconds": 2.0,
                "min_seconds": 2.0,
                "correctness_max_error": None,
                "correctness_skipped_reason": "estimated global bytes 2 exceed correctness cap 1",
            }
        },
    }

    row = summary_row(result)

    assert row["devices"] == 16
    assert row["local_devices"] == 8
    assert row["process_count"] == 2
    assert row["replica_axis"] == 1
    assert row["data_axis"] == 2
    assert row["expert_axis"] == 2
    assert row["model_axis"] == 1
    assert row["estimated_boundary_phases"] == phases
    assert row["estimated_boundary_phase_count"] == 3
    assert row["estimated_boundary_phase_global_bytes"] == 3 * estimates["global_update_bytes"]
    assert row["estimated_boundary_phase_ideal_collective_count"] == 6.0
    assert row["estimated_boundary_phase_all_gather_global_bytes"] == estimates["global_update_bytes"]
    assert row["estimated_boundary_phase_all_gather_ideal_collective_count"] == 2.0
    assert (
        row["estimated_boundary_phase_all_gather_global_bytes_per_ideal_collective"]
        == estimates["global_update_bytes"] / 2
    )
    assert row["estimated_boundary_phase_all_gather_global_gib_per_ideal_collective"] == bytes_to_gib(
        estimates["global_update_bytes"] / 2
    )
    assert (
        row["estimated_boundary_phase_all_gather_grouped_input_per_device_bytes_per_ideal_collective"]
        == estimates["grouped_input_per_device_bytes"] / 2
    )
    assert (
        row["estimated_boundary_phase_all_gather_fsdp_output_per_device_bytes_per_ideal_collective"]
        == estimates["fsdp_output_per_device_bytes"] / 2
    )
    assert (
        row["estimated_boundary_phase_all_gather_max_global_bytes_per_phase_ideal_collective"]
        == estimates["global_update_bytes"] / 2
    )
    assert row["estimated_boundary_phase_all_to_all_global_bytes"] == 2 * estimates["global_update_bytes"]
    assert row["estimated_boundary_phase_all_to_all_global_gib"] == bytes_to_gib(2 * estimates["global_update_bytes"])
    assert row["estimated_boundary_phase_all_to_all_ideal_collective_count"] == 4.0
    assert (
        row["estimated_boundary_phase_all_to_all_global_bytes_per_ideal_collective"]
        == estimates["global_update_bytes"] / 2
    )
    assert row["estimated_boundary_phase_all_to_all_global_gib_per_ideal_collective"] == bytes_to_gib(
        estimates["global_update_bytes"] / 2
    )
    assert (
        row["estimated_boundary_phase_all_to_all_grouped_input_per_device_bytes_per_ideal_collective"]
        == estimates["grouped_input_per_device_bytes"] / 2
    )
    assert (
        row["estimated_boundary_phase_all_to_all_fsdp_output_per_device_bytes_per_ideal_collective"]
        == estimates["fsdp_output_per_device_bytes"] / 2
    )
    assert (
        row["estimated_boundary_phase_all_to_all_max_global_bytes_per_phase_ideal_collective"]
        == estimates["global_update_bytes"] / 2
    )
    assert row["estimated_boundary_phase_none_global_bytes"] is None
    assert row["estimated_boundary_phase_none_ideal_collective_count"] is None
    assert row["estimated_boundary_lowered_collective_to_phase_ideal_ratio"] == 1.0
    assert row["estimated_boundary_compiled_collective_to_phase_ideal_ratio"] == 1.0
    assert row["estimated_boundary_lowered_all_to_all_collective_count"] == 4.0
    assert row["estimated_boundary_lowered_all_to_all_ideal_collective_count"] == 4.0
    assert row["estimated_boundary_lowered_all_to_all_excess_collective_count"] == 0.0
    assert row["estimated_boundary_lowered_all_to_all_collective_to_ideal_ratio"] == 1.0
    assert row["estimated_boundary_lowered_all_to_all_matches_ideal_collective_count"] is True
    assert row["estimated_boundary_lowered_all_gather_collective_count"] == 2.0
    assert row["estimated_boundary_lowered_all_gather_ideal_collective_count"] == 2.0
    assert row["estimated_boundary_lowered_all_gather_excess_collective_count"] == 0.0
    assert row["estimated_boundary_lowered_all_gather_collective_to_ideal_ratio"] == 1.0
    assert row["estimated_boundary_lowered_all_gather_matches_ideal_collective_count"] is True
    assert row["estimated_boundary_lowered_total_excess_collective_count"] == 0.0
    assert row["estimated_boundary_lowered_matches_ideal_collective_counts"] is True
    assert row["estimated_boundary_compiled_all_to_all_collective_count"] == 4.0
    assert row["estimated_boundary_compiled_all_to_all_ideal_collective_count"] == 4.0
    assert row["estimated_boundary_compiled_all_to_all_excess_collective_count"] == 0.0
    assert row["estimated_boundary_compiled_all_to_all_collective_to_ideal_ratio"] == 1.0
    assert row["estimated_boundary_compiled_all_to_all_matches_ideal_collective_count"] is True
    assert row["estimated_boundary_compiled_total_excess_collective_count"] == 0.0
    assert row["estimated_boundary_compiled_matches_ideal_collective_counts"] is True
    assert row["boundary_correctness_skipped_reason"] == "estimated global bytes 2 exceed correctness cap 1"
    assert row["mean_estimated_boundary_phase_global_gbps"] == 3 * estimates["global_update_bytes"] / 2.0 / 1e9
    assert row["median_estimated_boundary_phase_global_gbps"] == 3 * estimates["global_update_bytes"] / 2.0 / 1e9
    assert row["mean_estimated_boundary_phase_all_to_all_global_gbps"] == (
        2 * estimates["global_update_bytes"] / 2.0 / 1e9
    )
    assert row["median_estimated_boundary_phase_all_to_all_global_gbps"] == (
        2 * estimates["global_update_bytes"] / 2.0 / 1e9
    )
    assert row["mean_estimated_boundary_phase_all_gather_global_gbps"] == (estimates["global_update_bytes"] / 2.0 / 1e9)
    assert row["median_estimated_boundary_phase_all_gather_global_gbps"] == (
        estimates["global_update_bytes"] / 2.0 / 1e9
    )

    direction_phases = estimated_boundary_phase_estimates(config, EXPERT_FSDP_PACKED_BANK_DIRECTION_APPLY_BENCH)
    assert [phase["name"] for phase in direction_phases] == [
        "fsdp_grads_to_packed_grouped_bank",
        "packed_grouped_updates_to_fsdp_apply",
    ]
    grads_to_bank_phases = estimated_boundary_phase_estimates(
        config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
    )
    assert [phase["name"] for phase in grads_to_bank_phases] == ["fsdp_grads_to_packed_grouped_bank"]
    assert [phase["expected_collective_type"] for phase in grads_to_bank_phases] == ["all_to_all"]
    assert sum(phase["ideal_collective_count"] for phase in grads_to_bank_phases) == 2.0
    direct_apply_phases = estimated_boundary_phase_estimates(
        config,
        EXPERT_FSDP_PACKED_BANK_DIRECT_APPLY_BOUNDARY_BENCH,
    )
    assert [phase["name"] for phase in direct_apply_phases] == ["packed_grouped_updates_to_fsdp_apply"]
    update_only_phases = estimated_boundary_phase_estimates(config, EXPERT_FSDP_PACKED_BANK_MUONH_UPDATE_ONLY_BENCH)
    assert [phase["name"] for phase in update_only_phases] == [
        "fsdp_grads_to_packed_grouped_bank",
        "fsdp_params_to_packed_grouped_bank",
    ]
    direct_muonh_phases = estimated_boundary_phase_estimates(
        config,
        EXPERT_FSDP_PACKED_BANK_MUONH_DIRECT_APPLY_BENCH,
    )
    assert [phase["name"] for phase in direct_muonh_phases] == [
        "fsdp_grads_to_packed_grouped_bank",
        "fsdp_params_to_packed_grouped_bank",
        "packed_grouped_updates_to_fsdp_apply",
    ]
    assert [phase["expected_collective_type"] for phase in direct_muonh_phases] == [
        "all_to_all",
        "all_to_all",
        "all_gather",
    ]
    n1_config = replace(config, data_axis=1, ns4d_group_axis="none", ns4d_group_size=1)
    n1_phases = estimated_boundary_phase_estimates(n1_config, EXPERT_FSDP_PACKED_BANK_MUONH_APPLY_BENCH)
    assert n1_phases
    assert {phase["expected_collective_type"] for phase in n1_phases} == {"none"}
    assert sum(phase["ideal_collective_count"] for phase in n1_phases) == 0.0
    result["metadata"]["config"] = asdict(n1_config)
    result["metadata"]["bench_kind"] = EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH
    result["metadata"]["label"] = "expert_fsdp_grads_to_explicit_packed_grouped_bank_h1"
    result["lowered"]["hlo"]["all_to_all"] = 0
    result["timing"]["timing"]["compiled_hlo"]["all_to_all"] = 0
    n1_row = summary_row(result)
    assert n1_row["estimated_boundary_phase_ideal_collective_count"] == 0.0
    assert n1_row["estimated_boundary_phase_all_to_all_global_bytes"] is None
    assert n1_row["estimated_boundary_phase_all_to_all_ideal_collective_count"] is None
    assert n1_row["estimated_boundary_phase_none_global_bytes"] == estimates["global_update_bytes"]
    assert n1_row["estimated_boundary_phase_none_global_gib"] == bytes_to_gib(estimates["global_update_bytes"])
    assert n1_row["estimated_boundary_phase_none_ideal_collective_count"] == 0.0
    assert n1_row["estimated_boundary_lowered_ideal_collective_count"] == 0.0
    assert n1_row["estimated_boundary_compiled_ideal_collective_count"] == 0.0
    r4_config = replace(config, replica_axis=4, data_axis=1, ns4d_group_axis="replica_dcn", ns4d_group_size=4)
    r4_direction_phases = estimated_boundary_phase_estimates(
        r4_config,
        EXPERT_FSDP_PACKED_BANK_DIRECTION_APPLY_BENCH,
    )
    assert [phase["expected_collective_type"] for phase in r4_direction_phases] == ["none", "all_gather"]
    assert [phase["ideal_collective_count"] for phase in r4_direction_phases] == [0.0, 2.0]
    r4_grads_to_bank_phases = estimated_boundary_phase_estimates(
        r4_config,
        EXPERT_FSDP_GRADS_TO_EXPLICIT_PACKED_GROUPED_BANK_BENCH,
    )
    assert [phase["expected_collective_type"] for phase in r4_grads_to_bank_phases] == ["none"]
    assert sum(phase["ideal_collective_count"] for phase in r4_grads_to_bank_phases) == 0.0


@pytest.mark.parametrize(
    ("packed_entry", "packed_bank_compute", "chunk_local_boundaries", "expected_mode"),
    [
        (False, False, False, "per_chunk_reshard"),
        (True, False, False, "packed_entry"),
        (True, True, False, "packed_bank_compute"),
        (False, False, True, "chunk_local"),
        (True, True, True, "chunk_local"),
    ],
)
def test_summary_row_reports_grouped_muonh_boundary_mode(
    packed_entry: bool,
    packed_bank_compute: bool,
    chunk_local_boundaries: bool,
    expected_mode: str,
):
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
        expert_grouped_muonh_packed_entry=packed_entry,
        expert_grouped_muonh_packed_bank_compute=packed_bank_compute,
        expert_grouped_muonh_chunk_local_boundaries=chunk_local_boundaries,
    )
    result = {
        "metadata": {
            "label": "real_expert_fsdp_grouped_muonh_optimizer_update_h3",
            "bench_kind": REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH,
            "config": asdict(config),
            "devices": 16,
            "ns4d_group_size": 4,
            "ns4d_padded_group_size": 4,
            "ns4d_input_sharding_spec": "P('expert', 'data', 'model')",
            "ns4d_compute_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_result_sharding_spec": "P('expert', 'data', 'model')",
            "ns4d_boundary_status": "real_expert_fsdp_grouped_muonh_optimizer_update",
            "boundary_collectives_allowed": True,
            "boundary_collectives_required_absent": False,
            "grouped_expert_group_count": None,
            "group_estimates": [asdict(estimate) for estimate in estimate_grouping(config)],
        },
    }

    row = summary_row(result)

    assert row["expert_grouped_muonh_boundary_mode"] == expected_mode
    assert row["expert_grouped_muonh_packed_entry"] is packed_entry
    assert row["expert_grouped_muonh_packed_bank_compute"] is packed_bank_compute
    assert row["expert_grouped_muonh_chunk_local_boundaries"] is chunk_local_boundaries


@pytest.mark.parametrize(
    (
        "packed_entry",
        "packed_bank_compute",
        "chunk_local_boundaries",
        "expected_all_gather_count",
        "expected_all_to_all_count",
        "expected_all_gather_bytes_multiplier",
        "expected_all_to_all_bytes_multiplier",
        "expected_phase_count",
    ),
    [
        (False, False, False, 6.0, 4.0, 3, 2, 5),
        (True, False, False, 2.0, 6.0, 1, 3, 4),
        (True, True, False, 2.0, 4.0, 1, 2, 3),
        (False, False, True, 2.0, 4.0, 1, 2, 3),
    ],
)
def test_real_grouped_muonh_summary_row_reports_boundary_phase_estimates(
    packed_entry: bool,
    packed_bank_compute: bool,
    chunk_local_boundaries: bool,
    expected_all_gather_count: float,
    expected_all_to_all_count: float,
    expected_all_gather_bytes_multiplier: int,
    expected_all_to_all_bytes_multiplier: int,
    expected_phase_count: int,
):
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
        expert_grouped_muonh_packed_entry=packed_entry,
        expert_grouped_muonh_packed_bank_compute=packed_bank_compute,
        expert_grouped_muonh_chunk_local_boundaries=chunk_local_boundaries,
    )
    estimates = estimated_boundary_byte_estimates(config, REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH)
    result = {
        "metadata": {
            "label": "real_expert_fsdp_grouped_muonh_optimizer_update_h3",
            "bench_kind": REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH,
            "config": asdict(config),
            "devices": 16,
            "ns4d_group_size": 4,
            "ns4d_padded_group_size": 4,
            "ns4d_input_sharding_spec": "P('expert', 'data', 'model')",
            "ns4d_compute_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_result_sharding_spec": "P('expert', 'data', 'model')",
            "ns4d_boundary_status": "real_expert_fsdp_grouped_muonh_optimizer_update",
            "boundary_collectives_allowed": True,
            "boundary_collectives_required_absent": False,
            "grouped_expert_group_count": None,
            "group_estimates": [asdict(estimate) for estimate in estimate_grouping(config)],
        },
        "timing": {
            "timing": {
                "compile_seconds": 1.0,
                "compiled_hlo": {
                    "dot_general": 0,
                    "batched_stack_dot_general": 0,
                    "two_batch_axis_dot_general": 0,
                    "custom_call": 0,
                    "gpu_gemm_custom_call": 0,
                    "all_gather": int(expected_all_gather_count),
                    "all_reduce": 0,
                    "reduce_scatter": 0,
                    "all_to_all": int(expected_all_to_all_count),
                    "collective_permute": 0,
                },
                "compiled_memory": {},
                "median_seconds": 2.0,
                "mean_seconds": 2.0,
                "min_seconds": 2.0,
                "correctness_max_error": 0.0,
                "correctness_skipped_reason": None,
            }
        },
    }

    row = summary_row(result)

    assert estimates is not None
    assert row["estimated_boundary_phase_count"] == expected_phase_count
    assert row["estimated_boundary_phase_all_gather_ideal_collective_count"] == expected_all_gather_count
    assert row["estimated_boundary_phase_all_to_all_ideal_collective_count"] == expected_all_to_all_count
    assert row["estimated_boundary_compiled_all_gather_collective_count"] == expected_all_gather_count
    assert row["estimated_boundary_compiled_all_gather_ideal_collective_count"] == expected_all_gather_count
    assert row["estimated_boundary_compiled_all_gather_excess_collective_count"] == 0.0
    assert row["estimated_boundary_compiled_all_gather_collective_to_ideal_ratio"] == 1.0
    assert row["estimated_boundary_compiled_all_gather_matches_ideal_collective_count"] is True
    assert row["estimated_boundary_compiled_all_to_all_collective_count"] == expected_all_to_all_count
    assert row["estimated_boundary_compiled_all_to_all_ideal_collective_count"] == expected_all_to_all_count
    assert row["estimated_boundary_compiled_all_to_all_excess_collective_count"] == 0.0
    assert row["estimated_boundary_compiled_all_to_all_collective_to_ideal_ratio"] == 1.0
    assert row["estimated_boundary_compiled_all_to_all_matches_ideal_collective_count"] is True
    assert row["estimated_boundary_compiled_total_excess_collective_count"] == 0.0
    assert row["estimated_boundary_compiled_matches_ideal_collective_counts"] is True
    assert (
        row["estimated_boundary_phase_all_gather_global_bytes"]
        == expected_all_gather_bytes_multiplier * estimates["global_update_bytes"]
    )
    assert (
        row["estimated_boundary_phase_all_to_all_global_bytes"]
        == expected_all_to_all_bytes_multiplier * estimates["global_update_bytes"]
    )
    assert row["estimated_boundary_compiled_collective_to_phase_ideal_ratio"] == 1.0
    assert (
        row["mean_estimated_boundary_phase_global_gbps"]
        == (expected_all_gather_bytes_multiplier + expected_all_to_all_bytes_multiplier)
        * estimates["global_update_bytes"]
        / 2.0
        / 1e9
    )
    assert (
        row["mean_estimated_boundary_phase_all_gather_global_gbps"]
        == expected_all_gather_bytes_multiplier * estimates["global_update_bytes"] / 2.0 / 1e9
    )
    assert (
        row["mean_estimated_boundary_phase_all_to_all_global_gbps"]
        == expected_all_to_all_bytes_multiplier * estimates["global_update_bytes"] / 2.0 / 1e9
    )


def test_real_grouped_muonh_packed_bank_compute_phase_estimates_use_bank_count():
    config = BenchConfig(
        layers=6,
        ns4d_group_size=2,
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
        expert_grouped_muonh_packed_entry=True,
        expert_grouped_muonh_packed_bank_compute=True,
    )
    result = {
        "metadata": {
            "label": "real_expert_fsdp_grouped_muonh_optimizer_update_h3",
            "bench_kind": REAL_EXPERT_FSDP_GROUPED_MUONH_OPTIMIZER_UPDATE_BENCH,
            "config": asdict(config),
            "devices": 16,
            "ns4d_group_size": 2,
            "ns4d_padded_group_size": 2,
            "ns4d_input_sharding_spec": "P('expert', 'data', 'model')",
            "ns4d_compute_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_result_sharding_spec": "P('expert', 'data', 'model')",
            "ns4d_boundary_status": "real_expert_fsdp_grouped_muonh_optimizer_update",
            "boundary_collectives_allowed": True,
            "boundary_collectives_required_absent": False,
            "grouped_expert_group_count": None,
            "grouped_expert_packed_bank_count": 2,
            "group_estimates": [asdict(estimate) for estimate in estimate_grouping(config)],
        },
    }

    row = summary_row(result)

    assert row["grouped_expert_group_count"] == 3
    assert row["grouped_expert_packed_bank_count"] == 2
    assert row["estimated_boundary_phase_all_gather_ideal_collective_count"] == 2.0
    assert row["estimated_boundary_phase_all_to_all_ideal_collective_count"] == 4.0
    assert row["estimated_boundary_phase_ideal_collective_count"] == 6.0


def test_summary_row_flags_boundary_collective_type_excess():
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
        replica_axis=1,
        data_axis=2,
        expert_axis=2,
        model_axis=1,
        learning_rate=0.02,
    )
    result = {
        "metadata": {
            "label": "expert_fsdp_packed_bank_muonh_apply_h1",
            "bench_kind": EXPERT_FSDP_PACKED_BANK_MUONH_APPLY_BENCH,
            "config": asdict(config),
            "devices": 16,
            "ns4d_group_size": 4,
            "ns4d_padded_group_size": 4,
            "ns4d_input_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_compute_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_result_sharding_spec": "P(('replica_dcn', 'data'), 'expert', None, None)",
            "ns4d_boundary_status": "packed_bank_muonh_apply",
            "boundary_collectives_allowed": True,
            "boundary_collectives_required_absent": False,
            "grouped_expert_group_count": None,
            "group_estimates": [asdict(estimate) for estimate in estimate_grouping(config)],
        },
        "timing": {
            "timing": {
                "compile_seconds": 1.0,
                "compiled_hlo": {
                    "dot_general": 0,
                    "batched_stack_dot_general": 0,
                    "two_batch_axis_dot_general": 0,
                    "custom_call": 0,
                    "gpu_gemm_custom_call": 0,
                    "all_gather": 3,
                    "all_reduce": 0,
                    "reduce_scatter": 0,
                    "all_to_all": 6,
                    "collective_permute": 0,
                },
                "compiled_memory": {},
                "median_seconds": 2.0,
                "mean_seconds": 2.0,
                "min_seconds": 2.0,
                "correctness_max_error": None,
                "correctness_skipped_reason": "estimated global bytes 2 exceed correctness cap 1",
            }
        },
    }

    row = summary_row(result)

    assert row["estimated_boundary_phase_all_to_all_ideal_collective_count"] == 4.0
    assert row["estimated_boundary_phase_all_gather_ideal_collective_count"] == 2.0
    assert row["estimated_boundary_compiled_all_to_all_excess_collective_count"] == 2.0
    assert row["estimated_boundary_compiled_all_gather_excess_collective_count"] == 1.0
    assert row["estimated_boundary_compiled_total_excess_collective_count"] == 3.0
    assert row["estimated_boundary_compiled_matches_ideal_collective_counts"] is False


def test_strict_boundary_gate_includes_expert_fsdp_and_collective_permute():
    collective_summary = HloSummary(
        characters=0,
        dot_general=0,
        batched_stack_dot_general=0,
        two_batch_axis_dot_general=0,
        custom_call=0,
        gpu_gemm_custom_call=0,
        all_gather=0,
        all_reduce=0,
        reduce_scatter=0,
        all_to_all=0,
        collective_permute=1,
        grouped_scope_mentions=0,
        stack_sharded_scope_mentions=0,
        pad_scope_mentions=0,
        slice_scope_mentions=0,
    )

    assert grouped_apply_boundary_collectives(collective_summary) == {"collective_permute": 1}
    assert not should_check_grouped_apply_boundary_collectives(
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_DIRECT_APPLY_BENCH,
        require_no_boundary_collectives=False,
    )
    assert should_check_grouped_apply_boundary_collectives(
        EXPERT_FSDP_GROUPED_UPDATES_MUONH_DIRECT_APPLY_BENCH,
        require_no_boundary_collectives=True,
    )
    assert should_check_grouped_apply_boundary_collectives(
        EXPERT_GROUPED_BANK_CONSUMER_BENCH,
        require_no_boundary_collectives=False,
    )


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
