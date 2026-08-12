# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from click.testing import CliRunner
from fray.cluster import ResourceConfig
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import _compact_grug_mesh_shape
from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeConfig,
    MokLikeDebugCounters,
    MokLikeForwardXStorage,
    MokLikeMemoryPoolRankTelemetry,
    MokLikeMemoryPoolTrimTelemetry,
    MokLikeRuntimeHandle,
)
from levanter.kernels.mixture_of_kittens.schedule import schedule_capacity
from levanter.schedule import ScheduleStep
from marin.execution.lazy import StepContext

from experiments.grug import dispatch as grug_dispatch
from experiments.grug.moe_hero_ep import (
    grugmuon_hero,
    launch,
    launch_mok_like,
    model,
    mok_like_correctness,
    mok_like_stateful_parity,
    train,
)


def test_mok_like_debug_metrics_preserve_peer_wait_distribution() -> None:
    peer_values = tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    peer_events = tuple(
        tuple(tuple(rank * 100 + phase * 10 + peer for peer in range(4)) for phase in range(4)) for rank in range(4)
    )
    counters = MokLikeDebugCounters(
        peer_ready_waits=(0, 0, 0, 0),
        completion_waits=(0, 0, 0, 0),
        generation_mismatches=(0, 0, 0, 0),
        slot_reuse_failures=(0, 0, 0, 0),
        slot_acquisitions=((0, 0),) * 4,
        max_active_slots=(0, 0, 0, 0),
        peer_wait_events=peer_events,
        peer_wait_cycles=peer_values,
        peer_wait_max_cycles=peer_values,
        staging_copy_calls=((1, 4), (1, 4), (1, 4), (1, 4)),
        staging_copy_bytes=((100, 400), (100, 400), (100, 400), (100, 400)),
    )

    metrics = train._mok_like_debug_metrics(counters)

    assert metrics["mok_like/runtime/rank_2/backward_pre/peer_3/peer_wait_events"] == 223
    assert metrics["mok_like/runtime/staging_copy_calls_total"] == 20
    assert metrics["mok_like/runtime/staging_copy_bytes_total"] == 2000
    assert metrics["mok_like/runtime/rank_3/backward/staging_copy_calls"] == 4
    assert metrics["mok_like/runtime/rank_3/backward/staging_copy_bytes"] == 400


def test_mok_like_process_metrics_validate_every_process(monkeypatch: pytest.MonkeyPatch) -> None:
    peer_values = tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    counters = MokLikeDebugCounters(
        peer_ready_waits=(0, 0, 0, 0),
        completion_waits=(0, 0, 0, 0),
        generation_mismatches=(0, 0, 0, 0),
        slot_reuse_failures=(0, 0, 0, 0),
        slot_acquisitions=((10, 0),) * 4,
        max_active_slots=(1, 1, 1, 1),
        peer_wait_events=peer_values,
        peer_wait_cycles=peer_values,
        peer_wait_max_cycles=peer_values,
        staging_copy_calls=((0, 10),) * 4,
        staging_copy_bytes=((0, 10),) * 4,
    )
    gathered = np.stack(
        tuple(
            train._pack_mok_like_process_summary(summary)
            for summary in (
                (0, 100, 100, 0, 0, 4, 400, 400, 1200, 0, 1, 4, 0, 0, 4000, 3200, 800),
                (1, 100, 100, 0, 0, 8, 800, 400, 1200, 0, 1, 4, 0, 0, 8000, 6000, 2000),
            )
        )
    )
    monkeypatch.setattr(train.jax, "process_count", lambda: 2)
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.multihost_utils, "process_allgather", lambda value, tiled: gathered)

    metrics = train._mok_like_process_metrics(
        counters,
        (100, 100),
        train._MokLikeHostTrimAudit(),
        expected_handler_calls=100,
        expected_trim_count=4,
        forward_x_storage=MokLikeForwardXStorage.RUNTIME_STAGED,
        backward_peer_storage=MokLikeBackwardPeerStorage.RUNTIME_STAGED,
        num_tokens=1,
        hidden_dim=1,
        top_k=1,
        workspace_slots=1,
    )

    assert metrics["mok_like/runtime/process_count"] == 2
    assert metrics["mok_like/runtime/process_1/forward_calls"] == 100
    assert metrics["mok_like/runtime/processes_with_protocol_errors"] == 0
    assert metrics["mok_like/runtime/processes_with_forward_staging"] == 2
    assert metrics["mok_like/runtime/total_forward_staging_calls"] == 12
    assert metrics["mok_like/runtime/total_forward_staging_bytes"] == 1200
    assert metrics["mok_like/runtime/total_backward_staging_calls"] == 800
    assert metrics["mok_like/runtime/total_backward_staging_bytes"] == 2400
    assert metrics["mok_like/runtime/processes_using_slot1"] == 0
    assert metrics["mok_like/runtime/max_active_slots_across_processes"] == 1
    assert metrics["mok_like/runtime/expected_trim_count_across_processes"] == 8
    assert metrics["mok_like/runtime/actual_trim_count_across_processes"] == 8
    assert metrics["mok_like/runtime/processes_with_trim_anomalies"] == 0
    assert metrics["mok_like/runtime/total_trimmed_bytes_across_processes"] == 2800


def test_mok_like_process_metrics_uses_local_trim_and_runtime_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    large_byte_count = 1 << 34
    peer_values = tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    counters = MokLikeDebugCounters(
        peer_ready_waits=(0, 0, 0, 0),
        completion_waits=(0, 0, 0, 0),
        generation_mismatches=(0, 0, 0, 0),
        slot_reuse_failures=(0, 0, 0, 0),
        slot_acquisitions=((10, 0),) * 4,
        max_active_slots=(1, 1, 1, 1),
        peer_wait_events=peer_values,
        peer_wait_cycles=peer_values,
        peer_wait_max_cycles=peer_values,
        staging_copy_calls=((2, 0),) * 4,
        staging_copy_bytes=((2 * large_byte_count, 0),) * 4,
    )
    trim_audit = train._MokLikeHostTrimAudit()
    for reserved_before, reserved_after in (
        (4 * large_byte_count, 2 * large_byte_count),
        (2 * large_byte_count, large_byte_count),
    ):
        trim_audit.record(
            MokLikeMemoryPoolTrimTelemetry(
                ranks=(
                    MokLikeMemoryPoolRankTelemetry(
                        rank=0,
                        reserved_bytes_before=reserved_before,
                        used_bytes_before=600,
                        reserved_bytes_after=reserved_after,
                        used_bytes_after=600,
                        device_free_bytes_before=100,
                        device_total_bytes_before=1000,
                        device_free_bytes_after=200,
                        device_total_bytes_after=1000,
                        graph_reserved_bytes_after=0,
                        graph_used_bytes_after=0,
                    ),
                ),
                active_reservations=0,
                active_workspace_slots=0,
                wall_time_seconds=0.01,
            )
        )
    monkeypatch.setattr(train.jax, "process_count", lambda: 1)
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.multihost_utils, "process_allgather", lambda value, tiled: value)

    metrics = train._mok_like_process_metrics(
        counters,
        (100, 100),
        trim_audit,
        expected_handler_calls=100,
        expected_trim_count=2,
        forward_x_storage=MokLikeForwardXStorage.RUNTIME_STAGED,
        backward_peer_storage=MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL,
        num_tokens=1,
        hidden_dim=1,
        top_k=1,
        workspace_slots=1,
    )

    assert metrics["mok_like/runtime/process_0/forward_staging_calls"] == 8
    assert metrics["mok_like/runtime/process_0/forward_staging_bytes"] == 8 * large_byte_count
    assert metrics["mok_like/runtime/process_0/trim_count"] == 2
    assert metrics["mok_like/runtime/process_0/trim_reserved_bytes_before"] == 6 * large_byte_count
    assert metrics["mok_like/runtime/process_0/trim_reserved_bytes_after"] == 3 * large_byte_count
    assert metrics["mok_like/runtime/process_0/trimmed_bytes"] == 3 * large_byte_count


@pytest.mark.parametrize(
    ("storage", "backward_calls", "backward_bytes"),
    [
        (MokLikeBackwardPeerStorage.RUNTIME_STAGED, 28, 728),
        (MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL, 7, 280),
        (MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL, 0, 0),
    ],
)
def test_mok_like_process_metrics_validate_backward_staging_for_each_storage_mode(
    monkeypatch: pytest.MonkeyPatch,
    storage: MokLikeBackwardPeerStorage,
    backward_calls: int,
    backward_bytes: int,
) -> None:
    peer_values = tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    counters = MokLikeDebugCounters(
        peer_ready_waits=(0, 0, 0, 0),
        completion_waits=(0, 0, 0, 0),
        generation_mismatches=(0, 0, 0, 0),
        slot_reuse_failures=(0, 0, 0, 0),
        slot_acquisitions=((7, 0),) * 4,
        max_active_slots=(1, 1, 1, 1),
        peer_wait_events=peer_values,
        peer_wait_cycles=peer_values,
        peer_wait_max_cycles=peer_values,
        staging_copy_calls=((0, backward_calls), (0, 0), (0, 0), (0, 0)),
        staging_copy_bytes=((0, backward_bytes), (0, 0), (0, 0), (0, 0)),
    )
    monkeypatch.setattr(train.jax, "process_count", lambda: 1)
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.multihost_utils, "process_allgather", lambda value, tiled: value)

    metrics = train._mok_like_process_metrics(
        counters,
        (7, 7),
        train._MokLikeHostTrimAudit(),
        expected_handler_calls=7,
        expected_trim_count=0,
        forward_x_storage=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
        backward_peer_storage=storage,
        num_tokens=2,
        hidden_dim=3,
        top_k=5,
        workspace_slots=1,
    )

    assert metrics["mok_like/runtime/process_0/backward_staging_calls"] == backward_calls
    assert metrics["mok_like/runtime/process_0/backward_staging_bytes"] == backward_bytes
    assert metrics["mok_like/runtime/expected_backward_staging_calls_per_process"] == backward_calls
    assert metrics["mok_like/runtime/expected_backward_staging_bytes_per_process"] == backward_bytes


def test_mok_like_process_metrics_preserve_64_bit_backward_staging_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    backward_bytes = 1 << 36
    peer_values = tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    counters = MokLikeDebugCounters(
        peer_ready_waits=(0, 0, 0, 0),
        completion_waits=(0, 0, 0, 0),
        generation_mismatches=(0, 0, 0, 0),
        slot_reuse_failures=(0, 0, 0, 0),
        slot_acquisitions=((1, 0),) * 4,
        max_active_slots=(1, 1, 1, 1),
        peer_wait_events=peer_values,
        peer_wait_cycles=peer_values,
        peer_wait_max_cycles=peer_values,
        staging_copy_calls=((0, 1), (0, 0), (0, 0), (0, 0)),
        staging_copy_bytes=((0, backward_bytes), (0, 0), (0, 0), (0, 0)),
    )
    monkeypatch.setattr(train.jax, "process_count", lambda: 1)
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.multihost_utils, "process_allgather", lambda value, tiled: value)

    metrics = train._mok_like_process_metrics(
        counters,
        (1, 1),
        train._MokLikeHostTrimAudit(),
        expected_handler_calls=1,
        expected_trim_count=0,
        forward_x_storage=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
        backward_peer_storage=MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL,
        num_tokens=1 << 34,
        hidden_dim=1,
        top_k=1,
        workspace_slots=1,
    )

    assert metrics["mok_like/runtime/total_backward_staging_bytes"] == backward_bytes


def test_mok_like_process_metrics_reject_replay_or_protocol_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    peer_values = tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    counters = MokLikeDebugCounters(
        peer_ready_waits=(0, 0, 0, 0),
        completion_waits=(0, 0, 0, 0),
        generation_mismatches=(0, 0, 0, 0),
        slot_reuse_failures=(0, 0, 0, 0),
        slot_acquisitions=((10, 0),) * 4,
        max_active_slots=(1, 1, 1, 1),
        peer_wait_events=peer_values,
        peer_wait_cycles=peer_values,
        peer_wait_max_cycles=peer_values,
        staging_copy_calls=((0, 10),) * 4,
        staging_copy_bytes=((0, 10),) * 4,
    )
    gathered = np.stack(
        tuple(
            train._pack_mok_like_process_summary(summary)
            for summary in (
                (0, 101, 100, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
                (1, 100, 100, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            )
        )
    )
    monkeypatch.setattr(train.jax, "process_count", lambda: 2)
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.multihost_utils, "process_allgather", lambda value, tiled: gathered)

    with pytest.raises(RuntimeError, match="distributed runtime contract failed"):
        train._mok_like_process_metrics(
            counters,
            (101, 100),
            train._MokLikeHostTrimAudit(),
            expected_handler_calls=100,
            expected_trim_count=0,
            forward_x_storage=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
            backward_peer_storage=MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL,
            num_tokens=1,
            hidden_dim=1,
            top_k=1,
            workspace_slots=1,
        )


@pytest.mark.parametrize(
    ("field", "value"), [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 2), (11, 1), (12, 1), (13, 1)]
)
def test_mok_like_process_metrics_reject_zero_copy_slot_or_trim_contract_violation(
    monkeypatch: pytest.MonkeyPatch,
    field: int,
    value: int,
) -> None:
    peer_values = tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    counters = MokLikeDebugCounters(
        peer_ready_waits=(0, 0, 0, 0),
        completion_waits=(0, 0, 0, 0),
        generation_mismatches=(0, 0, 0, 0),
        slot_reuse_failures=(0, 0, 0, 0),
        slot_acquisitions=((10, 0),) * 4,
        max_active_slots=(1, 1, 1, 1),
        peer_wait_events=peer_values,
        peer_wait_cycles=peer_values,
        peer_wait_max_cycles=peer_values,
        staging_copy_calls=((0, 10),) * 4,
        staging_copy_bytes=((0, 10),) * 4,
    )
    summary = [0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    summary[field] = value
    gathered = train._pack_mok_like_process_summary(tuple(summary))
    monkeypatch.setattr(train.jax, "process_count", lambda: 1)
    monkeypatch.setattr(train.jax, "process_index", lambda: 0)
    monkeypatch.setattr(train.multihost_utils, "process_allgather", lambda local, tiled: gathered)

    with pytest.raises(RuntimeError, match="distributed runtime contract failed"):
        train._mok_like_process_metrics(
            counters,
            (100, 100),
            train._MokLikeHostTrimAudit(),
            expected_handler_calls=100,
            expected_trim_count=0,
            forward_x_storage=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
            backward_peer_storage=MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL,
            num_tokens=1,
            hidden_dim=1,
            top_k=1,
            workspace_slots=1,
        )


def test_binding_runtime_does_not_change_canonical_parameter_leaves() -> None:
    config = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=256,
        intermediate_dim=256,
        shared_expert_intermediate_dim=256,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        local_kv_heads=4,
        global_kv_heads=4,
        head_dim=64,
        mok_like=MokLikeConfig(),
        remat_mode="save_moe",
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        transformer = model.Transformer.init(config, key=jax.random.key(0))
    before_with_paths, _ = jax.tree_util.tree_flatten_with_path(transformer)

    bound = transformer.bind_mok_like_runtime(object.__new__(MokLikeRuntimeHandle))

    after_with_paths, _ = jax.tree_util.tree_flatten_with_path(bound)
    assert [path for path, _ in after_with_paths] == [path for path, _ in before_with_paths]
    assert all(before is after for (_, before), (_, after) in zip(before_with_paths, after_with_paths, strict=True))
    assert bound.mok_like_runtime is not None


def test_initial_state_binds_runtime_before_optimizer_state_construction() -> None:
    config = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=256,
        intermediate_dim=256,
        shared_expert_intermediate_dim=256,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        local_kv_heads=4,
        global_kv_heads=4,
        head_dim=64,
        mok_like=MokLikeConfig(),
        remat_mode="save_moe",
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    runtime = object.__new__(MokLikeRuntimeHandle)
    optimizer = optax.trace(decay=0.9, nesterov=False)

    with jax.set_mesh(mesh):
        state = train.initial_state(
            config,
            optimizer=optimizer,
            mp=jmp.get_policy("params=float32,compute=float32,output=float32"),
            key=jax.random.key(0),
            ema_beta=None,
            mok_like_runtime=runtime,
        )

    assert state.params.mok_like_runtime is runtime
    assert jax.tree.structure(state.params) == jax.tree.structure(state.opt_state.trace)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shared_expert_intermediate_dim": 512}, "matching routed and shared"),
        ({"remat_mode": "recompute_all"}, "requires remat_mode"),
    ],
)
def test_mok_like_model_config_rejects_unsupported_block_contract(
    overrides: dict[str, object],
    message: str,
) -> None:
    kwargs = {
        "vocab_size": 128,
        "hidden_dim": 256,
        "intermediate_dim": 256,
        "shared_expert_intermediate_dim": 256,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "mok_like": MokLikeConfig(),
        "remat_mode": "save_moe",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        model.GrugModelConfig(**kwargs)


def test_mok_like_launcher_keeps_capacity_limited_control_distinct_from_promoted_dropless_preset() -> None:
    baseline = launch_mok_like.build_mok_like_run(
        run_id="capacity-diagnostic",
        num_steps=1,
        num_layers=1,
        mok_like_preset=launch_mok_like.MokLikeExperimentPreset.CAPACITY_LIMITED_V12,
        version="dev",
    )
    strict = launch_mok_like.build_mok_like_run(
        run_id="capacity-diagnostic",
        num_steps=1,
        num_layers=1,
        version="dev",
    )

    strict_config = strict.build_config(StepContext.for_fingerprint(strict.runtime_args.keys(), strict.deps))
    baseline_config = baseline.build_config(StepContext.for_fingerprint(baseline.runtime_args.keys(), baseline.deps))
    strict_mok_like = strict_config.model.mok_like
    assert strict_mok_like is not None
    assert baseline.fingerprint() != strict.fingerprint()
    assert schedule_capacity(65_536, 4, 2, strict_mok_like) == 1_052_672
    assert "mok-like-schedule-capacity-4" in strict_config.trainer.trainer.tracker.tags
    assert "strict-dropless-four-rank-capacity" in strict_config.trainer.trainer.tracker.tags
    assert "mok-like-preset-promoted-dropless-v12" in strict_config.trainer.trainer.tracker.tags
    assert (
        baseline_config.mok_like_pinned_host_memory_limit_gb
        == strict_config.mok_like_pinned_host_memory_limit_gb
        == launch_mok_like.PROMOTED_MOK_LIKE_PINNED_HOST_MEMORY_LIMIT_GB
    )
    assert "mok-like-pinned-host-memory-176gb" in baseline_config.trainer.trainer.tracker.tags
    assert "mok-like-pinned-host-memory-176gb" in strict_config.trainer.trainer.tracker.tags


def test_mok_like_launcher_uses_one_production_workspace_slot_and_fingerprints_two_slot_stress() -> None:
    production = launch_mok_like.build_mok_like_run(
        run_id="workspace-slots",
        num_steps=1,
        num_layers=1,
        version="dev",
    )
    stress = launch_mok_like.build_mok_like_run(
        run_id="workspace-slots",
        num_steps=1,
        num_layers=1,
        mok_like_workspace_slots=2,
        version="dev",
    )

    production_config = production.build_config(
        StepContext.for_fingerprint(production.runtime_args.keys(), production.deps)
    )
    stress_config = stress.build_config(StepContext.for_fingerprint(stress.runtime_args.keys(), stress.deps))

    assert production_config.model.mok_like is not None
    assert stress_config.model.mok_like is not None
    assert production_config.model.mok_like.workspace_slots == 1
    assert stress_config.model.mok_like.workspace_slots == 2
    assert production.fingerprint() != stress.fingerprint()
    assert "mok-like-workspace-slots-1" in production_config.trainer.trainer.tracker.tags
    assert "mok-like-workspace-slots-2" in stress_config.trainer.trainer.tracker.tags


def test_non_mok_like_launcher_rejects_mok_like_schedule_capacity_override() -> None:
    with pytest.raises(ValueError, match="only supported by the mok_like backend"):
        launch_mok_like.build_backend_comparison_run(
            run_id="irrelevant-capacity",
            num_steps=1,
            backend=launch_mok_like.MoeBackend.EP,
            mok_like_schedule_capacity_factor=4.0,
            version="dev",
        )


def test_backend_comparison_exposes_device_memory_fraction_as_run_identity() -> None:
    baseline = launch_mok_like.build_mok_like_run(
        run_id="allocator-diagnostic",
        num_steps=1,
        num_layers=1,
        gpu_device_memory_fraction=0.85,
        version="dev",
    )
    treatment = launch_mok_like.build_mok_like_run(
        run_id="allocator-diagnostic",
        num_steps=1,
        num_layers=1,
        gpu_device_memory_fraction=0.87,
        version="dev",
    )

    config = treatment.build_config(StepContext.for_fingerprint(treatment.runtime_args.keys(), treatment.deps))

    assert baseline.fingerprint() != treatment.fingerprint()
    assert config.gpu_device_memory_fraction == 0.87
    assert "device-memory-0.87" in config.trainer.trainer.tracker.tags


def test_backend_comparison_child_plan_is_attempt_zero() -> None:
    run = launch_mok_like.build_mok_like_run(
        run_id="attempt-zero",
        num_steps=1,
        num_layers=1,
        version="dev",
    )

    config = run.build_config(StepContext.for_fingerprint(run.runtime_args.keys(), run.deps))

    assert (
        config.max_retries_failure,
        config.max_retries_preemption,
        config.max_task_failures,
    ) == (0, 0, 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retries_failure", -1),
        ("max_retries_preemption", -1),
        ("max_task_failures", -1),
        ("max_task_failures", False),
    ],
)
def test_grug_run_config_rejects_invalid_retry_limits(field: str, value: object) -> None:
    run = launch_mok_like.build_mok_like_run(
        run_id="invalid-retries",
        num_steps=1,
        num_layers=1,
        version="dev",
    )
    config = run.build_config(StepContext.for_fingerprint(run.runtime_args.keys(), run.deps))

    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        dataclasses.replace(config, **{field: value})


def test_backend_comparison_exposes_gpu_allocator_as_run_identity() -> None:
    baseline = launch_mok_like.build_mok_like_run(
        run_id="allocator-kind-diagnostic",
        num_steps=1,
        num_layers=1,
        gpu_allocator=train.GpuAllocator.CUDA_ASYNC,
        version="dev",
    )
    treatment = launch_mok_like.build_mok_like_run(
        run_id="allocator-kind-diagnostic",
        num_steps=1,
        num_layers=1,
        gpu_allocator=train.GpuAllocator.VMM,
        version="dev",
    )

    config = treatment.build_config(StepContext.for_fingerprint(treatment.runtime_args.keys(), treatment.deps))

    assert baseline.fingerprint() != treatment.fingerprint()
    assert config.gpu_allocator is train.GpuAllocator.VMM
    assert "allocator-vmm" in config.trainer.trainer.tracker.tags


def test_backend_comparison_exposes_separate_temp_buffer_pool_as_run_identity() -> None:
    baseline = launch_mok_like.build_mok_like_run(
        run_id="temp-buffer-pool",
        num_steps=1,
        num_layers=1,
        version="dev",
    )
    treatment = launch_mok_like.build_mok_like_run(
        run_id="temp-buffer-pool",
        num_steps=1,
        num_layers=1,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SEPARATE,
        version="dev",
    )

    config = treatment.build_config(StepContext.for_fingerprint(treatment.runtime_args.keys(), treatment.deps))

    assert baseline.fingerprint() != treatment.fingerprint()
    assert config.gpu_temp_buffer_pool is train.GpuTempBufferPool.SEPARATE
    assert config.gpu_default_pool_preallocation is train.GpuDefaultPoolPreallocation.ON_DEMAND
    assert "temp-buffer-pool-separate" in config.trainer.trainer.tracker.tags
    assert "default-pool-preallocation-on-demand" in config.trainer.trainer.tracker.tags


def test_separate_temp_buffer_pool_requires_cuda_async_allocator() -> None:
    with pytest.raises(ValueError, match="requires the cuda_async allocator"):
        launch_mok_like.build_mok_like_run(
            run_id="invalid-temp-buffer-pool",
            num_steps=1,
            gpu_allocator=train.GpuAllocator.VMM,
            gpu_temp_buffer_pool=train.GpuTempBufferPool.SEPARATE,
            version="dev",
        )


def test_mok_like_launcher_exposes_on_demand_default_pool_as_run_identity() -> None:
    baseline = launch_mok_like.build_mok_like_run(
        run_id="default-pool-preallocation",
        num_steps=1,
        num_layers=1,
        version="dev",
    )
    treatment = launch_mok_like.build_mok_like_run(
        run_id="default-pool-preallocation",
        num_steps=1,
        num_layers=1,
        gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.ON_DEMAND,
        version="dev",
    )

    config = treatment.build_config(StepContext.for_fingerprint(treatment.runtime_args.keys(), treatment.deps))

    assert baseline.fingerprint() != treatment.fingerprint()
    assert config.gpu_default_pool_preallocation is train.GpuDefaultPoolPreallocation.ON_DEMAND
    assert "default-pool-preallocation-on-demand" in config.trainer.trainer.tracker.tags


def test_on_demand_default_pool_rejects_vmm_allocator() -> None:
    with pytest.raises(ValueError, match="requires the cuda_async allocator"):
        launch_mok_like.build_mok_like_run(
            run_id="invalid-default-pool-preallocation",
            num_steps=1,
            gpu_allocator=train.GpuAllocator.VMM,
            gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.ON_DEMAND,
            version="dev",
        )


def test_mok_like_launcher_exposes_local_only_autotune_cache_as_run_identity() -> None:
    baseline = launch_mok_like.build_mok_like_run(
        run_id="autotune-cache-mode",
        num_steps=1,
        num_layers=1,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        version="dev",
    )
    treatment = launch_mok_like.build_mok_like_run(
        run_id="autotune-cache-mode",
        num_steps=1,
        num_layers=1,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.LOCAL_ONLY,
        version="dev",
    )

    config = treatment.build_config(StepContext.for_fingerprint(treatment.runtime_args.keys(), treatment.deps))

    assert baseline.fingerprint() != treatment.fingerprint()
    assert config.xla_autotune_cache_mode is train.XlaAutotuneCacheMode.LOCAL_ONLY
    assert "xla-autotune-cache-local-only" in config.trainer.trainer.tracker.tags


def test_non_mok_like_launcher_rejects_local_only_autotune_cache() -> None:
    with pytest.raises(ValueError, match="only supported by the mok_like backend"):
        launch_mok_like.build_backend_comparison_run(
            run_id="invalid-local-only-autotune-cache",
            num_steps=1,
            backend=launch_mok_like.MoeBackend.EP,
            xla_autotune_cache_mode=train.XlaAutotuneCacheMode.LOCAL_ONLY,
            version="dev",
        )


def test_mok_like_launcher_exposes_default_pool_trim_interval_as_run_identity() -> None:
    baseline = launch_mok_like.build_mok_like_run(
        run_id="default-pool-trim",
        num_steps=55,
        num_layers=1,
        version="dev",
    )
    treatment = launch_mok_like.build_mok_like_run(
        run_id="default-pool-trim",
        num_steps=55,
        num_layers=1,
        gpu_default_pool_trim_interval_updates=25,
        version="dev",
    )

    config = treatment.build_config(StepContext.for_fingerprint(treatment.runtime_args.keys(), treatment.deps))

    assert baseline.fingerprint() != treatment.fingerprint()
    assert config.gpu_default_pool_trim_interval_updates == 25
    assert "default-pool-trim-interval-updates-25" in config.trainer.trainer.tracker.tags


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gpu_allocator": train.GpuAllocator.VMM}, "requires the cuda_async allocator"),
        ({"gpu_temp_buffer_pool": train.GpuTempBufferPool.SEPARATE}, "requires the shared temp-buffer pool"),
    ],
)
def test_default_pool_trim_rejects_non_default_cuda_async_pool_modes(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        launch_mok_like.build_mok_like_run(
            run_id="invalid-default-pool-trim",
            num_steps=55,
            gpu_default_pool_trim_interval_updates=25,
            version="dev",
            **kwargs,
        )


def test_default_pool_trim_rejects_non_mok_like_backend() -> None:
    with pytest.raises(ValueError, match="only supported by the mok_like backend"):
        launch_mok_like.build_backend_comparison_run(
            run_id="invalid-default-pool-trim-backend",
            num_steps=55,
            backend=launch_mok_like.MoeBackend.EP,
            gpu_default_pool_trim_interval_updates=25,
            version="dev",
        )


@pytest.mark.parametrize(
    ("interval", "message"),
    [
        (0, "must be positive"),
        (56, "must not exceed num_steps"),
    ],
)
def test_default_pool_trim_rejects_invalid_interval(interval: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        launch_mok_like.build_mok_like_run(
            run_id="invalid-default-pool-trim-boundary",
            num_steps=55,
            gpu_default_pool_trim_interval_updates=interval,
            version="dev",
        )


def test_mok_like_launcher_cli_builds_the_promoted_scale_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed: dict[str, object] = {}
    built = []
    original_build = launch_mok_like.build_backend_comparison_run

    def capture_build(**kwargs):
        parsed.update(kwargs)
        run = original_build(**kwargs)
        built.append(run)
        return run

    monkeypatch.setattr(launch_mok_like, "build_backend_comparison_run", capture_build)
    result = CliRunner().invoke(
        launch_mok_like.main,
        [
            "--run-id",
            "parsed-default-pool-trim",
            "--num-steps",
            "25",
            "--num-nodes",
            "32",
            "--mok-like-preset",
            "promoted_dropless_v12",
            "--version",
            "dev",
        ],
    )

    assert result.exit_code == 0, result.output
    assert parsed["mok_like_preset"] is launch_mok_like.MokLikeExperimentPreset.PROMOTED_DROPLESS_V12
    run = built[0]
    config = run.build_config(StepContext.for_fingerprint(run.runtime_args.keys(), run.deps))
    assert config.trainer.trainer.train_batch_size == 2048
    assert config.model.mok_like is not None
    assert config.model.mok_like.schedule_capacity_factor == 4.0
    assert config.model.mok_like.forward_x_storage is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL
    assert config.model.mok_like.backward_peer_storage is MokLikeBackwardPeerStorage.RUNTIME_STAGED
    assert config.gpu_device_memory_fraction == 0.80
    assert config.mok_like_pinned_host_memory_limit_gb == 176
    assert config.xla_autotune_cache_mode is train.XlaAutotuneCacheMode.LOCAL_ONLY
    assert "mok-like-pinned-host-memory-176gb" in config.trainer.trainer.tracker.tags
    assert (config.max_retries_failure, config.max_retries_preemption, config.max_task_failures) == (0, 0, 0)


@pytest.mark.parametrize("fraction", [0.0, 1.01])
def test_backend_comparison_rejects_invalid_device_memory_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="gpu_device_memory_fraction must be in"):
        launch_mok_like.build_mok_like_run(
            run_id="invalid-allocator-fraction",
            num_steps=1,
            gpu_device_memory_fraction=fraction,
            version="dev",
        )


def test_mok_like_launcher_labels_direct_backward_inputs_separately() -> None:
    run = launch_mok_like.build_mok_like_run(
        run_id="direct-backward-inputs",
        num_steps=1,
        num_layers=1,
        backward_peer_storage=MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL,
        version="dev",
    )

    config = run.build_config(StepContext.for_fingerprint(run.runtime_args.keys(), run.deps))

    assert config.model.mok_like is not None
    assert config.model.mok_like.backward_peer_storage is MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL
    assert "backward-inputs-zero-copy" in config.trainer.trainer.tracker.tags
    assert "backward-peer-zero-copy" not in config.trainer.trainer.tracker.tags


@pytest.mark.parametrize(
    ("num_nodes", "scale_tag", "wandb_group"),
    [
        (1, "four-gb200", "moe-backend-comparison-4gb200"),
        (2, "two-node", "moe-backend-comparison-2node"),
        (16, "one-rack", "moe-backend-comparison-1rack"),
        (32, "two-rack", "moe-backend-comparison-2rack"),
    ],
)
def test_mok_like_launcher_weak_scales_with_process_local_expert_groups(
    monkeypatch: pytest.MonkeyPatch,
    num_nodes: int,
    scale_tag: str,
    wandb_group: str,
) -> None:
    run = launch_mok_like.build_mok_like_run(
        run_id="weak-scale-contract",
        num_steps=25,
        num_nodes=num_nodes,
        version="dev",
    )
    config = run.build_config(StepContext.for_fingerprint(run.runtime_args.keys(), run.deps))
    resources = run.runtime_args["train_resources"]
    trainer = config.trainer.trainer
    submitted = []
    monkeypatch.setattr(
        grug_dispatch,
        "current_client",
        lambda: SimpleNamespace(
            submit=lambda request: (submitted.append(request) or SimpleNamespace(wait=lambda **_: None))
        ),
    )

    mesh_shape = _compact_grug_mesh_shape(
        process_count=resources.replicas,
        local_device_count=resources.device.chip_count(),
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        model_axis_size=1,
    )

    assert mesh_shape == (num_nodes, 1, 4, 1)
    assert trainer.train_batch_size // resources.chip_count() == 16
    assert resources.ram == launch_mok_like.RAM_PER_NODE
    assert config.model.num_layers == 48
    assert config.model.mok_like is not None
    assert config.model.mok_like.schedule_capacity_factor == 4.0
    assert config.model.mok_like.workspace_slots == 1
    assert config.model.mok_like.forward_x_storage is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL
    assert config.model.mok_like.backward_peer_storage is MokLikeBackwardPeerStorage.RUNTIME_STAGED
    assert config.gpu_allocator is train.GpuAllocator.CUDA_ASYNC
    assert config.gpu_temp_buffer_pool is train.GpuTempBufferPool.SHARED
    assert config.gpu_default_pool_preallocation is train.GpuDefaultPoolPreallocation.EAGER
    assert config.gpu_default_pool_trim_interval_updates is None
    assert config.gpu_device_memory_fraction == 0.80
    assert config.mok_like_pinned_host_memory_limit_gb == 176
    assert config.xla_autotune_cache_mode is train.XlaAutotuneCacheMode.LOCAL_ONLY
    assert config.processes_per_task == 1
    assert trainer.profiler.enabled
    assert trainer.profiler.start_step == 5
    assert trainer.profiler.num_steps == 5
    assert trainer.profiler.process_index == 0
    assert scale_tag in trainer.tracker.tags
    assert f"nodes-{num_nodes}" in trainer.tracker.tags
    assert trainer.tracker.group == wandb_group
    assert {
        "mok-like-preset-promoted-dropless-v12",
        "mok-like-schedule-capacity-4",
        "mok-like-workspace-slots-1",
        "forward-x-zero-copy",
        "forward-x-storage-xla-peer-experimental",
        "backward-peer-storage-runtime-staged",
        "allocator-cuda_async",
        "temp-buffer-pool-shared",
        "default-pool-preallocation-eager",
        "default-pool-trim-disabled",
        "xla-autotune-cache-local-only",
        "device-memory-0.8",
        "mok-like-pinned-host-memory-176gb",
    }.issubset(trainer.tracker.tags)

    control = launch_mok_like.build_mok_like_run(
        run_id="weak-scale-contract",
        num_steps=25,
        num_nodes=num_nodes,
        mok_like_preset=launch_mok_like.MokLikeExperimentPreset.CAPACITY_LIMITED_V12,
        version="dev",
    )
    assert run.fingerprint() != control.fingerprint()

    train.run_grug(dataclasses.replace(config, resources=resources))
    assert len(submitted) == 1
    request = submitted[0]
    assert (request.max_retries_failure, request.max_retries_preemption, request.max_task_failures) == (0, 0, 0)
    assert request.environment.env_vars[train.XLA_AUTOTUNE_CACHE_MODE_ENV] == "local_only"


@pytest.mark.parametrize(
    ("backend", "expected_mesh_shape"),
    [
        (launch_mok_like.MoeBackend.MOK_LIKE, (2, 1, 4, 1)),
        (launch_mok_like.MoeBackend.EP, (2, 1, 4, 1)),
        (launch_mok_like.MoeBackend.FSDP, (2, 4, 1, 1)),
    ],
)
def test_backend_comparison_keeps_the_same_weak_scaling_contract(
    backend: launch_mok_like.MoeBackend,
    expected_mesh_shape: tuple[int, ...],
) -> None:
    run = launch_mok_like.build_backend_comparison_run(
        run_id="matched-two-node",
        num_steps=1,
        num_nodes=2,
        backend=backend,
        num_layers=1,
        version="dev",
    )
    config = run.build_config(StepContext.for_fingerprint(run.runtime_args.keys(), run.deps))
    resources = run.runtime_args["train_resources"]

    mesh_shape = _compact_grug_mesh_shape(
        process_count=resources.replicas,
        local_device_count=resources.device.chip_count(),
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
        model_axis_size=1,
    )

    assert mesh_shape == expected_mesh_shape
    assert config.trainer.trainer.train_batch_size == 128
    assert resources.replicas == 2
    assert resources.cpu == launch_mok_like.CPUS_PER_NODE
    assert config.mok_like_pinned_host_memory_limit_gb == (
        176 if backend is launch_mok_like.MoeBackend.MOK_LIKE else None
    )


def test_backend_comparison_rejects_unreviewed_node_counts() -> None:
    with pytest.raises(ValueError, match="num_nodes must be one of"):
        launch_mok_like.build_mok_like_run(
            run_id="unsupported-scale",
            num_steps=1,
            num_nodes=3,
            version="dev",
        )


def test_mok_like_rejects_changing_batch_schedule_before_runtime_initialization() -> None:
    config = SimpleNamespace(
        model=model.GrugModelConfig(
            vocab_size=128,
            hidden_dim=256,
            intermediate_dim=256,
            shared_expert_intermediate_dim=256,
            num_experts=8,
            num_experts_per_token=2,
            mok_like=MokLikeConfig(),
            remat_mode="save_moe",
        ),
        mok_like_build=None,
        eval=None,
        trainer=SimpleNamespace(
            trainer=train.TrainerConfig(
                train_batch_size=[ScheduleStep(start=0, value=4), ScheduleStep(start=2, value=8)]
            )
        ),
    )

    with pytest.raises(ValueError, match="fixed batch size"):
        train._initialize_mok_like_for_config(config, object(), batch_size=4)


def test_mok_like_rejects_eval_with_a_different_static_token_shape() -> None:
    config = SimpleNamespace(
        model=model.GrugModelConfig(
            vocab_size=128,
            hidden_dim=256,
            intermediate_dim=256,
            shared_expert_intermediate_dim=256,
            num_experts=8,
            num_experts_per_token=2,
            mok_like=MokLikeConfig(),
            remat_mode="save_moe",
        ),
        mok_like_build=object(),
        eval=train.GrugEvalConfig(eval_batch_size=8),
        trainer=SimpleNamespace(trainer=train.TrainerConfig(train_batch_size=4)),
    )

    with pytest.raises(ValueError, match="evaluation must use the training batch size"):
        train._initialize_mok_like_for_config(config, object(), batch_size=4)


def test_full_bank_top_k_is_rejected_before_launch():
    # QB routing reads the (k+1)-th logit as its threshold, so a full-bank top-k asks `top_k` for
    # more entries than there are experts. Without this the job dies in the router, which is after
    # the 16-node gang is allocated.
    with pytest.raises(ValueError, match="must be < num_experts"):
        launch.build_hero_run(run_id="full-bank", num_steps=1, num_experts_per_token=128, version="dev")


@pytest.mark.parametrize("scenario", list(mok_like_correctness.RouteScenario))
def test_mok_like_correctness_routes_have_the_intended_local_expert_distribution(
    scenario: mok_like_correctness.RouteScenario,
) -> None:
    routes = mok_like_correctness._routes(512, scenario)
    local_experts = routes % mok_like_correctness.NUM_LOCAL_EXPERTS
    destination_ranks = routes // mok_like_correctness.NUM_LOCAL_EXPERTS
    counts = np.bincount(local_experts.reshape(-1), minlength=mok_like_correctness.NUM_LOCAL_EXPERTS)

    assert routes.shape == (4, 512, 4)
    if scenario is mok_like_correctness.RouteScenario.BALANCED:
        np.testing.assert_array_equal(counts, [4096, 4096])
    elif scenario is mok_like_correctness.RouteScenario.ZERO_TOKEN_EXPERT:
        np.testing.assert_array_equal(counts, [8192, 0])
    elif scenario is mok_like_correctness.RouteScenario.SKEWED:
        np.testing.assert_array_equal(counts, [6144, 2048])
    else:
        np.testing.assert_array_equal(counts, [4096, 4096])
        np.testing.assert_array_equal(destination_ranks, 0)


@pytest.mark.parametrize(("macrobatch_size", "expected"), [(2048, 1), (1024, 2), (256, 8)])
def test_mok_like_correctness_counts_nonpadding_macrobuffers(macrobatch_size: int, expected: int) -> None:
    routes = mok_like_correctness._routes(512, mok_like_correctness.RouteScenario.BALANCED)

    assert (
        mok_like_correctness._real_macrobuffers(
            routes,
            capacity=2816,
            macrobatch_size=macrobatch_size,
        )
        == expected
    )


def test_mok_like_correctness_cli_shuts_down_distributed_runtime_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls = 0

    def fail() -> None:
        raise RuntimeError("gate failed")

    def shutdown() -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1
        raise RuntimeError("shutdown failed")

    monkeypatch.setattr(mok_like_correctness, "main", fail)
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(jax.distributed, "shutdown", shutdown)

    with pytest.raises(RuntimeError, match="gate failed") as error:
        mok_like_correctness._run_cli()

    assert shutdown_calls == 1
    assert error.value.__notes__ == ["JAX distributed shutdown also failed: shutdown failed"]


def test_mok_like_stateful_parity_route_plan_alternates_imbalance_and_exact_capacity_boundary() -> None:
    plan = mok_like_stateful_parity._stateful_route_plan(8)
    config = MokLikeConfig(schedule_capacity_factor=3.75)
    capacity = schedule_capacity(
        512,
        mok_like_stateful_parity.TOP_K,
        mok_like_stateful_parity.NUM_LOCAL_EXPERTS,
        config,
    )
    all_to_one_routes = mok_like_stateful_parity._routes(
        512,
        mok_like_stateful_parity.RouteScenario.ALL_TO_ONE,
    )

    assert plan == (
        mok_like_stateful_parity.RouteScenario.BALANCED,
        mok_like_stateful_parity.RouteScenario.ZERO_TOKEN_EXPERT,
        mok_like_stateful_parity.RouteScenario.ALL_TO_ONE,
        mok_like_stateful_parity.RouteScenario.SKEWED,
        mok_like_stateful_parity.RouteScenario.ALL_TO_ONE,
        mok_like_stateful_parity.RouteScenario.ZERO_TOKEN_EXPERT,
        mok_like_stateful_parity.RouteScenario.ALL_TO_ONE,
        mok_like_stateful_parity.RouteScenario.SKEWED,
    )
    assert mok_like_stateful_parity._required_schedule_capacity(all_to_one_routes) == capacity
    assert mok_like_stateful_parity._route_metrics(all_to_one_routes, capacity=capacity) == {
        "assignment_counts": [4096, 4096, 0, 0, 0, 0, 0, 0],
        "zero_token_experts": [2, 3, 4, 5, 6, 7],
        "required_schedule_capacity": 8192,
        "schedule_capacity": 8192,
        "at_capacity_boundary": True,
    }


def test_mok_like_stateful_parity_inputs_are_fixed_per_replica_and_distinct_between_replicas() -> None:
    first = mok_like_stateful_parity._step_inputs(
        seed=17,
        replica_index=0,
        step=3,
        num_tokens=256,
        hidden_dim=256,
    )
    repeated = mok_like_stateful_parity._step_inputs(
        seed=17,
        replica_index=0,
        step=3,
        num_tokens=256,
        hidden_dim=256,
    )
    other_replica = mok_like_stateful_parity._step_inputs(
        seed=17,
        replica_index=1,
        step=3,
        num_tokens=256,
        hidden_dim=256,
    )

    for first_array, repeated_array in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(first_array, repeated_array)
    assert any(
        not np.array_equal(first_array, other_array)
        for first_array, other_array in zip(first, other_replica, strict=True)
    )
    np.testing.assert_allclose(first[1].sum(axis=-1), 2.5, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("replica_index", "replica_count", "environ", "expected"),
    [
        (1, 3, {}, (1, 3)),
        (None, None, {"IRIS_TASK_ID": "/dlwh/parity/1:0", "IRIS_NUM_TASKS": "2"}, (1, 2)),
        (None, None, {}, (0, 1)),
    ],
)
def test_mok_like_stateful_parity_resolves_replica_identity(
    replica_index: int | None,
    replica_count: int | None,
    environ: dict[str, str],
    expected: tuple[int, int],
) -> None:
    assert (
        mok_like_stateful_parity._replica_identity(
            replica_index,
            replica_count,
            environ=environ,
        )
        == expected
    )


def test_mok_like_stateful_parity_error_metrics_report_pointwise_and_relative_drift() -> None:
    metrics = mok_like_stateful_parity._reduced_error_metrics(
        jnp.asarray([1.0, 3.5, -2.0]),
        jnp.asarray([1.0, 3.0, -1.0]),
        absolute_tolerance=0.25,
    )

    assert not metrics["allclose"]
    assert metrics["max_absolute_error"] == 1.0
    assert metrics["mean_absolute_error"] == pytest.approx(0.5)
    assert metrics["mismatch_fraction"] == pytest.approx(2 / 3)
    assert metrics["relative_l2_error"] > 0


def test_mok_like_stateful_parity_rejects_optimizer_state_divergence() -> None:
    close = {"allclose": True}
    divergent = {"allclose": False, "max_absolute_error": 0.5}
    metrics = {
        "step": 7,
        "dropped_assignments": 0,
        "output": close,
        "loss": close,
        "gradients": {"routed_gate": close},
        "parameters": {"routed_gate": close},
        "optimizer_state": {"routed_gate": divergent},
    }

    with pytest.raises(AssertionError, match="diverged at step 7"):
        mok_like_stateful_parity._assert_step_parity(metrics)


def test_mok_like_stateful_parity_rejects_gradient_on_inactive_routed_expert() -> None:
    gradients = (
        jnp.zeros((8, 1, 1)),
        jnp.zeros((8, 1, 1)),
        jnp.zeros((8, 1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    bad_gradient = gradients[0].at[7, 0, 0].set(1.0)

    with pytest.raises(AssertionError, match="inactive routed expert received a gradient"):
        mok_like_stateful_parity._validated_inactive_expert_gradient_maxima(
            mok_like_stateful_parity._routes(
                1,
                mok_like_stateful_parity.RouteScenario.ALL_TO_ONE,
            ),
            (bad_gradient, *gradients[1:]),
        )


def test_mok_like_stateful_parity_derives_inactive_experts_from_the_full_route_batch() -> None:
    routes = mok_like_stateful_parity._routes(
        512,
        mok_like_stateful_parity.RouteScenario.SKEWED,
    )
    gradients = (
        jnp.ones((8, 1, 1)),
        jnp.ones((8, 1, 1)),
        jnp.ones((8, 1, 1)),
    )

    assert mok_like_stateful_parity._validated_inactive_expert_gradient_maxima(routes, gradients) == {}


def test_mok_like_stateful_parity_optimizer_carries_state_across_updates() -> None:
    parameters = (jnp.asarray([1.0], dtype=jnp.float32),)
    momentum_state = (jnp.asarray([0.0], dtype=jnp.float32),)
    gradients = (jnp.asarray([2.0], dtype=jnp.float32),)

    first_parameters, first_momentum = mok_like_stateful_parity._optimizer_update(
        parameters,
        momentum_state,
        gradients,
        learning_rate=0.1,
        momentum=0.5,
    )
    second_parameters, second_momentum = mok_like_stateful_parity._optimizer_update(
        first_parameters,
        first_momentum,
        gradients,
        learning_rate=0.1,
        momentum=0.5,
    )

    np.testing.assert_allclose(first_parameters[0], [0.8])
    np.testing.assert_allclose(first_momentum[0], [2.0])
    np.testing.assert_allclose(second_parameters[0], [0.5])
    np.testing.assert_allclose(second_momentum[0], [3.0])


def test_expert_bank_override_must_divide_the_expert_axis():
    # `moe_mlp` raises on an indivisible bank only once the 16-node gang is already allocated and
    # its workspace is built, so the launcher has to reject it while it is still free to do so.
    with pytest.raises(ValueError, match="must divide the expert axis"):
        launch.build_hero_run(run_id="bad-bank", num_steps=1, num_experts=200, version="dev")


@pytest.mark.parametrize(
    ("preallocation", "expected_env"),
    [
        (train.GpuDefaultPoolPreallocation.EAGER, "true"),
        (train.GpuDefaultPoolPreallocation.ON_DEMAND, "false"),
    ],
)
def test_run_grug_applies_ep_xla_defaults_and_default_pool_preallocation(
    monkeypatch: pytest.MonkeyPatch,
    preallocation: train.GpuDefaultPoolPreallocation,
    expected_env: str,
) -> None:
    explicit_overlap = "--xla_gpu_experimental_parallel_collective_overlap_limit=2"
    monkeypatch.setenv("XLA_FLAGS", explicit_overlap)
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=None),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=None,
        gpu_allocator=train.GpuAllocator.CUDA_ASYNC,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SHARED,
        gpu_default_pool_preallocation=preallocation,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=None,
        xla_flag_overrides=(),
        max_retries_failure=3,
        max_retries_preemption=100,
        max_task_failures=10,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    flags = os.environ["XLA_FLAGS"].split()
    assert explicit_overlap in flags
    assert "--xla_gpu_experimental_parallel_collective_overlap_limit=4" not in flags
    assert "--xla_gpu_enable_latency_hiding_scheduler=true" in flags
    assert train.XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG in flags
    for name, value in train.HERO_EP_RUNTIME_ENV.items():
        assert os.environ[name] == value
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == expected_env
    assert f"{train.XLA_SEPARATE_TEMP_BUFFER_FLAG}=false" in flags


def test_grug_dispatch_sends_attempt_zero_limits_on_child_request(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted = []
    waits: list[bool] = []

    def submit(request):
        submitted.append(request)
        return SimpleNamespace(wait=lambda *, raise_on_failure: waits.append(raise_on_failure))

    monkeypatch.setattr(grug_dispatch, "current_client", lambda: SimpleNamespace(submit=submit))
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=None),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="attempt-zero")),
        resources=ResourceConfig(cpu=1, ram="1g", disk="1g"),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=None,
        gpu_allocator=train.GpuAllocator.CUDA_ASYNC,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SHARED,
        gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.EAGER,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.LOCAL_ONLY,
        gpu_device_memory_fraction=None,
        xla_flag_overrides=(),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
    )

    train.run_grug(config)

    assert len(submitted) == 1
    request = submitted[0]
    assert (request.max_retries_failure, request.max_retries_preemption, request.max_task_failures) == (0, 0, 0)
    assert request.environment.env_vars[train.XLA_AUTOTUNE_CACHE_MODE_ENV] == "local_only"
    assert waits == [True]


def test_run_grug_applies_backend_xla_flag_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=None),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=None,
        gpu_allocator=train.GpuAllocator.CUDA_ASYNC,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SHARED,
        gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.EAGER,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=0.85,
        xla_flag_overrides=(
            "--xla_gpu_enable_latency_hiding_scheduler=false",
            "--xla_gpu_experimental_parallel_collective_overlap_limit=1",
        ),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    flags = os.environ["XLA_FLAGS"].split()
    assert "--xla_gpu_enable_latency_hiding_scheduler=false" in flags
    assert "--xla_gpu_experimental_parallel_collective_overlap_limit=1" in flags
    assert "--xla_gpu_enable_latency_hiding_scheduler=true" not in flags
    assert "--xla_gpu_experimental_parallel_collective_overlap_limit=4" not in flags


def test_run_grug_isolates_temp_buffer_pool_without_default_pool_preallocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=None),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=None,
        gpu_allocator=train.GpuAllocator.CUDA_ASYNC,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SEPARATE,
        gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.ON_DEMAND,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=0.85,
        xla_flag_overrides=(),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert f"{train.XLA_SEPARATE_TEMP_BUFFER_FLAG}=true" in os.environ["XLA_FLAGS"].split()
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


@pytest.mark.parametrize(
    ("allocator", "temp_pool", "preallocation", "message"),
    [
        (
            train.GpuAllocator.CUDA_ASYNC,
            train.GpuTempBufferPool.SEPARATE,
            train.GpuDefaultPoolPreallocation.EAGER,
            "separate GPU temp-buffer pool requires on-demand",
        ),
        (
            train.GpuAllocator.VMM,
            train.GpuTempBufferPool.SHARED,
            train.GpuDefaultPoolPreallocation.ON_DEMAND,
            "on-demand default-pool preallocation requires the cuda_async allocator",
        ),
    ],
)
def test_run_grug_rejects_inapplicable_default_pool_preallocation(
    allocator: train.GpuAllocator,
    temp_pool: train.GpuTempBufferPool,
    preallocation: train.GpuDefaultPoolPreallocation,
    message: str,
) -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=None),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=None,
        gpu_allocator=allocator,
        gpu_temp_buffer_pool=temp_pool,
        gpu_default_pool_preallocation=preallocation,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=None,
        xla_flag_overrides=(),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        with pytest.raises(ValueError, match=message):
            train.run_grug(config)

    dispatch.assert_not_called()


@pytest.mark.parametrize(
    ("allocator", "temp_pool", "message"),
    [
        (train.GpuAllocator.VMM, train.GpuTempBufferPool.SHARED, "requires the cuda_async allocator"),
        (
            train.GpuAllocator.CUDA_ASYNC,
            train.GpuTempBufferPool.SEPARATE,
            "requires the shared temp-buffer pool",
        ),
    ],
)
def test_run_grug_rejects_default_pool_trim_outside_shared_cuda_async_mode(
    allocator: train.GpuAllocator,
    temp_pool: train.GpuTempBufferPool,
    message: str,
) -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=MokLikeConfig(), remat_mode="save_moe"),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run", num_train_steps=55)),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=192,
        gpu_allocator=allocator,
        gpu_temp_buffer_pool=temp_pool,
        gpu_default_pool_preallocation=(
            train.GpuDefaultPoolPreallocation.ON_DEMAND
            if temp_pool is train.GpuTempBufferPool.SEPARATE
            else train.GpuDefaultPoolPreallocation.EAGER
        ),
        gpu_default_pool_trim_interval_updates=25,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=0.85,
        xla_flag_overrides=(),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        with pytest.raises(ValueError, match=message):
            train.run_grug(config)

    dispatch.assert_not_called()


def test_default_pool_trim_blocks_complete_update_and_logs_native_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step_result = (object(), {"train/loss": object()}, {"grads/norm": object()})
    telemetry = MokLikeMemoryPoolTrimTelemetry(
        ranks=tuple(
            MokLikeMemoryPoolRankTelemetry(
                rank=rank,
                reserved_bytes_before=1000 + rank,
                used_bytes_before=700 + rank,
                reserved_bytes_after=800 + rank,
                used_bytes_after=700 + rank,
                device_free_bytes_before=10_000 + rank,
                device_total_bytes_before=20_000 + rank,
                device_free_bytes_after=11_000 + rank,
                device_total_bytes_after=20_000 + rank,
                graph_reserved_bytes_after=500 + rank,
                graph_used_bytes_after=400 + rank,
            )
            for rank in range(4)
        ),
        active_reservations=0,
        active_workspace_slots=0,
        wall_time_seconds=0.125,
    )
    runtime = SimpleNamespace(trim_default_memory_pools=lambda: telemetry)
    blocked: list[object] = []
    logged: list[tuple[dict[str, int | float], int]] = []
    monkeypatch.setattr(train.jax, "block_until_ready", lambda value: blocked.append(value))
    monkeypatch.setattr(train.levanter.tracker, "log", lambda metrics, *, step: logged.append((metrics, step)))
    config = SimpleNamespace(gpu_default_pool_trim_interval_updates=25)

    result = train._maybe_trim_default_memory_pools(
        config,
        runtime,
        completed_update=25,
        train_step_result=step_result,
    )

    assert result is telemetry
    assert blocked == [step_result]
    assert logged == [
        (
            {
                "mok_like/runtime/default_pool_trim/completed_update": 25,
                "mok_like/runtime/default_pool_trim/trim_ordinal": 1,
                "mok_like/runtime/default_pool_trim/active_reservations": 0,
                "mok_like/runtime/default_pool_trim/active_workspace_slots": 0,
                "mok_like/runtime/default_pool_trim/wall_time_seconds": 0.125,
                **{
                    f"mok_like/runtime/default_pool_trim/rank_{rank}/{name}": value
                    for rank in range(4)
                    for name, value in (
                        ("reserved_bytes_before", 1000 + rank),
                        ("used_bytes_before", 700 + rank),
                        ("reserved_bytes_after", 800 + rank),
                        ("used_bytes_after", 700 + rank),
                        ("device_free_bytes_before", 10_000 + rank),
                        ("device_total_bytes_before", 20_000 + rank),
                        ("device_free_bytes_after", 11_000 + rank),
                        ("device_total_bytes_after", 20_000 + rank),
                        ("graph_reserved_bytes_after", 500 + rank),
                        ("graph_used_bytes_after", 400 + rank),
                        ("device_bytes_outside_default_pool_after", 8200 - rank),
                        ("device_bytes_outside_default_and_graph_pools_after", 7700 - 2 * rank),
                    )
                },
            },
            24,
        )
    ]


def test_default_pool_trim_interval_runs_exactly_at_updates_25_50_75_and_100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = MokLikeMemoryPoolTrimTelemetry(
        ranks=(),
        active_reservations=0,
        active_workspace_slots=0,
        wall_time_seconds=0.01,
    )
    trimmed_updates: list[int] = []
    current_update = 0

    def trim():
        trimmed_updates.append(current_update)
        return telemetry

    blocked_updates: list[int] = []
    logged: list[tuple[dict[str, int | float], int]] = []
    monkeypatch.setattr(train.jax, "block_until_ready", lambda _: blocked_updates.append(current_update))
    monkeypatch.setattr(train.levanter.tracker, "log", lambda metrics, *, step: logged.append((metrics, step)))
    config = SimpleNamespace(gpu_default_pool_trim_interval_updates=25)
    runtime = SimpleNamespace(trim_default_memory_pools=trim)

    for update in range(1, 101):
        current_update = update
        train._maybe_trim_default_memory_pools(
            config,
            runtime,
            completed_update=update,
            train_step_result=(object(), object(), object()),
        )

    assert trimmed_updates == [25, 50, 75, 100]
    assert blocked_updates == trimmed_updates
    assert [step for _, step in logged] == [24, 49, 74, 99]
    assert [metrics["mok_like/runtime/default_pool_trim/completed_update"] for metrics, _ in logged] == trimmed_updates
    assert [metrics["mok_like/runtime/default_pool_trim/trim_ordinal"] for metrics, _ in logged] == [1, 2, 3, 4]


def test_default_pool_trim_disabled_does_not_block_or_touch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train.jax,
        "block_until_ready",
        lambda _: pytest.fail("disabled trim must not add a full-result synchronization"),
    )

    result = train._maybe_trim_default_memory_pools(
        SimpleNamespace(gpu_default_pool_trim_interval_updates=None),
        None,
        completed_update=25,
        train_step_result=(object(), object(), object()),
    )

    assert result is None


def test_run_grug_requires_explicit_pinned_host_memory_for_mok_like_offload(monkeypatch) -> None:
    monkeypatch.delenv("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB", raising=False)
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=MokLikeConfig(), remat_mode="offload_moe"),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=None,
        gpu_allocator=train.GpuAllocator.CUDA_ASYNC,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SHARED,
        gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.EAGER,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=0.85,
        xla_flag_overrides=(),
        max_retries_failure=3,
        max_retries_preemption=100,
        max_task_failures=10,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        with pytest.raises(ValueError, match="explicit pinned-host memory limit"):
            train.run_grug(config)

    dispatch.assert_not_called()
    assert "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB" not in os.environ


def test_run_grug_applies_explicit_mok_like_pinned_host_memory_limit(monkeypatch) -> None:
    monkeypatch.setenv("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB", "64")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=MokLikeConfig(), remat_mode="offload_moe"),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=192,
        gpu_allocator=train.GpuAllocator.VMM,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SHARED,
        gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.EAGER,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=0.85,
        xla_flag_overrides=(),
        max_retries_failure=3,
        max_retries_preemption=100,
        max_task_failures=10,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"] == "192"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "vmm"
    assert os.environ["XLA_CLIENT_MEM_FRACTION"] == "0.85"
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ


def test_run_grug_requires_explicit_mok_like_device_memory_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=MokLikeConfig(), remat_mode="save_moe"),
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        mok_like_pinned_host_memory_limit_gb=192,
        gpu_allocator=train.GpuAllocator.CUDA_ASYNC,
        gpu_temp_buffer_pool=train.GpuTempBufferPool.SHARED,
        gpu_default_pool_preallocation=train.GpuDefaultPoolPreallocation.EAGER,
        gpu_default_pool_trim_interval_updates=None,
        xla_autotune_cache_mode=train.XlaAutotuneCacheMode.REMOTE_SYNC,
        gpu_device_memory_fraction=None,
        xla_flag_overrides=(),
        max_retries_failure=3,
        max_retries_preemption=100,
        max_task_failures=10,
    )

    with patch.object(train, "dispatch_grug_training_run") as dispatch:
        with pytest.raises(ValueError, match="explicit device memory fraction"):
            train.run_grug(config)

    dispatch.assert_not_called()


def test_ep_newton_schulz_returns_to_expert_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 64, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    input_sharding = NamedSharding(mesh, P(None, "expert", None, None))
    x = jax.ShapeDtypeStruct((48, 256, 8, 4), jnp.float32, sharding=input_sharding)

    def apply_ns(y):
        path = (jax.tree_util.GetAttrKey("w_gate"),)
        return grugmuon_hero._newtonschulz_4d_distributed(
            path,
            y,
            steps=0,
            eps=1e-8,
            coefficient_type="quintic",
            use_syrk=False,
        )

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(apply_ns, x)

    assert output.sharding == NamedSharding(mesh, P(None, "expert", "data", "model"))


def test_ep_newton_schulz_matches_replicated_path():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from experiments.grug.moe_hero_ep.grugmuon_hero import (
            _newtonschulz_4d_distributed,
            _zeropower_via_newtonschulz_replicated,
        )

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 2, 1),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        x = jax.random.normal(jax.random.key(0), (1, 2, 4, 2), dtype=jnp.float32)
        x_sharded = jax.device_put(x, NamedSharding(mesh, P(None, "expert", "data", "model")))
        path = (jax.tree_util.GetAttrKey("w_gate"),)
        expected = jax.vmap(
            jax.vmap(
                lambda matrix: _zeropower_via_newtonschulz_replicated(
                    matrix, steps=1, eps=1e-7, coefficient_type="quintic"
                )
            )
        )(x)

        apply_ns = jax.jit(
            lambda y: _newtonschulz_4d_distributed(
                path,
                y,
                steps=1,
                eps=1e-7,
                coefficient_type="quintic",
                use_syrk=False,
            )
        )
        with jax.set_mesh(mesh):
            actual = apply_ns(x_sharded)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ep_padded_newton_schulz_returns_to_parameter_sharding():
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 64, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    parameter_sharding = NamedSharding(mesh, P(None, "expert", None))
    x = jax.ShapeDtypeStruct((48, 64, 4), jnp.float32, sharding=parameter_sharding)

    def apply_ns(y):
        return grugmuon_hero._newtonschulz_padded_stack_sharded(
            y,
            steps=0,
            eps=1e-8,
            coefficient_type="quintic",
            target_sharding=parameter_sharding,
        )

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(apply_ns, x)

    assert output.sharding == parameter_sharding
