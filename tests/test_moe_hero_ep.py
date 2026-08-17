# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
import os
import subprocess
import sys
import textwrap
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from click.testing import CliRunner
from fray.cluster import ResourceConfig
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, set_mesh, use_abstract_mesh
from jax.sharding import PartitionSpec as P
from levanter.callbacks.watch import WatchConfig, compute_watch_stats
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
from experiments.grug.moe_hero_ep import small_scale_abl_launch as abl


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
        ({"shared_expert_intermediate_dim": 384}, "divisible by 256"),
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


def test_mok_like_model_config_accepts_a_routed_intermediate_wider_than_the_shared_one() -> None:
    """The hero widens the routed experts against the shared one; the fused backend now carries it."""

    config = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=256,
        intermediate_dim=512,
        shared_expert_intermediate_dim=256,
        num_experts=8,
        num_experts_per_token=2,
        mok_like=MokLikeConfig(),
        remat_mode="save_moe",
    )

    assert (config.intermediate_dim, config.shared_expert_intermediate_dim) == (512, 256)


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


def test_hero_run_without_shape_overrides_uses_the_selected_model():
    step = launch.build_hero_run(run_id="selected-default", dp_racks=1, num_steps=1, version="dev")
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert (
        config.model.hidden_dim,
        config.model.num_layers,
        config.model.num_experts,
        config.model.intermediate_dim,
        config.model.num_experts_per_token,
        config.model.latent_dim,
        config.model.capacity_factor,
        config.model.pooled_transport_capacity_factor,
        config.model.num_expert_waves,
        config.model.moe_implementation,
        config.trainer.trainer.train_batch_size,
        config.model.max_seq_len,
        config.processes_per_task,
        config.trainer.trainer.mp.param_dtype,
        config.trainer.trainer.mp.compute_dtype,
        config.trainer.master_param_mode,
    ) == (
        6144,
        48,
        384,
        3072,
        8,
        3072,
        1.15,
        1.10,
        3,
        "fixed_pooled_wave_all_to_all",
        1024,
        4096,
        1,
        jnp.bfloat16,
        jnp.bfloat16,
        train.MasterParamMode.FP32_PINNED_HOST,
    )


def test_full_bank_top_k_is_rejected_before_launch():
    # QB routing reads the (k+1)-th logit as its threshold, so a full-bank top-k asks `top_k` for
    # more entries than there are experts. Without this the job dies in the router, which is after
    # the 16-node gang is allocated.
    with pytest.raises(ValueError, match="must be < num_experts"):
        launch.build_hero_run(
            run_id="full-bank",
            dp_racks=1,
            num_steps=1,
            num_experts=128,
            num_experts_per_token=128,
            version="dev",
        )


def test_checkpoint_path_overrides_the_step_output_path():
    """A run that only exercises the checkpoint write sends it to disposable storage."""
    temp_path = "s3://marin-us-east-02a/tmp/ttl=1d/hero-ckpt-smoke"
    step = launch.build_hero_run(
        run_id="ckpt-elsewhere",
        dp_racks=1,
        num_steps=1,
        save_checkpoints=True,
        checkpoint_path=temp_path,
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.trainer.checkpointer.base_path == temp_path


def test_checkpoint_path_defaults_under_the_step_output_path():
    step = launch.build_hero_run(run_id="ckpt-default", dp_racks=1, num_steps=1, version="dev")
    ctx = StepContext.for_fingerprint(step.runtime_args, step.deps)
    config = step.build_config(ctx)

    assert config.trainer.trainer.checkpointer.base_path == f"{ctx.output_path}/checkpoints"


def test_checkpoint_interval_must_be_positive():
    with pytest.raises(ValueError, match="checkpoint_interval must be positive"):
        launch.build_hero_run(
            run_id="bad-checkpoint-interval",
            dp_racks=1,
            num_steps=1,
            checkpoint_interval=timedelta(0),
            version="dev",
        )


@pytest.mark.parametrize(
    ("profile_steps", "profile_start_step"),
    [(-1, 0), (1, -1), (1, 3)],
)
def test_profile_window_must_fall_inside_the_run(profile_steps, profile_start_step):
    with pytest.raises(ValueError, match="profile"):
        launch.build_hero_run(
            run_id="bad-profile-window",
            dp_racks=1,
            num_steps=3,
            profile_steps=profile_steps,
            profile_start_step=profile_start_step,
            version="dev",
        )


def test_data_parallel_racks_keep_the_global_batch_explicit():
    step = launch.build_hero_run(run_id="two-racks", dp_racks=2, num_steps=1, version="dev")
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.replica_axis_size == 2
    assert config.trainer.trainer.train_batch_size == launch.HERO_EP_BATCH_SIZE
    assert step.runtime_args["train_resources"].replicas == 2 * launch.HERO_EP_NODES


def test_schedule_steps_do_not_extend_the_run():
    step = launch.build_hero_run(
        run_id="schedule-head",
        dp_racks=1,
        num_steps=5,
        schedule_steps=100,
        version="dev",
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.trainer.num_train_steps == 100
    assert config.stop_after_steps == 5


@pytest.mark.parametrize("scenario", list(mok_like_correctness.RouteScenario))
def test_mok_like_correctness_routes_have_the_intended_local_expert_distribution(
    scenario: mok_like_correctness.RouteScenario,
) -> None:
    routes = mok_like_correctness._routes(512, scenario, mok_like_correctness.DEFAULT_TOP_K)
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
    routes = mok_like_correctness._routes(
        512, mok_like_correctness.RouteScenario.BALANCED, mok_like_correctness.DEFAULT_TOP_K
    )

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
        mok_like_stateful_parity.TOP_K,
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
                mok_like_stateful_parity.TOP_K,
            ),
            (bad_gradient, *gradients[1:]),
        )


def test_mok_like_stateful_parity_derives_inactive_experts_from_the_full_route_batch() -> None:
    routes = mok_like_stateful_parity._routes(
        512,
        mok_like_stateful_parity.RouteScenario.SKEWED,
        mok_like_stateful_parity.TOP_K,
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


def test_expert_bank_override_must_be_divisible_by_the_expert_axis():
    # `moe_mlp` raises on an indivisible bank only once the 16-node gang is already allocated and
    # its workspace is built, so the launcher has to reject it while it is still free to do so.
    with pytest.raises(ValueError, match="must be divisible by 64"):
        launch.build_hero_run(run_id="bad-bank", dp_racks=1, num_steps=1, num_experts=200, version="dev")


def test_expert_bank_override_must_support_three_waves():
    with pytest.raises(ValueError, match="local expert count=4 must be divisible by num_expert_waves=3"):
        launch.build_hero_run(run_id="bad-waves", dp_racks=1, num_steps=1, num_experts=256, version="dev")


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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=1)),
            watch_mode=train.WatchMode.INLINE,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="attempt-zero", watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=ResourceConfig(cpu=1, ram="1g", disk="1g"),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", num_train_steps=55, watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=0)),
            watch_mode=train.WatchMode.DIAGNOSTIC,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
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


def _run_grug_config(**overrides):
    """A config carrying every field `run_grug` reads, for the runtime-environment tests."""
    config = SimpleNamespace(
        model=SimpleNamespace(mok_like=None),
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=1)),
            watch_mode=train.WatchMode.INLINE,
        ),
        resources=object(),
        processes_per_task=1,
        pip_packages=(),
        extra_setup_scripts=(),
        mok_like_pinned_host_memory_limit_gb=None,
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
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_run_grug_applies_the_hero_runtime_environment(monkeypatch):
    """The base runtime env reaches os.environ on a per-node run."""
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = _run_grug_config()

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["JAX_ENABLE_PGLE"] == "false"
    assert os.environ["XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"] == "192"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"


def test_run_grug_defaults_pgle_off_for_per_gpu_processes(monkeypatch):
    # Per-GPU processes cannot profile: the per-process CUPTI sessions collide with
    # each other and with the cluster's DCGM, and auto-PGLE's recompile path has
    # wedged per-node gangs (#7344). Per-GPU runs therefore default PGLE off, while
    # an explicit env setting still wins.
    for name in train.HERO_EP_RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = _run_grug_config(processes_per_task=4)

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["JAX_ENABLE_PGLE"] == "false"

    monkeypatch.setenv("JAX_ENABLE_PGLE", "true")
    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)
    assert os.environ["JAX_ENABLE_PGLE"] == "true"


def test_run_grug_keeps_explicit_ep_runtime_values(monkeypatch):
    """Env overrides win for settings with no config field; typed config fields win over env.

    `JAX_ENABLE_PGLE` exists only in `HERO_EP_RUNTIME_ENV`, which is applied with `setdefault`, so
    an operator's explicit value survives. The allocator became `GrugRunConfig.gpu_allocator`, which
    is labelled into the run identity; letting an ambient env silently override it would make the
    run name disagree with what actually ran, so the config field is authoritative.
    """
    monkeypatch.setenv("JAX_ENABLE_PGLE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = _run_grug_config(processes_per_task=1, gpu_allocator=train.GpuAllocator.VMM)

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    assert os.environ["JAX_ENABLE_PGLE"] == "false"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == train.GpuAllocator.VMM.value


@pytest.mark.parametrize(
    ("watch_mode", "watch_interval", "expected_overlap_limit"),
    [
        (train.WatchMode.INLINE, 1, train.INLINE_WATCH_COLLECTIVE_OVERLAP_LIMIT),
        (train.WatchMode.DIAGNOSTIC, 1, train.DEFAULT_COLLECTIVE_OVERLAP_LIMIT),
        (train.WatchMode.INLINE, 0, train.DEFAULT_COLLECTIVE_OVERLAP_LIMIT),
    ],
)
def test_run_grug_reduces_collective_overlap_only_for_inline_watch(
    monkeypatch, watch_mode, watch_interval, expected_overlap_limit
):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    config = _run_grug_config(
        trainer=SimpleNamespace(
            trainer=SimpleNamespace(id="test-run", watch=WatchConfig(interval=watch_interval)),
            watch_mode=watch_mode,
        ),
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

    flags = os.environ["XLA_FLAGS"].split()
    assert f"{train.XLA_COLLECTIVE_OVERLAP_FLAG}={expected_overlap_limit}" in flags


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


def test_dropless_local_transform_swaps_moe_backend_and_shares_weights():
    # The dropless eval transform must retarget only the static MoE backend fields and keep every
    # weight leaf shared by identity, so the eval scores the trained weights with no capacity drops.
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        report_capacity_overflow=True,
    )
    with set_mesh(mesh):
        m = model.Transformer.init(cfg, key=jax.random.key(0))
    dropless = train._to_dropless_local(m)

    original = m.stacked_blocks.stacked.mlp.expert_mlp
    swapped = dropless.stacked_blocks.stacked.mlp.expert_mlp
    assert original.implementation == "fixed_all_to_all"  # input model left untouched
    assert swapped.implementation == "sonic_cute"
    assert swapped.expert_chunks == 1
    orig_leaves = jax.tree_util.tree_leaves(original)
    swapped_leaves = jax.tree_util.tree_leaves(swapped)
    assert len(orig_leaves) == len(swapped_leaves)
    assert all(a is b for a, b in zip(orig_leaves, swapped_leaves, strict=True))


def test_eval_every_adds_the_held_out_suites_as_dependencies():
    # Held-out sets are what make a run scoreable; a throughput-only run should not pay for them.
    off = launch.build_hero_run(run_id="eval-off", dp_racks=1, num_steps=1, version="dev")
    on = launch.build_hero_run(run_id="eval-on", dp_racks=1, num_steps=1, eval_every=50, version="dev")
    off_config = off.build_config(StepContext.for_fingerprint(off.runtime_args, off.deps))
    on_config = on.build_config(StepContext.for_fingerprint(on.runtime_args, on.deps))

    assert len(off.deps) == 1
    assert len(on.deps) > len(off.deps)
    assert off_config.eval is None
    assert on_config.eval is not None
    assert on_config.eval.steps_per_eval == 50


def test_ep_ablation_defaults_match_the_documented_arm_and_scale_per_rack():
    one = abl.build_small_run(run_id="d768", size="d768", flavor="ep", version="dev")
    cfg = one.build_config(StepContext.for_fingerprint(one.runtime_args, one.deps))
    m = cfg.model
    # The EP rung is a downsized hero: latent = hidden/2, capacity 1.33, top-k QB (the hero default).
    assert m.latent_dim == m.hidden_dim // 2
    assert m.capacity_factor == 1.33
    assert m.qb_estimator == model.QbEstimator.TOPK
    assert m.num_layers % 2 == 0  # even depth applied in the launcher, not GrugModelConfig
    # The histogram QB estimator is selectable on through the builder.
    hist = abl.build_small_run(run_id="d768-hist", size="d768", flavor="ep", qb_use_histogram=True, version="dev")
    assert hist.build_config(StepContext.for_fingerprint(hist.runtime_args, hist.deps)).model.qb_estimator == (
        model.QbEstimator.HIST
    )
    # The global batch scales with the rack count, holding the per-rack token load constant.
    four = abl.build_small_run(run_id="d2048", size="d2048", flavor="ep", dp_racks=4, version="dev")
    four_cfg = four.build_config(StepContext.for_fingerprint(four.runtime_args, four.deps))
    assert four_cfg.trainer.trainer.train_batch_size == cfg.trainer.trainer.train_batch_size * 4


def test_odd_depth_config_is_not_silently_rounded():
    # GrugModelConfig must preserve an odd depth so HF round-trips and odd configs stay faithful;
    # even-rounding is the launcher's job.
    cfg = model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
    )
    assert cfg.num_layers == 3


def test_hybrid_kv_branches_agree_on_sharding_when_model_axis_is_wide():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    script = """
        import math

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, set_mesh

        from experiments.grug.moe_hero_ep import model

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 2, 2),
            ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4,
        )
        cfg = model.GrugModelConfig(
            vocab_size=128,
            hidden_dim=32,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_shared_experts=1,
            num_experts=4,
            num_experts_per_token=1,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            local_kv_heads=2,
            global_kv_heads=1,
            head_dim=8,
            max_seq_len=8,
            sliding_window=4,
            global_every=2,
            capacity_factor=1.0,
            initializer_std=0.5 / math.sqrt(32),
            qk_mult=1.3,
            attention_implementation="reference",
            moe_implementation="fixed_all_to_all",
            report_capacity_overflow=True,
        )
        tokens = jax.ShapeDtypeStruct((2, 8), jnp.int32)
        with set_mesh(mesh):
            output = jax.eval_shape(
                lambda token_ids: model.Transformer.init(cfg, key=jax.random.key(0))(token_ids)[0],
                tokens,
            )
        assert output.shape == (2, 8, 32)
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _explicit_mesh(*axis_sizes):
    return Mesh(
        np.asarray(jax.devices()).reshape(*axis_sizes),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _latent_config(latent_dim=None):
    return model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        latent_dim=latent_dim,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        report_capacity_overflow=True,
    )


def test_latent_moe_shrinks_the_dispatched_width_but_not_the_token():
    # The point of LatentMoE is that the all-to-all payload narrows while the residual stream does
    # not, so the expert weights must be latent-wide and the layer output hidden-wide.
    mesh = _explicit_mesh(1, 1, 1, 1)
    cfg = _latent_config(latent_dim=16)
    tokens = jax.ShapeDtypeStruct((1, 8), jnp.int32)
    with set_mesh(mesh):
        built = jax.eval_shape(lambda: model.MoEMLP.init(cfg, key=jax.random.key(0)))
        out = jax.eval_shape(lambda t: model.Transformer.init(cfg, key=jax.random.key(0))(t)[0], tokens)

    assert built.w_latent_down.shape == (cfg.hidden_dim, 16)
    # Normalizing the latent keeps the expert input at unit scale despite the down-projection.
    assert built.latent_norm.weight.shape == (16,)
    assert built.w_latent_up.shape == (16, cfg.hidden_dim)
    # Expert banks are latent-wide: this is what narrows the dispatch.
    assert built.expert_mlp.w_gate.shape[1] == 16
    assert built.expert_mlp.w_up.shape[1] == 16
    assert built.expert_mlp.w_down.shape[2] == 16
    # The residual stream is untouched.
    assert out.shape[-1] == cfg.hidden_dim


def test_latent_moe_is_absent_by_default():
    # A config without a latent width must keep the standard MoE layer.
    mesh = _explicit_mesh(1, 1, 1, 1)
    cfg = _latent_config(latent_dim=None)
    with set_mesh(mesh):
        built = jax.eval_shape(lambda: model.MoEMLP.init(cfg, key=jax.random.key(0)))
    assert built.w_latent_down is None and built.w_latent_up is None
    assert built.latent_norm is None
    assert built.expert_mlp.w_gate.shape[1] == cfg.hidden_dim


# mok_like constrains hidden / intermediate / shared-intermediate to multiples of 256.
_TRACE_HIDDEN = 256


def _traceable_config(moe_implementation, *, mok_like=None, **kw):
    return model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=_TRACE_HIDDEN,
        intermediate_dim=256,
        shared_expert_intermediate_dim=256,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        latent_dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=64,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(_TRACE_HIDDEN),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation=moe_implementation,
        report_capacity_overflow=True,
        mok_like=mok_like,
        **kw,
    )


def _fake_mok_like_mlp(x, *args, **kwargs):
    """Shape-faithful stand-in for the fused CUDA call, which needs Linux + a GPU."""
    latent_up = kwargs.get("latent_up")
    width = x.shape[-1] if latent_up is None else latent_up.shape[-1]
    return jnp.zeros((x.shape[0], width), dtype=x.dtype), jnp.array(0, dtype=jnp.int32)


def _traced_router_metric_keys(cfg, *, mok_runtime=None):
    """Trace a loss step and return the metric keys the model actually publishes."""
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )

    def step(tokens, weights):
        built = model.Transformer.init(cfg, key=jax.random.key(0))
        if mok_runtime is not None:
            built = built.bind_mok_like_runtime(mok_runtime)
        return built.next_token_loss(tokens, weights, return_router_metrics=True)

    with use_abstract_mesh(mesh):
        _, metrics = jax.eval_shape(
            step,
            jax.ShapeDtypeStruct((4, 8), jnp.int32),
            jax.ShapeDtypeStruct((4, 8), jnp.float32),
        )
    return set(metrics)


def test_every_moe_backend_publishes_the_same_router_metrics(monkeypatch):
    # Regression: the mok_like arm published only `capacity_overflow` while the shared
    # consumer indexed the sender/receiver split that the pooled-wave work introduced,
    # so every rank died with KeyError('sender_capacity_overflow') on the first step.
    # Importing the module and printing a launch plan both miss this -- the offending
    # dict is only built while tracing -- so trace each backend and compare key sets.
    ep_keys = _traced_router_metric_keys(
        _traceable_config(
            "fixed_pooled_wave_all_to_all",
            num_expert_waves=1,
            pooled_transport_capacity_factor=1.1,
        )
    )
    ragged_keys = _traced_router_metric_keys(_traceable_config("ragged_all_to_all"))

    monkeypatch.setattr(model, "mok_like_mlp", _fake_mok_like_mlp)
    mok_keys = _traced_router_metric_keys(
        _traceable_config("fixed_all_to_all", mok_like=MokLikeConfig(), remat_mode="offload_moe"),
        mok_runtime=object(),
    )

    assert ep_keys == ragged_keys == mok_keys
    # The keys train.py indexes on the live training path.
    assert {
        "qb_beta_per_layer",
        "train/cross_entropy_loss",
        "moe/dropped_assignments",
        "moe/sender_dropped_assignments",
        "moe/receiver_dropped_assignments",
    } <= mok_keys


def test_mok_like_runtime_handle_survives_the_master_param_round_trip():
    # Under FP32_PINNED_HOST the live params are recast from master_params every step,
    # and `mok_like_runtime` is a static field, so binding it only to `params` would
    # drop it after one update and fail the *second* step.
    sentinel = object()
    cfg = _traceable_config("fixed_all_to_all", mok_like=MokLikeConfig(), remat_mode="offload_moe")
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 4, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with use_abstract_mesh(mesh):
        state = jax.eval_shape(
            lambda: train.initial_state(
                cfg,
                optimizer=optax.adam(1e-3),
                mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
                key=jax.random.key(0),
                ema_beta=None,
                master_param_mode=train.MasterParamMode.FP32_PINNED_HOST,
                mok_like_runtime=sentinel,
            )
        )

    assert state.params.mok_like_runtime is sentinel
    assert state.master_params.mok_like_runtime is sentinel


def test_latent_moe_hf_config_roundtrip_preserves_the_architecture():
    cfg = _latent_config(latent_dim=16)

    hf_config = cfg.to_hf_config(cfg.vocab_size)
    roundtripped = model.GrugModelConfig.from_hf_config(hf_config)

    assert hf_config.to_dict()["latent_dim"] == 16
    assert hf_config.to_dict()[model.GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY] == 2
    assert roundtripped.latent_dim == 16


def test_latent_moe_state_dict_contains_the_projection_state():
    mesh = _explicit_mesh(1, 1, 1, 1)
    cfg = _latent_config(latent_dim=16)
    with set_mesh(mesh):
        built = model.Transformer.init(cfg, key=jax.random.key(0))
        state_dict = built.to_state_dict()
    block = next(iter(built.stacked_blocks.unstacked()))
    assert block.mlp.w_latent_down is not None
    assert block.mlp.latent_norm is not None
    assert block.mlp.w_latent_up is not None

    expected = {
        "model.layers.0.mlp.latent_down_proj.weight": jnp.swapaxes(block.mlp.w_latent_down, -1, -2),
        "model.layers.0.mlp.latent_norm.weight": block.mlp.latent_norm.weight,
        "model.layers.0.mlp.latent_up_proj.weight": jnp.swapaxes(block.mlp.w_latent_up, -1, -2),
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(state_dict[name], value)


def test_latent_dim_above_hidden_is_rejected():
    # A latent wider than the hidden dim adds communication instead of removing it.
    with pytest.raises(ValueError, match="latent_dim must be in"):
        _latent_config(latent_dim=99999)


def test_latent_moe_flops_replace_routed_width_and_add_projections():
    latent_dim = 16
    full_width_config = _latent_config(latent_dim=None)
    latent_config = _latent_config(latent_dim=latent_dim)
    full_width_flops, _ = train._compute_flops(model_config=full_width_config)
    latent_flops, _ = train._compute_flops(model_config=latent_config)

    routed_delta = (
        2
        * 3
        * latent_config.intermediate_dim
        * latent_config.num_experts_per_token
        * (latent_dim - latent_config.hidden_dim)
    )
    projection_flops = 2 * 2 * latent_config.hidden_dim * latent_dim
    expected_delta = 3 * latent_config.max_seq_len * latent_config.num_layers * (routed_delta + projection_flops)

    assert latent_flops - full_width_flops == expected_delta


class _TinyWatchModel(eqx.Module):
    weight: jax.Array

    def next_token_loss(
        self,
        tokens,
        loss_weight,
        *,
        mask,
        reduction,
        logsumexp_weight,
        return_router_metrics,
    ):
        del mask, reduction, logsumexp_weight, return_router_metrics
        error = self.weight * tokens.astype(self.weight.dtype) - loss_weight
        return jnp.mean(error**2), {}


def test_diagnostic_watch_stats_match_direct_gradient_and_parameter_norms():
    params = _TinyWatchModel(weight=jnp.array(2.0))
    batch = SimpleNamespace(
        tokens=jnp.array([1, 3], dtype=jnp.int32),
        loss_weight=jnp.array([0.5, 1.0]),
        attn_mask=None,
    )
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    watch = WatchConfig(interval=1)

    actual = train._compute_diagnostic_watch_stats(params, batch, mp, None, watch)
    grads = jax.grad(
        lambda model: model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=None,
            return_router_metrics=True,
        )[0]
    )(params)
    expected = compute_watch_stats(
        watch_targets=watch.watch_targets,
        include_norms=watch.include_norms,
        include_per_parameter_norms=watch.include_per_parameter_norms,
        include_histogram=watch.include_histograms,
        split_scan_layers=watch.split_scan_layers,
        params=params,
        grads=grads,
        model_tree_type=type(params),
    )

    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key])


def test_inline_watch_computes_stats_on_every_train_step(monkeypatch):
    params = _TinyWatchModel(weight=jnp.array(2.0))
    optimizer = optax.sgd(0.1)
    state = train.GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        master_params=None,
        opt_state=optimizer.init(params),
        ema_params=None,
        pending_qb_betas=jnp.zeros((1, 1)),
    )

    def loss_and_grads(current_params, batch, mp, z_loss):
        del batch, mp, z_loss
        loss = current_params.weight**2
        grads = _TinyWatchModel(weight=2 * current_params.weight)
        metrics = {"qb_beta_per_layer": jnp.zeros((1, 1))}
        return (loss, metrics), grads

    monkeypatch.setattr(train, "_apply_qb_betas", lambda model, qb_betas: model)
    monkeypatch.setattr(train, "_loss_and_grads", loss_and_grads)
    train_step = train._make_train_step(
        optimizer,
        jmp.get_policy("params=float32,compute=float32,output=float32"),
        z_loss_weight=0,
        ema_beta=None,
        watch_config=WatchConfig(interval=10),
    )

    state, _, step_zero_stats = train_step(state, jnp.array(0))
    state, _, step_one_stats = train_step(state, jnp.array(0))

    assert step_zero_stats is not None
    assert step_one_stats is not None
    np.testing.assert_allclose(step_zero_stats["grad/norm/total"], 4.0)
    np.testing.assert_allclose(step_one_stats["grad/norm/total"], 3.2)


def test_fp32_host_master_accumulates_updates_before_bfloat16_cast(monkeypatch):
    params = _TinyWatchModel(weight=jnp.array(1.0, dtype=jnp.bfloat16))
    master_params = _TinyWatchModel(weight=jnp.array(1.0, dtype=jnp.float32))
    optimizer = optax.sgd(0.1)
    state = train.GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        master_params=master_params,
        opt_state=optimizer.init(master_params),
        ema_params=None,
        pending_qb_betas=jnp.zeros((1, 1)),
    )

    def loss_and_grads(current_params, batch, mp, z_loss):
        del current_params, batch, mp, z_loss
        loss = jnp.array(0.0)
        grads = _TinyWatchModel(weight=jnp.array(0.01, dtype=jnp.bfloat16))
        metrics = {"qb_beta_per_layer": jnp.zeros((1, 1))}
        return (loss, metrics), grads

    monkeypatch.setattr(train, "_apply_qb_betas", lambda model, qb_betas: model)
    monkeypatch.setattr(train, "_loss_and_grads", loss_and_grads)
    train_step = train._make_train_step(
        optimizer,
        jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16"),
        z_loss_weight=0,
        ema_beta=None,
        master_param_mode=train.MasterParamMode.FP32_PINNED_HOST,
    )

    for _ in range(10):
        state, _, _ = train_step(state, jnp.array(0))

    assert state.master_params is not None
    assert state.master_params.weight.dtype == jnp.float32
    assert state.params.weight.dtype == jnp.bfloat16
    expected_master = 1.0 - 10 * 0.1 * float(jnp.array(0.01, dtype=jnp.bfloat16))
    np.testing.assert_allclose(state.master_params.weight, expected_master, rtol=1e-6)
    np.testing.assert_allclose(state.params.weight, jnp.asarray(expected_master, dtype=jnp.bfloat16))


def test_fp32_host_master_preserves_float32_initialization(monkeypatch):
    config = _latent_config()
    key = jax.random.key(17)
    mesh = _explicit_mesh(1, 1, 1, 1)
    monkeypatch.setattr(train, "_tree_to_memory_kind", lambda tree, memory_kind: tree)

    with set_mesh(mesh):
        expected = model.Transformer.init(config, key=key)
        state = train.initial_state(
            config,
            optimizer=optax.sgd(0.1),
            mp=jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16"),
            key=key,
            ema_beta=None,
            master_param_mode=train.MasterParamMode.FP32_PINNED_HOST,
        )

    assert state.master_params is not None
    expected_leaves = jax.tree.leaves(expected)
    master_leaves = jax.tree.leaves(state.master_params)
    param_leaves = jax.tree.leaves(state.params)
    for expected_leaf, master_leaf, param_leaf in zip(expected_leaves, master_leaves, param_leaves, strict=True):
        np.testing.assert_array_equal(master_leaf, expected_leaf)
        np.testing.assert_array_equal(param_leaf, expected_leaf.astype(jnp.bfloat16))
    assert any(
        not np.array_equal(master_leaf, param_leaf.astype(jnp.float32))
        for master_leaf, param_leaf in zip(master_leaves, param_leaves, strict=True)
    )


def test_drop_metrics_reports_sender_and_receiver_fractions():
    metrics = train._drop_metrics(
        jnp.array(5, dtype=jnp.int32),
        jnp.array(2, dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        batch_size=2,
        sequence_length=4,
        top_k=2,
        num_layers=1,
    )

    assert metrics == {
        "moe/dropped_assignments": 5,
        "moe/drop_fraction": 5 / 16,
        "moe/sender_dropped_assignments": 2,
        "moe/sender_drop_fraction": 2 / 16,
        "moe/receiver_dropped_assignments": 3,
        "moe/receiver_drop_fraction": 3 / 16,
        "moe/receiver_drop_fraction_of_received": 3 / 14,
    }
