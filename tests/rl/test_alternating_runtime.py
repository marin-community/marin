# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marin.rl.alternating import (
    AlternatingRunPaths,
    ExistingPodPhaseHooks,
    HostPhaseStatus,
    SamplingHostAssignment,
    SamplingHostStatusManifest,
    SamplingManifest,
    write_sampling_host_status,
)


def test_sampling_wait_fails_fast_when_any_host_reports_failure(tmp_path):
    paths = AlternatingRunPaths(run_root=tmp_path.as_posix())
    manifest = SamplingManifest(
        phase_id=0,
        policy_version=0,
        policy_manifest_path=paths.policy_manifest_path(0),
        curriculum_state_path=paths.curriculum_state_path,
        curriculum_snapshot_path=paths.sampling_curriculum_snapshot_path(0),
        num_hosts=2,
        local_tensor_parallel_size=1,
        coordinator_host_ordinal=0,
        host_assignments=[
            SamplingHostAssignment(host_ordinal=0, seed=11, target_train_groups=1),
            SamplingHostAssignment(host_ordinal=1, seed=12, target_train_groups=1),
        ],
        frozen_lesson_weights={"math_full": 1.0},
        rollout_output_root=paths.sampling_phase_dir(0),
    )
    write_sampling_host_status(
        paths.sampling_host_status_path(0, 0),
        SamplingHostStatusManifest(
            phase_id=0,
            policy_version=0,
            host_ordinal=0,
            status=HostPhaseStatus.FAILED,
            rollout_file_paths=[],
            num_train_groups=0,
            lesson_rewards={},
            created_at="2026-03-20T00:00:00Z",
            error_message="host crashed",
        ),
    )

    hooks = ExistingPodPhaseHooks()
    with pytest.raises(RuntimeError, match="sampling host reported failure before phase completion"):
        hooks.wait_for_sampling_phase(None, None, manifest, paths)


def test_sampling_wait_fails_when_container_exits_without_status(monkeypatch, tmp_path):
    paths = AlternatingRunPaths(run_root=tmp_path.as_posix())
    manifest = SamplingManifest(
        phase_id=0,
        policy_version=0,
        policy_manifest_path=paths.policy_manifest_path(0),
        curriculum_state_path=paths.curriculum_state_path,
        curriculum_snapshot_path=paths.sampling_curriculum_snapshot_path(0),
        num_hosts=1,
        local_tensor_parallel_size=1,
        coordinator_host_ordinal=0,
        host_assignments=[
            SamplingHostAssignment(host_ordinal=0, seed=11, target_train_groups=1),
        ],
        frozen_lesson_weights={"math_full": 1.0},
        rollout_output_root=paths.sampling_phase_dir(0),
    )
    config = SimpleNamespace(
        cluster=SimpleNamespace(
            tpu_name="test-pod",
            zone="us-east5-a",
            node_count=1,
        )
    )
    hooks = ExistingPodPhaseHooks()

    monkeypatch.setattr(
        "marin.rl.alternating.runtime.DEFAULT_POLL_INTERVAL",
        0,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.runtime.tpus.container_exists_on_worker",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.runtime.time.sleep",
        lambda _: None,
    )

    with pytest.raises(RuntimeError, match="container exited before writing status"):
        hooks.wait_for_sampling_phase(config, None, manifest, paths)
