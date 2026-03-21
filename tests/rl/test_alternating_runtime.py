# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from marin.rl.alternating import (
    AlternatingRunPaths,
    ExistingPodPhaseHooks,
    HostPhaseStatus,
    SamplingHostAssignment,
    SamplingHostStatusManifest,
    SamplingManifest,
    save_controller_config,
    write_sampling_host_status,
)
from marin.rl.alternating.io import read_pickle
from marin.rl.environments.inference_ctx import VllmSamplingConfig, vLLMInferenceContextConfig


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


def test_sampling_wait_reports_tpu_infrastructure_loss(monkeypatch, tmp_path):
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

    monkeypatch.setattr("marin.rl.alternating.runtime.DEFAULT_POLL_INTERVAL", 0)
    monkeypatch.setattr(
        "marin.rl.alternating.runtime.tpus.container_exists_on_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["gcloud"])),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.runtime.tpus.describe_tpu_queued_resource",
        lambda *args, **kwargs: {"state": {"state": "SUSPENDING"}},
    )

    with pytest.raises(RuntimeError, match="TPU became unavailable while waiting for sampling host completion"):
        hooks.wait_for_sampling_phase(config, None, manifest, paths)


def test_controller_config_pickle_round_trip_preserves_vllm_sampling_config(tmp_path):
    inference = vLLMInferenceContextConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=2048,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        sampling_params=VllmSamplingConfig(
            temperature=0.7,
            n=8,
            max_tokens=256,
            top_k=4096,
            stop=["<|eot_id|>"],
            include_stop_str_in_output=True,
            logprobs=1,
        ),
    )
    config = SimpleNamespace(
        run_id="pickle-test",
        run_root=tmp_path.as_posix(),
        inference=inference,
    )
    paths = SimpleNamespace(controller_config_path=tmp_path.joinpath("controller_config.pkl").as_posix())

    save_controller_config(config, paths)
    loaded = read_pickle(paths.controller_config_path)

    assert isinstance(loaded.inference.sampling_params, VllmSamplingConfig)
    assert loaded.inference.sampling_params.stop == ["<|eot_id|>"]
    assert loaded.inference.sampling_params.include_stop_str_in_output is True
