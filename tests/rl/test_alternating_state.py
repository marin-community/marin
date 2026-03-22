# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from iris.marin_fs import url_to_fs

from marin.rl.alternating import (
    AlternatingClusterConfig,
    AlternatingPhaseQuotaConfig,
    AlternatingRLConfig,
    AlternatingRunPaths,
    AlternatingRunState,
    BootstrapCheckpointDType,
    HostPhaseStatus,
    MaterializedBatchesManifest,
    PhaseMetricsManifest,
    PolicyBootstrapFormat,
    PolicyManifest,
    RunStatus,
    SamplingHostAssignment,
    SamplingHostStatusManifest,
    SamplingManifest,
    read_materialized_batches_manifest,
    read_phase_metrics_manifest,
    read_policy_manifest,
    read_run_state,
    read_sampling_host_status,
    read_sampling_manifest,
    update_phase_metrics,
    utc_now_iso,
    write_materialized_batches_manifest,
    write_policy_manifest,
    write_run_state,
    write_sampling_host_status,
    write_sampling_manifest,
)
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig


@dataclass(frozen=True)
class DummyCheckpointer:
    base_path: str = "unused"


@dataclass(frozen=True)
class DummyTrainer:
    train_batch_size: int = 8
    checkpointer: DummyCheckpointer = DummyCheckpointer()
    seed: int = 0


@dataclass(frozen=True)
class DummyInference:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    tensor_parallel_size: int = 4


def _curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.math_env.MathEnv",
                    env_args={"seed": 0},
                ),
                sampling_params=SamplingParams(n_prompts=2, n_generations_per_prompt=4, max_output_tokens=32),
            )
        },
        max_seq_len=128,
        eval_n_examples=4,
    )


def _make_config(tmp_path) -> AlternatingRLConfig:
    return AlternatingRLConfig(
        run_id="alt-state-test",
        shared_root=tmp_path.as_posix(),
        image_digest="image@sha256:test",
        seed=7,
        cluster=AlternatingClusterConfig(
            tpu_name="test-pod",
            tpu_type="v5p-16",
            zone="us-east5-a",
            num_hosts=2,
            local_tensor_parallel_size=4,
        ),
        quotas=AlternatingPhaseQuotaConfig(
            steps_per_phase=3,
            num_train_steps=12,
            groups_per_training_step=2,
            eval_examples_per_lesson=4,
        ),
        trainer=DummyTrainer(),
        model=object(),
        optimizer=object(),
        loss=object(),
        curriculum=_curriculum(),
        inference=DummyInference(),
        replay_buffer=SimpleNamespace(alpha=1.0, max_samples=1, filter_out_groups_with_no_variance=False),
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
    )


def test_run_state_and_manifest_roundtrip(tmp_path):
    config = _make_config(tmp_path)
    paths = AlternatingRunPaths.from_config(config)

    run_state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.SAMPLING,
        phase_id=3,
        policy_version=3,
        source_global_step=9,
        num_hosts=config.cluster.num_hosts,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(3),
        current_levanter_checkpoint_path=f"{paths.levanter_checkpoints_root}/step-9",
        current_sampling_manifest=paths.sampling_manifest_path(3),
        current_materialized_manifest=None,
        last_completed_phase=2,
    )
    write_run_state(paths.run_state_path, run_state)
    assert read_run_state(paths.run_state_path) == run_state

    policy_manifest = PolicyManifest(
        policy_version=3,
        phase_id=2,
        source_global_step=9,
        policy_path=paths.policy_dir(3),
        levanter_checkpoint_path=f"{paths.levanter_checkpoints_root}/step-9",
        bootstrap_checkpoint_path=paths.policy_bootstrap_checkpoint_path(3, BootstrapCheckpointDType.BF16),
        bootstrap_checkpoint_dtype=BootstrapCheckpointDType.BF16,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        enable_fast_bootstrap=True,
        created_at=utc_now_iso(),
    )
    write_policy_manifest(paths.policy_manifest_path(3), policy_manifest)
    assert read_policy_manifest(paths.policy_manifest_path(3)) == policy_manifest

    sampling_manifest = SamplingManifest(
        phase_id=3,
        policy_version=3,
        policy_manifest_path=paths.policy_manifest_path(3),
        curriculum_state_path=paths.curriculum_state_path,
        curriculum_snapshot_path=paths.sampling_curriculum_snapshot_path(3),
        num_hosts=2,
        local_tensor_parallel_size=4,
        coordinator_host_ordinal=0,
        host_assignments=[
            SamplingHostAssignment(host_ordinal=0, seed=1000, target_train_groups=16),
            SamplingHostAssignment(host_ordinal=1, seed=1001, target_train_groups=16),
        ],
        frozen_lesson_weights={"math_full": 1.0},
        rollout_output_root=paths.sampling_phase_dir(3),
    )
    write_sampling_manifest(paths.sampling_manifest_path(3), sampling_manifest)
    assert read_sampling_manifest(paths.sampling_manifest_path(3)) == sampling_manifest

    host_status = SamplingHostStatusManifest(
        phase_id=3,
        policy_version=3,
        host_ordinal=1,
        status=HostPhaseStatus.SUCCEEDED,
        rollout_file_paths=[f"{paths.sampling_host_rollout_dir(3, 1)}/0001.pkl"],
        num_train_groups=16,
        lesson_rewards={"math_full": [1.0, 0.0]},
        created_at=utc_now_iso(),
    )
    write_sampling_host_status(paths.sampling_host_status_path(3, 1), host_status)
    assert read_sampling_host_status(paths.sampling_host_status_path(3, 1)) == host_status

    materialized_manifest = MaterializedBatchesManifest(
        phase_id=3,
        policy_version=3,
        input_rollout_paths=host_status.rollout_file_paths,
        num_rollout_groups=32,
        num_individual_rollouts=128,
        num_training_batches=12,
        global_batch_size=256,
        max_seq_len=2048,
        batch_paths=[
            paths.materialized_batch_path(3, 0),
            paths.materialized_batch_path(3, 1),
        ],
    )
    write_materialized_batches_manifest(paths.materialized_manifest_path(3), materialized_manifest)
    assert read_materialized_batches_manifest(paths.materialized_manifest_path(3)) == materialized_manifest


def test_json_writer_overwrites_atomically(tmp_path):
    config = _make_config(tmp_path)
    paths = AlternatingRunPaths.from_config(config)
    state_a = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.SAMPLING,
        phase_id=0,
        policy_version=0,
        source_global_step=0,
        num_hosts=config.cluster.num_hosts,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(0),
        current_levanter_checkpoint_path=None,
        current_sampling_manifest=None,
        current_materialized_manifest=None,
        last_completed_phase=-1,
    )
    state_b = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.TRAINING,
        phase_id=1,
        policy_version=1,
        source_global_step=3,
        num_hosts=config.cluster.num_hosts,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(1),
        current_levanter_checkpoint_path=f"{paths.levanter_checkpoints_root}/step-3",
        current_sampling_manifest=paths.sampling_manifest_path(1),
        current_materialized_manifest=paths.materialized_manifest_path(1),
        last_completed_phase=0,
    )

    write_run_state(paths.run_state_path, state_a)
    write_run_state(paths.run_state_path, state_b)

    fs, fs_path = url_to_fs(paths.run_state_path)
    with fs.open(fs_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["status"] == RunStatus.TRAINING
    assert payload["phase_id"] == 1
    assert read_run_state(paths.run_state_path) == state_b


def test_phase_metrics_merge_updates_fields_without_dropping_prior_values(tmp_path):
    config = _make_config(tmp_path)
    paths = AlternatingRunPaths.from_config(config)

    first = update_phase_metrics(
        paths.phase_metrics_path(2),
        phase_id=2,
        prepare_sampling_seconds=12.5,
        sampling_seconds=120.0,
    )
    second = update_phase_metrics(
        paths.phase_metrics_path(2),
        phase_id=2,
        training_seconds=45.0,
        export_seconds=8.0,
    )

    assert isinstance(first, PhaseMetricsManifest)
    assert first.prepare_sampling_seconds == 12.5
    assert first.sampling_seconds == 120.0
    assert first.training_seconds is None

    stored = read_phase_metrics_manifest(paths.phase_metrics_path(2))
    assert stored == second
    assert stored.prepare_sampling_seconds == 12.5
    assert stored.sampling_seconds == 120.0
    assert stored.training_seconds == 45.0
    assert stored.export_seconds == 8.0
    assert stored.total_recorded_seconds == 185.5


def test_policy_manifest_defaults_bootstrap_format_for_old_payloads(tmp_path):
    config = _make_config(tmp_path)
    paths = AlternatingRunPaths.from_config(config)
    fs, fs_path = url_to_fs(paths.policy_manifest_path(1))
    fs.makedirs(fs_path.rsplit("/", 1)[0], exist_ok=True)
    payload = {
        "policy_version": 1,
        "phase_id": 0,
        "source_global_step": 1,
        "policy_path": paths.policy_dir(1),
        "levanter_checkpoint_path": f"{paths.levanter_checkpoints_root}/step-1",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "enable_fast_bootstrap": True,
        "created_at": utc_now_iso(),
    }
    with fs.open(fs_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    manifest = read_policy_manifest(paths.policy_manifest_path(1))

    assert manifest.bootstrap_format == PolicyBootstrapFormat.HF_EXPORT
    assert manifest.bootstrap_checkpoint_path is None
    assert manifest.bootstrap_checkpoint_dtype is None
