# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from iris.marin_fs import url_to_fs
from levanter.tracker.wandb import WandbConfig

from marin.rl.alternating import (
    AlternatingClusterConfig,
    AlternatingPhaseQuotaConfig,
    AlternatingRLConfig,
    AlternatingRunPaths,
    HostPhaseStatus,
    MaterializedBatchesManifest,
    PolicyManifest,
    RunStatus,
    SamplingHostStatusManifest,
    bootstrap_or_resume,
    build_sampling_host_assignments,
    read_run_state,
    run_controller,
    update_phase_metrics,
    utc_now_iso,
    write_materialized_batches_manifest,
    write_policy_manifest,
    write_sampling_host_status,
)
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig


@dataclass(frozen=True)
class DummyCheckpointer:
    base_path: str = "unused"


@dataclass(frozen=True)
class DummyTrainer:
    train_batch_size: int = 8
    per_device_parallelism: int = 1
    checkpointer: DummyCheckpointer = DummyCheckpointer()
    seed: int = 0


@dataclass(frozen=True)
class DummyInference:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.9


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


def _make_config(tmp_path, *, image_digest: str = "image@sha256:test") -> AlternatingRLConfig:
    return AlternatingRLConfig(
        run_id="alt-controller-test",
        shared_root=tmp_path.as_posix(),
        image_digest=image_digest,
        seed=17,
        cluster=AlternatingClusterConfig(
            tpu_name="test-pod",
            tpu_type="v5p-16",
            zone="us-east5-a",
            num_hosts=2,
            local_tensor_parallel_size=4,
        ),
        quotas=AlternatingPhaseQuotaConfig(
            steps_per_phase=2,
            num_train_steps=2,
            groups_per_training_step=1,
            eval_examples_per_lesson=4,
            max_phases=1,
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


@dataclass
class FakeHooks:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    def bootstrap_initial_policy(self, config: AlternatingRLConfig, paths: AlternatingRunPaths) -> tuple[str, int]:
        manifest = PolicyManifest(
            policy_version=0,
            phase_id=-1,
            source_global_step=0,
            policy_path=paths.policy_dir(0),
            levanter_checkpoint_path=None,
            model_name=self.model_name,
            tokenizer_name=config.tokenizer_name,
            enable_fast_bootstrap=True,
            created_at=utc_now_iso(),
        )
        manifest_path = paths.policy_manifest_path(0)
        write_policy_manifest(manifest_path, manifest)
        return manifest_path, config.cluster.num_hosts

    def initialize_curriculum_state(self, config: AlternatingRLConfig, paths: AlternatingRunPaths) -> None:
        fs, fs_path = url_to_fs(paths.curriculum_state_path)
        fs.makedirs(paths.curriculum_root, exist_ok=True)
        with fs.open(fs_path, "wt", encoding="utf-8") as handle:
            json.dump({"initialized": True}, handle)

    def frozen_lesson_weights(self, config: AlternatingRLConfig, state, paths: AlternatingRunPaths) -> dict[str, float]:
        del config, state, paths
        return {"math_full": 1.0}

    def launch_sampling_phase(self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths) -> None:
        del config, state
        for assignment in manifest.host_assignments:
            status = SamplingHostStatusManifest(
                phase_id=manifest.phase_id,
                policy_version=manifest.policy_version,
                host_ordinal=assignment.host_ordinal,
                status=HostPhaseStatus.SUCCEEDED,
                rollout_file_paths=[
                    f"{paths.sampling_host_rollout_dir(manifest.phase_id, assignment.host_ordinal)}/0001.pkl"
                ],
                num_train_groups=assignment.target_train_groups,
                lesson_rewards={"math_full": [1.0, 0.0]},
                created_at=utc_now_iso(),
            )
            write_sampling_host_status(
                paths.sampling_host_status_path(manifest.phase_id, assignment.host_ordinal),
                status,
            )

    def wait_for_sampling_phase(self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths) -> None:
        del config, state, manifest, paths

    def update_curriculum_state(self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths) -> None:
        del config, state
        fs, fs_path = url_to_fs(paths.curriculum_state_path)
        with fs.open(fs_path, "wt", encoding="utf-8") as handle:
            json.dump({"updated_for_phase": manifest.phase_id}, handle)

    def materialize_training_batches(
        self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths
    ) -> str:
        del config, state
        materialized = MaterializedBatchesManifest(
            phase_id=manifest.phase_id,
            policy_version=manifest.policy_version,
            input_rollout_paths=[f"{paths.sampling_host_rollout_dir(manifest.phase_id, 0)}/0001.pkl"],
            num_rollout_groups=2,
            num_individual_rollouts=8,
            num_training_batches=2,
            global_batch_size=8,
            max_seq_len=128,
            batch_paths=[
                paths.materialized_batch_path(manifest.phase_id, 0),
                paths.materialized_batch_path(manifest.phase_id, 1),
            ],
        )
        manifest_path = paths.materialized_manifest_path(manifest.phase_id)
        write_materialized_batches_manifest(manifest_path, materialized)
        return manifest_path

    def run_training_phase(
        self,
        config: AlternatingRLConfig,
        state,
        manifest: MaterializedBatchesManifest,
        paths: AlternatingRunPaths,
    ) -> str:
        next_version = state.policy_version + 1
        policy = PolicyManifest(
            policy_version=next_version,
            phase_id=state.phase_id,
            source_global_step=state.source_global_step + len(manifest.batch_paths),
            policy_path=paths.policy_dir(next_version),
            levanter_checkpoint_path=(
                f"{paths.levanter_checkpoints_root}/step-{state.source_global_step + len(manifest.batch_paths)}"
            ),
            model_name=self.model_name,
            tokenizer_name=config.tokenizer_name,
            enable_fast_bootstrap=True,
            created_at=utc_now_iso(),
        )
        manifest_path = paths.policy_manifest_path(next_version)
        write_policy_manifest(manifest_path, policy)
        return manifest_path


@dataclass
class RecordingTracker:
    hparams: list[dict[str, object]]
    logs: list[tuple[dict[str, object], int | None]]
    summaries: list[dict[str, object]]
    finish_calls: int = 0

    def __init__(self):
        self.hparams = []
        self.logs = []
        self.summaries = []

    def log_hyperparameters(self, hparams: dict[str, object]) -> None:
        self.hparams.append(hparams)

    def log(self, metrics, *, step, commit=None) -> None:
        del commit
        self.logs.append((dict(metrics), step))

    def log_summary(self, metrics: dict[str, object]) -> None:
        self.summaries.append(dict(metrics))

    def finish(self) -> None:
        self.finish_calls += 1


class MetricHooks(FakeHooks):
    def launch_sampling_phase(self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths) -> None:
        super().launch_sampling_phase(config, state, manifest, paths)
        update_phase_metrics(
            paths.phase_metrics_path(manifest.phase_id),
            phase_id=manifest.phase_id,
            prepare_sampling_seconds=1.5,
        )

    def wait_for_sampling_phase(self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths) -> None:
        del config, state
        update_phase_metrics(
            paths.phase_metrics_path(manifest.phase_id),
            phase_id=manifest.phase_id,
            sampling_seconds=2.5,
        )

    def update_curriculum_state(self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths) -> None:
        super().update_curriculum_state(config, state, manifest, paths)
        update_phase_metrics(
            paths.phase_metrics_path(manifest.phase_id),
            phase_id=manifest.phase_id,
            curriculum_update_seconds=3.5,
        )

    def materialize_training_batches(
        self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths
    ) -> str:
        manifest_path = super().materialize_training_batches(config, state, manifest, paths)
        update_phase_metrics(
            paths.phase_metrics_path(manifest.phase_id),
            phase_id=manifest.phase_id,
            materialization_seconds=4.5,
        )
        return manifest_path

    def run_training_phase(
        self,
        config: AlternatingRLConfig,
        state,
        manifest: MaterializedBatchesManifest,
        paths: AlternatingRunPaths,
    ) -> str:
        manifest_path = super().run_training_phase(config, state, manifest, paths)
        update_phase_metrics(
            paths.phase_metrics_path(state.phase_id),
            phase_id=state.phase_id,
            training_seconds=5.5,
            export_seconds=6.5,
        )
        return manifest_path


class FailingHooks(FakeHooks):
    def materialize_training_batches(
        self, config: AlternatingRLConfig, state, manifest, paths: AlternatingRunPaths
    ) -> str:
        del config, state, manifest, paths
        raise RuntimeError("materialization exploded")


def test_build_sampling_host_assignments_derives_per_host_quota(tmp_path):
    config = _make_config(tmp_path)
    assignments = build_sampling_host_assignments(config, phase_id=3, num_hosts=config.cluster.num_hosts)

    assert [assignment.host_ordinal for assignment in assignments] == [0, 1]
    assert [assignment.seed for assignment in assignments] == [23, 24]
    assert all(assignment.target_train_groups == 1 for assignment in assignments)


def test_build_sampling_host_assignments_distributes_exact_total_when_uneven(tmp_path):
    config = _make_config(tmp_path)
    config = replace(
        config,
        quotas=AlternatingPhaseQuotaConfig(
            steps_per_phase=5,
            num_train_steps=5,
            groups_per_training_step=1,
            eval_examples_per_lesson=4,
            max_phases=1,
        ),
    )
    assignments = build_sampling_host_assignments(config, phase_id=0, num_hosts=2)

    assert [assignment.target_train_groups for assignment in assignments] == [3, 2]
    assert sum(assignment.target_train_groups for assignment in assignments) == 5


def test_validate_rejects_impossible_training_batch_for_cluster(tmp_path):
    config = _make_config(tmp_path)
    config = replace(
        config,
        cluster=replace(config.cluster, num_hosts=1),
        trainer=replace(config.trainer, train_batch_size=16, per_device_parallelism=16),
    )

    with pytest.raises(ValueError, match=r"trainer\.train_batch_size must be divisible by"):
        config.validate()


def test_validate_rejects_invalid_inference_memory_utilization(tmp_path):
    config = _make_config(tmp_path)
    config = replace(
        config,
        inference=replace(config.inference, gpu_memory_utilization=1.1),
    )

    with pytest.raises(ValueError, match=r"inference\.gpu_memory_utilization must be in the interval"):
        config.validate()


def test_controller_runs_one_phase_and_completes(tmp_path):
    config = _make_config(tmp_path)
    final_state = run_controller(config, FakeHooks())
    paths = AlternatingRunPaths.from_config(config)
    stored_state = read_run_state(paths.run_state_path)

    assert final_state == stored_state
    assert final_state.status == RunStatus.COMPLETED
    assert final_state.phase_id == 1
    assert final_state.policy_version == 1
    assert final_state.source_global_step == 2
    assert final_state.last_completed_phase == 0
    assert final_state.current_policy_manifest_path == paths.policy_manifest_path(1)
    assert final_state.current_levanter_checkpoint_path == f"{paths.levanter_checkpoints_root}/step-2"
    assert final_state.current_sampling_manifest is None
    assert final_state.current_materialized_manifest is None


def test_init_alternating_controller_tracker_uses_nested_replicate_path(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    config = replace(
        config,
        trainer=SimpleNamespace(
            tracker=WandbConfig(
                project="alternate_rl",
                name="base-run",
                tags=["rl"],
            )
        ),
    )
    paths = AlternatingRunPaths.from_config(config)
    captured = {}

    def _fake_init(self, run_id):
        captured["run_id"] = run_id
        captured["name"] = self.name
        captured["group"] = self.group
        captured["tags"] = self.tags
        captured["save_code"] = self.save_code
        captured["replicate_path"] = self.replicate_path
        return object()

    monkeypatch.setattr("marin.rl.alternating.wandb.WandbConfig.init", _fake_init)

    from marin.rl.alternating.wandb import init_alternating_controller_tracker

    tracker = init_alternating_controller_tracker(config, paths)

    assert tracker is not None
    assert captured == {
        "run_id": "alt-controller-test-alternating-controller",
        "name": "alt-controller-test-alternating-controller",
        "group": "alt-controller-test",
        "tags": ["rl", "alternating", "alternating-controller"],
        "save_code": False,
        "replicate_path": f"{paths.state_root}/wandb/alternating-controller",
    }


def test_controller_logs_phase_events_and_timings(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    tracker = RecordingTracker()
    monkeypatch.setattr(
        "marin.rl.alternating.controller.init_alternating_controller_tracker",
        lambda _config, _paths: tracker,
    )

    final_state = run_controller(config, MetricHooks())

    assert final_state.status == RunStatus.COMPLETED
    assert tracker.finish_calls == 1
    assert tracker.hparams == [
        {
            "alternating/run_id": config.run_id,
            "alternating/shared_root": config.shared_root,
            "alternating/image_digest": config.image_digest,
            "alternating/seed": config.seed,
            "alternating/resumed": False,
            "alternating/cluster/tpu_name": config.cluster.tpu_name,
            "alternating/cluster/tpu_type": config.cluster.tpu_type,
            "alternating/cluster/zone": config.cluster.zone,
            "alternating/cluster/num_hosts": config.cluster.num_hosts,
            "alternating/cluster/local_tensor_parallel_size": config.cluster.local_tensor_parallel_size,
            "alternating/cluster/capacity_type": config.cluster.capacity_type,
            "alternating/quotas/steps_per_phase": config.quotas.steps_per_phase,
            "alternating/quotas/num_train_steps": config.quotas.num_train_steps,
            "alternating/quotas/groups_per_training_step": config.quotas.groups_per_training_step,
            "alternating/quotas/eval_examples_per_lesson": config.quotas.eval_examples_per_lesson,
            "alternating/initial_checkpoint": config.initial_checkpoint,
            "alternating/tokenizer_name": config.tokenizer_name,
            "alternating/model_name": config.inference.model_name,
        }
    ]

    events = [metrics["alternating/controller/event"] for metrics, _step in tracker.logs]
    assert events == [
        "controller_started",
        "sampling_manifest_ready",
        "sampling_completed",
        "materialization_started",
        "materialization_completed",
        "training_completed",
        "phase_advanced",
        "completed",
    ]

    training_metrics = next(
        metrics for metrics, _step in tracker.logs if metrics["alternating/controller/event"] == "training_completed"
    )
    assert training_metrics["alternating/prepare_sampling_seconds"] == 1.5
    assert training_metrics["alternating/sampling_seconds"] == 2.5
    assert training_metrics["alternating/curriculum_update_seconds"] == 3.5
    assert training_metrics["alternating/materialization_seconds"] == 4.5
    assert training_metrics["alternating/training_seconds"] == 5.5
    assert training_metrics["alternating/export_seconds"] == 6.5
    assert training_metrics["alternating/phase_total_seconds"] == 24.0
    assert training_metrics["alternating/controller/materialized_training_batches"] == 2

    assert tracker.summaries == [
        {
            "alternating/controller/final_status": RunStatus.COMPLETED.value,
            "alternating/controller/final_phase_id": 1,
            "alternating/controller/final_policy_version": 1,
            "alternating/controller/final_source_global_step": 2,
            "alternating/controller/last_completed_phase": 0,
        }
    ]


def test_controller_finishes_tracker_and_records_failure(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    tracker = RecordingTracker()
    monkeypatch.setattr(
        "marin.rl.alternating.controller.init_alternating_controller_tracker",
        lambda _config, _paths: tracker,
    )

    with pytest.raises(RuntimeError, match="materialization exploded"):
        run_controller(config, FailingHooks())

    paths = AlternatingRunPaths.from_config(config)
    failed_state = read_run_state(paths.run_state_path)

    assert failed_state.status == RunStatus.FAILED
    assert tracker.finish_calls == 1
    assert tracker.logs[-1][0]["alternating/controller/event"] == "failed"
    assert tracker.summaries[-1]["alternating/controller/final_status"] == RunStatus.FAILED.value
    assert "materialization exploded" in tracker.summaries[-1]["alternating/controller/error_message"]


def test_bootstrap_rejects_image_digest_drift(tmp_path):
    hooks = FakeHooks()
    config = _make_config(tmp_path, image_digest="image@sha256:one")
    paths = AlternatingRunPaths.from_config(config)
    bootstrap_or_resume(config, hooks, paths)

    mismatched = _make_config(tmp_path, image_digest="image@sha256:two")
    with pytest.raises(ValueError, match="image digest drifted"):
        bootstrap_or_resume(mismatched, hooks, paths)
