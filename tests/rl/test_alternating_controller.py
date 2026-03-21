# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from iris.marin_fs import url_to_fs

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


def test_bootstrap_rejects_image_digest_drift(tmp_path):
    hooks = FakeHooks()
    config = _make_config(tmp_path, image_digest="image@sha256:one")
    paths = AlternatingRunPaths.from_config(config)
    bootstrap_or_resume(config, hooks, paths)

    mismatched = _make_config(tmp_path, image_digest="image@sha256:two")
    with pytest.raises(ValueError, match="image digest drifted"):
        bootstrap_or_resume(mismatched, hooks, paths)
