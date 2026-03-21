# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from marin.rl.alternating import (
    AlternatingClusterConfig,
    AlternatingPhaseQuotaConfig,
    AlternatingRLConfig,
    AlternatingRunPaths,
    HostPhaseStatus,
    MaterializedBatchesManifest,
    PolicyManifest,
    SamplingHostAssignment,
    SamplingHostStatusManifest,
    SamplingManifest,
    read_materialized_batches_manifest,
    write_policy_manifest,
    write_sampling_host_status,
    write_sampling_manifest,
)
from marin.rl.alternating.io import read_pickle, write_pickle
from marin.rl.alternating.materialization import run_materialization
from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.types import Rollout, RolloutBatch, RolloutGroup, RolloutMetadata


@dataclass(frozen=True)
class DummyCheckpointer:
    base_path: str = "unused"


@dataclass(frozen=True)
class DummyTrainer:
    train_batch_size: int = 2
    checkpointer: DummyCheckpointer = DummyCheckpointer()
    seed: int = 0


@dataclass(frozen=True)
class DummyInference:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: int = 256


@dataclass(frozen=True)
class DummyModel:
    flash_attention_block_size: int = 4


class DummyLoss:
    def compute_advantages(self, rollout_group: list[Rollout]) -> np.ndarray:
        return np.array([0.5, -0.5], dtype=np.float32)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2


def _curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        lessons={
            "math_full": LessonConfig(
                lesson_id="math_full",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.math_env.MathEnv",
                    env_args={"seed": 0},
                ),
                sampling_params=SamplingParams(n_prompts=1, n_generations_per_prompt=2, max_output_tokens=8),
            )
        },
        max_seq_len=256,
        eval_n_examples=4,
    )


def _make_config(tmp_path) -> AlternatingRLConfig:
    return AlternatingRLConfig(
        run_id="alt-materialize-test",
        shared_root=tmp_path.as_posix(),
        image_digest="image@sha256:test",
        seed=11,
        cluster=AlternatingClusterConfig(
            tpu_name="test-pod",
            tpu_type="v5p-8",
            zone="us-east5-a",
            num_hosts=1,
            local_tensor_parallel_size=1,
        ),
        quotas=AlternatingPhaseQuotaConfig(
            steps_per_phase=1,
            num_train_steps=4,
            groups_per_training_step=1,
            eval_examples_per_lesson=4,
        ),
        trainer=DummyTrainer(),
        model=DummyModel(),
        optimizer=object(),
        loss=DummyLoss(),
        curriculum=_curriculum(),
        inference=DummyInference(),
        replay_buffer=SimpleNamespace(alpha=1.0, max_samples=1, filter_out_groups_with_no_variance=False),
        tokenizer_name="unused-tokenizer",
    )


def _make_rollout(unique_id: int, reward: float, logprobs: list[float]) -> Rollout:
    return Rollout(
        env_name="math_full",
        env_example_id=f"example-{unique_id}",
        prompt_tokens=np.array([10 + unique_id, 20 + unique_id], dtype=np.int32),
        response_tokens=np.array([30 + unique_id, 40 + unique_id], dtype=np.int32),
        response_logprobs=np.array(logprobs, dtype=np.float32),
        token_rewards=np.array([reward / 2.0, reward / 2.0], dtype=np.float32),
        episode_reward=reward,
        temperature=1.0,
        top_k=16,
        is_truncated=False,
        metadata=RolloutMetadata(
            worker_id="host-0",
            timestamp=1.0,
            weight_step=0,
            policy_version=0,
            phase_id=0,
            source_global_step=0,
        ),
    )


def test_materialization_preserves_policy_logprobs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "marin.rl.alternating.materialization.AutoTokenizer.from_pretrained",
        lambda _name: DummyTokenizer(),
    )

    config = _make_config(tmp_path)
    paths = AlternatingRunPaths.from_config(config)

    write_policy_manifest(
        paths.policy_manifest_path(0),
        PolicyManifest(
            policy_version=0,
            phase_id=-1,
            source_global_step=0,
            policy_path=paths.policy_dir(0),
            levanter_checkpoint_path=None,
            model_name=config.inference.model_name,
            tokenizer_name=config.tokenizer_name,
            enable_fast_bootstrap=True,
            created_at="2026-03-20T00:00:00Z",
        ),
    )
    write_sampling_manifest(
        paths.sampling_manifest_path(0),
        SamplingManifest(
            phase_id=0,
            policy_version=0,
            policy_manifest_path=paths.policy_manifest_path(0),
            curriculum_state_path=paths.curriculum_state_path,
            curriculum_snapshot_path=paths.sampling_curriculum_snapshot_path(0),
            num_hosts=1,
            local_tensor_parallel_size=1,
            coordinator_host_ordinal=0,
            host_assignments=[SamplingHostAssignment(host_ordinal=0, seed=11, target_train_groups=1)],
            frozen_lesson_weights={"math_full": 1.0},
            rollout_output_root=paths.sampling_phase_dir(0),
        ),
    )

    rollout_path = f"{paths.sampling_host_rollout_dir(0, 0)}/0001.pkl"
    write_pickle(
        rollout_path,
        RolloutBatch(
            groups=[
                RolloutGroup(
                    rollouts=[
                        _make_rollout(1, 1.0, [-0.1, -0.2]),
                        _make_rollout(2, 0.0, [-0.3, -0.4]),
                    ]
                )
            ],
            metadata=RolloutMetadata(
                worker_id="host-0",
                timestamp=1.0,
                weight_step=0,
                policy_version=0,
                phase_id=0,
                source_global_step=0,
            ),
        ),
    )
    write_sampling_host_status(
        paths.sampling_host_status_path(0, 0),
        SamplingHostStatusManifest(
            phase_id=0,
            policy_version=0,
            host_ordinal=0,
            status=HostPhaseStatus.SUCCEEDED,
            rollout_file_paths=[rollout_path],
            num_train_groups=1,
            lesson_rewards={"math_full": [1.0, 0.0]},
            created_at="2026-03-20T00:00:00Z",
        ),
    )

    manifest_path = run_materialization(config, paths, phase_id=0)
    manifest = read_materialized_batches_manifest(manifest_path)
    assert isinstance(manifest, MaterializedBatchesManifest)
    assert manifest.input_rollout_paths == [rollout_path]
    assert manifest.num_training_batches == 1

    batch = read_pickle(manifest.batch_paths[0])
    first_four_logprobs = sorted(tuple(np.round(row[:4], 6)) for row in batch.policy_logprobs.array)
    assert first_four_logprobs == [
        (0.0, 0.0, -0.3, -0.4),
        (0.0, 0.0, -0.1, -0.2),
    ]


def test_materialization_does_not_reuse_rollouts_when_max_samples_is_one(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "marin.rl.alternating.materialization.AutoTokenizer.from_pretrained",
        lambda _name: DummyTokenizer(),
    )

    config = replace(
        _make_config(tmp_path),
        quotas=AlternatingPhaseQuotaConfig(
            steps_per_phase=2,
            num_train_steps=4,
            groups_per_training_step=1,
            eval_examples_per_lesson=4,
        ),
    )
    paths = AlternatingRunPaths.from_config(config)

    write_policy_manifest(
        paths.policy_manifest_path(0),
        PolicyManifest(
            policy_version=0,
            phase_id=-1,
            source_global_step=0,
            policy_path=paths.policy_dir(0),
            levanter_checkpoint_path=None,
            model_name=config.inference.model_name,
            tokenizer_name=config.tokenizer_name,
            enable_fast_bootstrap=True,
            created_at="2026-03-20T00:00:00Z",
        ),
    )
    write_sampling_manifest(
        paths.sampling_manifest_path(0),
        SamplingManifest(
            phase_id=0,
            policy_version=0,
            policy_manifest_path=paths.policy_manifest_path(0),
            curriculum_state_path=paths.curriculum_state_path,
            curriculum_snapshot_path=paths.sampling_curriculum_snapshot_path(0),
            num_hosts=1,
            local_tensor_parallel_size=1,
            coordinator_host_ordinal=0,
            host_assignments=[SamplingHostAssignment(host_ordinal=0, seed=11, target_train_groups=2)],
            frozen_lesson_weights={"math_full": 1.0},
            rollout_output_root=paths.sampling_phase_dir(0),
        ),
    )

    rollout_path = f"{paths.sampling_host_rollout_dir(0, 0)}/0001.pkl"
    write_pickle(
        rollout_path,
        RolloutBatch(
            groups=[
                RolloutGroup(
                    rollouts=[
                        _make_rollout(1, 1.0, [-0.1, -0.2]),
                        _make_rollout(2, 0.0, [-0.3, -0.4]),
                    ]
                ),
                RolloutGroup(
                    rollouts=[
                        _make_rollout(3, 1.0, [-0.5, -0.6]),
                        _make_rollout(4, 0.0, [-0.7, -0.8]),
                    ]
                ),
            ],
            metadata=RolloutMetadata(
                worker_id="host-0",
                timestamp=1.0,
                weight_step=0,
                policy_version=0,
                phase_id=0,
                source_global_step=0,
            ),
        ),
    )
    write_sampling_host_status(
        paths.sampling_host_status_path(0, 0),
        SamplingHostStatusManifest(
            phase_id=0,
            policy_version=0,
            host_ordinal=0,
            status=HostPhaseStatus.SUCCEEDED,
            rollout_file_paths=[rollout_path],
            num_train_groups=2,
            lesson_rewards={"math_full": [1.0, 0.0, 1.0, 0.0]},
            created_at="2026-03-20T00:00:00Z",
        ),
    )

    manifest_path = run_materialization(config, paths, phase_id=0)
    manifest = read_materialized_batches_manifest(manifest_path)
    assert manifest.num_training_batches == 2

    all_rows = []
    for batch_path in manifest.batch_paths:
        batch = read_pickle(batch_path)
        all_rows.extend(tuple(np.round(row[:4], 6)) for row in batch.policy_logprobs.array)

    assert sorted(all_rows) == [
        (0.0, 0.0, -0.7, -0.8),
        (0.0, 0.0, -0.5, -0.6),
        (0.0, 0.0, -0.3, -0.4),
        (0.0, 0.0, -0.1, -0.2),
    ]
