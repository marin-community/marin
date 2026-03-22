# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampling-phase entrypoints for alternating RL."""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import socket
import time
from collections import defaultdict

import equinox as eqx
from iris.marin_fs import url_to_fs

from marin.rl.alternating.config import AlternatingRLConfig
from marin.rl.alternating.io import glob_paths
from marin.rl.alternating.state import (
    AlternatingRunPaths,
    HostPhaseStatus,
    PolicyBootstrapFormat,
    SamplingHostStatusManifest,
    read_policy_manifest,
    read_sampling_manifest,
    write_sampling_host_status,
    write_sampling_manifest,
)
from marin.rl.alternating.topology import apply_local_vllm_topology, local_vllm_topology
from marin.rl.curriculum import create_local_curriculum
from marin.rl.environments.base import load_environment_from_spec
from marin.rl.rollout_storage import FileRolloutWriter
from marin.rl.rollout_worker import create_inference_context
from marin.rl.types import RolloutBatch, RolloutGroup, RolloutMetadata, RolloutStats

logger = logging.getLogger(__name__)


def _curriculum_checkpoint_dir(paths: AlternatingRunPaths) -> str:
    return paths.curriculum_root


def _snapshot_curriculum_state(source_path: str, snapshot_path: str) -> None:
    source_fs, source_fs_path = url_to_fs(source_path)
    snapshot_fs, snapshot_fs_path = url_to_fs(snapshot_path)
    snapshot_parent = os.path.dirname(snapshot_fs_path)
    if snapshot_parent:
        snapshot_fs.makedirs(snapshot_parent, exist_ok=True)
    with source_fs.open(source_fs_path, "rb") as source_handle, snapshot_fs.open(snapshot_fs_path, "wb") as dest_handle:
        dest_handle.write(source_handle.read())


def _reward_history_from_batch(batch: RolloutBatch) -> dict[str, list[float]]:
    reward_history: dict[str, list[float]] = defaultdict(list)
    for group in batch.groups:
        for rollout in group.rollouts:
            reward_history[rollout.env_name].append(float(rollout.episode_reward))
    return dict(reward_history)


def _rollout_stats_for_lesson(batch: RolloutBatch, lesson_id: str) -> list[RolloutStats]:
    stats: list[RolloutStats] = []
    for group in batch.groups:
        for rollout in group.rollouts:
            stats.append(
                RolloutStats(
                    episode_reward=float(rollout.episode_reward),
                    env_example_id=rollout.env_example_id,
                    lesson_id=lesson_id,
                    temperature=float(rollout.temperature),
                    top_k=rollout.top_k,
                )
            )
    return stats


def _sample_lesson(weights: dict[str, float], rng: random.Random) -> str:
    lesson_ids = list(weights.keys())
    probs = [weights[lesson_id] for lesson_id in lesson_ids]
    return rng.choices(lesson_ids, weights=probs, k=1)[0]


def _policy_metadata(policy_version: int, phase_id: int, source_global_step: int) -> RolloutMetadata:
    return RolloutMetadata(
        worker_id=f"{socket.gethostname()}_{os.getpid()}",
        timestamp=time.time(),
        weight_step=source_global_step,
        policy_version=policy_version,
        phase_id=phase_id,
        source_global_step=source_global_step,
    )


def _sampling_inference_config(config: AlternatingRLConfig, policy_manifest, *, tensor_parallel_size: int):
    base_config = dataclasses.replace(
        config.inference,
        tensor_parallel_size=tensor_parallel_size,
    )

    if policy_manifest.bootstrap_format == PolicyBootstrapFormat.LEVANTER_CHECKPOINT:
        checkpoint_path = (
            policy_manifest.bootstrap_checkpoint_path
            or policy_manifest.levanter_checkpoint_path
            or policy_manifest.policy_path
        )
        if checkpoint_path is None:
            raise ValueError("checkpoint-native policy manifest is missing a checkpoint path")

        return dataclasses.replace(
            base_config,
            model_name=policy_manifest.model_name,
            enable_fast_bootstrap=True,
            bootstrap_checkpoint_path=checkpoint_path,
            bootstrap_checkpoint_format=policy_manifest.bootstrap_format.value,
            bootstrap_levanter_model_config=config.model,
            bootstrap_tokenizer_name=policy_manifest.tokenizer_name,
            bootstrap_vocab_size=config.vocab_size,
        )

    return dataclasses.replace(
        base_config,
        model_name=(
            policy_manifest.policy_path if not policy_manifest.enable_fast_bootstrap else policy_manifest.model_name
        ),
        enable_fast_bootstrap=policy_manifest.enable_fast_bootstrap,
        bootstrap_checkpoint_path=policy_manifest.policy_path if policy_manifest.enable_fast_bootstrap else None,
        bootstrap_checkpoint_format=policy_manifest.bootstrap_format.value,
    )


def _sample_batch(
    config: AlternatingRLConfig,
    lesson_id: str,
    inference_ctx,
    env_cache: dict[str, object],
    mode: str,
    rng_seed: int,
    metadata: RolloutMetadata,
) -> RolloutBatch | None:
    if lesson_id not in env_cache:
        env_cache[lesson_id] = load_environment_from_spec(config.curriculum.lessons[lesson_id].env_config)
    env = env_cache[lesson_id]
    lesson = config.curriculum.lessons[lesson_id]
    rollout_groups, _metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=lesson.sampling_params.n_prompts if mode == "train" else config.curriculum.eval_n_examples,
        n_generations=lesson.sampling_params.n_generations_per_prompt if mode == "train" else 1,
        temperature=lesson.sampling_params.temperature,
        prng_key=rng_seed,
        mode=mode,
        max_tokens=lesson.sampling_params.max_output_tokens,
        top_k=lesson.sampling_params.top_k,
        stop=lesson.sampling_params.stop_tokens,
        system_prompt=config.system_prompt,
    )
    if not rollout_groups:
        return None

    groups_with_metadata = []
    for group in rollout_groups:
        rollouts_with_metadata = [eqx.tree_at(lambda r: r.metadata, rollout, metadata) for rollout in group.rollouts]
        groups_with_metadata.append(RolloutGroup(rollouts=rollouts_with_metadata))

    return RolloutBatch(groups=groups_with_metadata, metadata=metadata)


def prepare_sampling_phase(config: AlternatingRLConfig, paths: AlternatingRunPaths, phase_id: int) -> None:
    """Run the worker-0 evaluation/freeze step before all-host sampling starts."""
    manifest_path = paths.sampling_manifest_path(phase_id)
    manifest = read_sampling_manifest(manifest_path)
    policy_manifest = read_policy_manifest(manifest.policy_manifest_path)
    topology = local_vllm_topology(config.cluster.tpu_type, manifest.local_tensor_parallel_size)
    apply_local_vllm_topology(topology)
    inference_config = _sampling_inference_config(
        config,
        policy_manifest,
        tensor_parallel_size=topology.tensor_parallel_size,
    )
    inference_ctx = create_inference_context("vllm", inference_config, False)
    env_cache: dict[str, object] = {}
    curriculum = create_local_curriculum(config.curriculum, checkpoint_path=_curriculum_checkpoint_dir(paths))
    rng = random.Random(config.seed + phase_id)
    metadata = _policy_metadata(policy_manifest.policy_version, phase_id, policy_manifest.source_global_step)

    try:
        for lesson_id in config.curriculum.lessons:
            batch = _sample_batch(
                config,
                lesson_id,
                inference_ctx,
                env_cache,
                "eval",
                rng.randint(0, 2**31 - 1),
                metadata,
            )
            if batch is None:
                continue
            curriculum.update_lesson_stats(
                _rollout_stats_for_lesson(batch, lesson_id),
                mode="eval",
                current_step=policy_manifest.source_global_step,
            )
        curriculum.save_checkpoint(_curriculum_checkpoint_dir(paths))
        _snapshot_curriculum_state(paths.curriculum_state_path, manifest.curriculum_snapshot_path)
        updated_manifest = dataclasses.replace(
            manifest,
            frozen_lesson_weights=curriculum.compute_sampling_weights(),
        )
        write_sampling_manifest(manifest_path, updated_manifest)
    finally:
        inference_ctx.shutdown()


def run_sampling_host(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    phase_id: int,
    host_ordinal: int,
) -> None:
    """Run one host-local vLLM sampler until its phase quota is satisfied."""
    manifest = read_sampling_manifest(paths.sampling_manifest_path(phase_id))
    policy_manifest = read_policy_manifest(manifest.policy_manifest_path)
    assignment = next(item for item in manifest.host_assignments if item.host_ordinal == host_ordinal)
    topology = local_vllm_topology(config.cluster.tpu_type, manifest.local_tensor_parallel_size)
    apply_local_vllm_topology(topology)
    inference_config = _sampling_inference_config(
        config,
        policy_manifest,
        tensor_parallel_size=topology.tensor_parallel_size,
    )
    inference_ctx = create_inference_context("vllm", inference_config, False)
    writer = FileRolloutWriter(paths.sampling_host_rollout_dir(phase_id, host_ordinal), max_rollout_files=1_000_000)
    env_cache: dict[str, object] = {}
    reward_history: dict[str, list[float]] = defaultdict(list)
    generated_groups = 0
    rng = random.Random(assignment.seed)
    metadata = _policy_metadata(policy_manifest.policy_version, phase_id, policy_manifest.source_global_step)
    status_path = paths.sampling_host_status_path(phase_id, host_ordinal)

    try:
        if not manifest.frozen_lesson_weights:
            raise ValueError(f"Sampling manifest has no frozen lesson weights for phase {phase_id}")

        while generated_groups < assignment.target_train_groups:
            lesson_id = _sample_lesson(manifest.frozen_lesson_weights, rng)
            batch = _sample_batch(
                config,
                lesson_id,
                inference_ctx,
                env_cache,
                "train",
                rng.randint(0, 2**31 - 1),
                metadata,
            )
            if batch is None:
                continue
            writer.write_batch(batch)
            reward_history[lesson_id].extend(
                reward for rewards in _reward_history_from_batch(batch).values() for reward in rewards
            )
            generated_groups += len(batch.groups)

        rollout_dir = paths.sampling_host_rollout_dir(phase_id, host_ordinal)
        write_sampling_host_status(
            status_path,
            SamplingHostStatusManifest(
                phase_id=phase_id,
                policy_version=policy_manifest.policy_version,
                host_ordinal=host_ordinal,
                status=HostPhaseStatus.SUCCEEDED,
                rollout_file_paths=glob_paths(f"{rollout_dir}/*.pkl"),
                num_train_groups=generated_groups,
                lesson_rewards=dict(reward_history),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
    except Exception as exc:
        logger.exception("sampling host failed for phase %d host %d", phase_id, host_ordinal)
        write_sampling_host_status(
            status_path,
            SamplingHostStatusManifest(
                phase_id=phase_id,
                policy_version=policy_manifest.policy_version,
                host_ordinal=host_ordinal,
                status=HostPhaseStatus.FAILED,
                rollout_file_paths=[],
                num_train_groups=generated_groups,
                lesson_rewards={},
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                error_message=str(exc),
            ),
        )
        raise
    finally:
        inference_ctx.shutdown()
