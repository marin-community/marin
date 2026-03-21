# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline materialization of raw rollout batches into training batches."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from levanter.layers.attention import AttentionBackend, DEFAULT_SPLASH_BLOCK_SIZE
from levanter.models.flash_attention import BLOCK_SIZE as DEFAULT_FLASH_BLOCK_SIZE
from transformers import AutoTokenizer

from marin.rl.alternating.config import AlternatingRLConfig
from marin.rl.alternating.io import read_pickle, write_pickle
from marin.rl.alternating.state import (
    AlternatingRunPaths,
    MaterializedBatchesManifest,
    SamplingManifest,
    read_sampling_host_status,
    read_sampling_manifest,
    write_materialized_batches_manifest,
)
from marin.rl.replay_buffer import RolloutWithCount
from marin.rl.train_batch import create_training_batch_from_rollouts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SpillShard:
    env_name: str
    path: str
    num_records: int


@dataclass
class _SpillIndex:
    spill_root: str
    shards_by_env: dict[str, list[_SpillShard]] = field(default_factory=dict)
    num_input_rollout_files: int = 0
    num_rollout_groups: int = 0
    num_individual_rollouts: int = 0

    @property
    def num_spill_shards(self) -> int:
        return sum(len(shards) for shards in self.shards_by_env.values())


@dataclass(frozen=True)
class _BatchAssignment:
    batch_index: int
    slot_index: int


@dataclass
class _MaterializationPlan:
    assignments_by_env: dict[str, dict[int, list[_BatchAssignment]]] = field(default_factory=dict)


def _pad_to_multiple(config: AlternatingRLConfig) -> int:
    is_splash = getattr(config.model, "attn_backend", None) == AttentionBackend.SPLASH
    flash_block_size = getattr(config.model, "flash_attention_block_size", None)
    if is_splash:
        return flash_block_size or DEFAULT_SPLASH_BLOCK_SIZE
    return flash_block_size or DEFAULT_FLASH_BLOCK_SIZE


def _sample_rollout_indices(
    available_indices_by_env: dict[str, list[int]],
    usage_counts_by_env: dict[str, list[int]],
    *,
    batch_size: int,
    alpha: float,
    max_samples: int,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    env_names = [name for name, rollout_indices in available_indices_by_env.items() if rollout_indices]
    if not env_names:
        raise ValueError("materializer has no rollout data to sample")

    available_counts = np.asarray([len(available_indices_by_env[env_name]) for env_name in env_names], dtype=np.int64)
    total_count = int(available_counts.sum())

    if total_count < batch_size:
        raise ValueError(
            "materializer does not have enough individual rollouts to build the next batch: "
            f"have={total_count}, need={batch_size}"
        )

    sampled_env_positions = rng.choice(total_count, size=batch_size, replace=False)
    cumulative_counts = np.cumsum(available_counts)
    env_bucket_indices = np.searchsorted(cumulative_counts, sampled_env_positions, side="right")
    env_counts: dict[str, int] = {}
    for env_bucket_index in env_bucket_indices.tolist():
        env_name = env_names[int(env_bucket_index)]
        env_counts[env_name] = env_counts.get(env_name, 0) + 1

    sampled: list[tuple[str, int]] = []
    for env_name, count in env_counts.items():
        available_indices = available_indices_by_env[env_name]
        weights = np.arange(len(available_indices), dtype=np.float64) + 1
        weights = weights**alpha
        weights = weights / weights.sum()
        selected_positions = rng.choice(len(available_indices), size=count, replace=False, p=weights)
        selected_position_list = selected_positions.tolist()
        for position in selected_position_list:
            rollout_index = available_indices[position]
            usage_counts_by_env[env_name][rollout_index] += 1
            sampled.append((env_name, rollout_index))

        if max_samples >= 0:
            for position in sorted(selected_position_list, reverse=True):
                rollout_index = available_indices[position]
                if usage_counts_by_env[env_name][rollout_index] >= max_samples:
                    del available_indices[position]

    return sampled


def _spill_records(
    spill_index: _SpillIndex,
    *,
    env_name: str,
    records: list[RolloutWithCount],
) -> None:
    if not records:
        return

    env_dir = os.path.join(spill_index.spill_root, env_name)
    os.makedirs(env_dir, exist_ok=True)
    shard_ordinal = len(spill_index.shards_by_env.get(env_name, []))
    shard_path = os.path.join(env_dir, f"shard_{shard_ordinal:06d}.pkl")
    write_pickle(shard_path, records)
    spill_index.shards_by_env.setdefault(env_name, []).append(
        _SpillShard(env_name=env_name, path=shard_path, num_records=len(records))
    )


def _spill_rollouts_for_phase(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    sampling_manifest: SamplingManifest,
    *,
    phase_id: int,
) -> tuple[_SpillIndex, list[str]]:
    spill_root = tempfile.mkdtemp(prefix=f"marin-alt-phase-{phase_id:04d}-")
    spill_index = _SpillIndex(spill_root=spill_root)
    input_rollout_paths: list[str] = []

    for assignment in sampling_manifest.host_assignments:
        status = read_sampling_host_status(paths.sampling_host_status_path(phase_id, assignment.host_ordinal))
        input_rollout_paths.extend(status.rollout_file_paths)

    spill_index.num_input_rollout_files = len(input_rollout_paths)

    for rollout_path in sorted(input_rollout_paths):
        batch = read_pickle(rollout_path)
        if batch.metadata.policy_version != sampling_manifest.policy_version:
            raise ValueError(
                "rollout batch policy_version mismatch during materialization: "
                f"path={rollout_path}, batch={batch.metadata.policy_version}, "
                f"expected={sampling_manifest.policy_version}"
            )

        per_env_records: dict[str, list[RolloutWithCount]] = defaultdict(list)
        for group in batch.groups:
            spill_index.num_rollout_groups += 1
            advantages = config.loss.compute_advantages(group.rollouts)
            episode_rewards = np.array([rollout.episode_reward for rollout in group.rollouts], dtype=np.float32)
            if np.std(episode_rewards) == 0.0 and config.replay_buffer.filter_out_groups_with_no_variance:
                continue

            for rollout, advantage in zip(group.rollouts, advantages, strict=True):
                per_env_records[rollout.env_name].append(
                    RolloutWithCount(
                        rollout=rollout,
                        advantage=float(advantage),
                        usage_count=0,
                        weight_step=rollout.metadata.weight_step,
                    )
                )
                spill_index.num_individual_rollouts += 1

        for env_name, records in per_env_records.items():
            _spill_records(spill_index, env_name=env_name, records=records)

    logger.info(
        "spilled phase %d rollouts into %d local shards under %s",
        phase_id,
        spill_index.num_spill_shards,
        spill_index.spill_root,
    )
    return spill_index, sorted(input_rollout_paths)


def _plan_materialized_batches(
    config: AlternatingRLConfig,
    spill_index: _SpillIndex,
    *,
    phase_id: int,
) -> _MaterializationPlan:
    available_indices_by_env = {
        env_name: list(range(sum(shard.num_records for shard in shards)))
        for env_name, shards in spill_index.shards_by_env.items()
    }
    usage_counts_by_env = {
        env_name: [0] * len(available_indices) for env_name, available_indices in available_indices_by_env.items()
    }
    plan = _MaterializationPlan(assignments_by_env=defaultdict(dict))
    rng = np.random.default_rng(config.seed + phase_id)

    for batch_index in range(config.quotas.steps_per_phase):
        sampled_rollout_indices = _sample_rollout_indices(
            available_indices_by_env=available_indices_by_env,
            usage_counts_by_env=usage_counts_by_env,
            batch_size=config.global_batch_size,
            alpha=config.replay_buffer.alpha,
            max_samples=config.replay_buffer.max_samples,
            rng=rng,
        )
        for slot_index, (env_name, rollout_index) in enumerate(sampled_rollout_indices):
            env_assignments = plan.assignments_by_env.setdefault(env_name, {})
            env_assignments.setdefault(rollout_index, []).append(
                _BatchAssignment(batch_index=batch_index, slot_index=slot_index)
            )

    return plan


def _append_batch_assignment(batch_record_path: str, assignment: _BatchAssignment, rollout: RolloutWithCount) -> None:
    os.makedirs(os.path.dirname(batch_record_path), exist_ok=True)
    with open(batch_record_path, "ab") as handle:
        pickle.dump((assignment.slot_index, rollout), handle)


def _iter_pickled_records(path: str):
    with open(path, "rb") as handle:
        while True:
            try:
                yield pickle.load(handle)
            except EOFError:
                return


def _materialize_spill_plan(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
    spill_index: _SpillIndex,
    *,
    phase_id: int,
    pad_token_id: int,
    pad_to_multiple: int,
) -> list[str]:
    plan = _plan_materialized_batches(config, spill_index, phase_id=phase_id)
    batch_record_root = os.path.join(spill_index.spill_root, "planned-batches")
    batch_record_paths = [
        os.path.join(batch_record_root, f"batch_{batch_index:06d}.pkl")
        for batch_index in range(config.quotas.steps_per_phase)
    ]

    for env_name, shards in spill_index.shards_by_env.items():
        env_assignments = plan.assignments_by_env.get(env_name)
        if not env_assignments:
            continue

        rollout_index = 0
        for shard in shards:
            records: list[RolloutWithCount] = read_pickle(shard.path)
            for record in records:
                assignments = env_assignments.get(rollout_index)
                if assignments:
                    for assignment in assignments:
                        _append_batch_assignment(batch_record_paths[assignment.batch_index], assignment, record)
                rollout_index += 1

    batch_paths: list[str] = []
    for batch_index, batch_record_path in enumerate(batch_record_paths):
        slotted_rollouts = list(_iter_pickled_records(batch_record_path))
        if len(slotted_rollouts) != config.global_batch_size:
            raise ValueError(
                "materializer planned a batch with the wrong number of rollouts: "
                f"batch_index={batch_index}, have={len(slotted_rollouts)}, expected={config.global_batch_size}"
            )

        sampled_rollouts = [rollout for _, rollout in sorted(slotted_rollouts, key=lambda item: item[0])]
        batch = create_training_batch_from_rollouts(
            sampled_rollouts,
            config.curriculum.max_seq_len,
            pad_token_id,
            pad_to_multiple,
        )
        batch_path = paths.materialized_batch_path(phase_id, batch_index)
        write_pickle(batch_path, batch)
        batch_paths.append(batch_path)

    return batch_paths


def run_materialization(config: AlternatingRLConfig, paths: AlternatingRunPaths, phase_id: int) -> str:
    """Materialize deterministic training batches for one completed sampling phase."""
    sampling_manifest = read_sampling_manifest(paths.sampling_manifest_path(phase_id))
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(f"Tokenizer {config.tokenizer_name} has neither pad_token_id nor eos_token_id")

    spill_index, input_rollout_paths = _spill_rollouts_for_phase(
        config,
        paths,
        sampling_manifest,
        phase_id=phase_id,
    )
    try:
        pad_to_multiple = _pad_to_multiple(config)
        batch_paths = _materialize_spill_plan(
            config,
            paths,
            spill_index,
            phase_id=phase_id,
            pad_token_id=pad_token_id,
            pad_to_multiple=pad_to_multiple,
        )

        manifest = MaterializedBatchesManifest(
            phase_id=phase_id,
            policy_version=sampling_manifest.policy_version,
            input_rollout_paths=input_rollout_paths,
            num_rollout_groups=spill_index.num_rollout_groups,
            num_individual_rollouts=spill_index.num_individual_rollouts,
            num_training_batches=len(batch_paths),
            global_batch_size=config.global_batch_size,
            max_seq_len=config.curriculum.max_seq_len,
            batch_paths=batch_paths,
        )
        manifest_path = paths.materialized_manifest_path(phase_id)
        write_materialized_batches_manifest(manifest_path, manifest)
        logger.info(
            "materialized phase %d into %d batches from %d rollout files via %d spill shards",
            phase_id,
            len(batch_paths),
            len(input_rollout_paths),
            spill_index.num_spill_shards,
        )
        return manifest_path
    finally:
        shutil.rmtree(spill_index.spill_root, ignore_errors=True)
