# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming two-pass materializer for alternating RL.

Converts raw rollout files from a sampling phase into a deterministic finite
set of TrainingBatch files ready for Levanter training.

Pass 1: ingest raw rollout files, compute advantages, spill to local disk
         per environment.
Pass 2: sample and pack exactly `steps_per_phase` training batches from the
         spill files.

Memory is bounded by one input rollout file + small per-env metadata.
"""

import logging
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np
from iris.marin_fs import url_to_fs
from transformers import AutoTokenizer

from marin.rl.alternating.state import (
    MaterializationManifest,
    SamplingManifest,
    read_json_from_path,
    write_json_to_path,
)
from marin.rl.rl_losses import RLLossModule
from marin.rl.train_batch import create_training_batch_from_rollouts
from marin.rl.types import RolloutBatch, RolloutWithAdvantage

logger = logging.getLogger(__name__)


@dataclass
class MaterializerConfig:
    """Configuration for the materializer."""

    sampling_manifest_path: str
    output_dir: str
    loss_module: RLLossModule
    tokenizer_name: str
    steps_per_phase: int
    global_batch_size: int
    max_seq_len: int
    alpha: float = 3.0
    max_samples: int = 1
    seed: int | None = None
    spill_dir: str | None = None
    pad_to_multiple: int | None = None


@dataclass
class _SpillIndex:
    """Lightweight in-memory index over spill files."""

    env_shards: dict[str, list[str]] = field(default_factory=dict)
    env_counts: dict[str, int] = field(default_factory=dict)
    total_groups: int = 0
    total_individual: int = 0


def run_materializer(config: MaterializerConfig) -> MaterializationManifest:
    """Run the full two-pass materialization pipeline."""
    manifest = SamplingManifest.from_json(read_json_from_path(config.sampling_manifest_path))
    seed = config.seed if config.seed is not None else manifest.phase_id

    spill_base = config.spill_dir or tempfile.mkdtemp(prefix="marin-alt-materializer-")
    spill_dir = os.path.join(spill_base, f"phase_{manifest.phase_id:04d}")
    os.makedirs(spill_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    try:
        # Pass 1: ingest and spool
        logger.info("Pass 1: ingesting raw rollouts for phase %d", manifest.phase_id)
        spill_index = _pass1_ingest(
            manifest=manifest,
            loss_module=config.loss_module,
            policy_version=manifest.policy_version,
            spill_dir=spill_dir,
        )
        logger.info(
            "Pass 1 done: %d groups, %d individual rollouts across %d envs",
            spill_index.total_groups,
            spill_index.total_individual,
            len(spill_index.env_shards),
        )

        # Pass 2: sample and pack
        logger.info("Pass 2: sampling and packing %d training batches", config.steps_per_phase)
        mat_manifest = _pass2_sample_and_pack(
            spill_index=spill_index,
            manifest=manifest,
            config=config,
            seed=seed,
            pad_token_id=pad_token_id,
        )

        # Write manifest
        manifest_path = os.path.join(config.output_dir, "manifest.json")
        write_json_to_path(manifest_path, mat_manifest.to_json())
        logger.info("Materialization manifest written to %s", manifest_path)

        return mat_manifest

    finally:
        # Clean up spill directory
        if config.spill_dir is None and os.path.exists(spill_base):
            shutil.rmtree(spill_base, ignore_errors=True)


def _pass1_ingest(
    manifest: SamplingManifest,
    loss_module: RLLossModule,
    policy_version: int,
    spill_dir: str,
) -> _SpillIndex:
    """Pass 1: read raw rollout files, compute advantages, spill to disk."""
    index = _SpillIndex()

    # Collect all train rollout file paths across hosts in sorted order
    rollout_paths = _collect_rollout_paths(manifest)
    logger.info("Found %d raw rollout files", len(rollout_paths))

    num_input_files = 0
    for rollout_path in rollout_paths:
        fs, _ = url_to_fs(rollout_path)
        with fs.open(rollout_path, "rb") as f:
            batch: RolloutBatch = pickle.load(f)

        num_input_files += 1

        for group in batch.groups:
            if not group.rollouts:
                continue

            # Verify policy version
            first_meta = group.rollouts[0].metadata
            if first_meta.policy_version >= 0 and first_meta.policy_version != policy_version:
                logger.warning(
                    "Skipping group with policy_version=%d (expected %d)",
                    first_meta.policy_version,
                    policy_version,
                )
                continue

            env_name = group.rollouts[0].env_name
            advantages = loss_module.compute_advantages(group.rollouts)

            records = []
            for rollout, advantage in zip(group.rollouts, advantages, strict=True):
                records.append(RolloutWithAdvantage(rollout=rollout, advantage=float(advantage)))

            # Spill to env-specific shard
            _spill_records(spill_dir, env_name, records, index)
            index.total_groups += 1
            index.total_individual += len(records)

        if num_input_files % 50 == 0:
            logger.info(
                "Ingested %d/%d files, %d groups so far", num_input_files, len(rollout_paths), index.total_groups
            )

    return index


def _collect_rollout_paths(manifest: SamplingManifest) -> list[str]:
    """Collect and sort all train/*.pkl paths across hosts."""
    paths = []
    for ha in manifest.host_assignments:
        host_dir = f"{manifest.output_root}/host_{ha.host_ordinal:03d}/train"
        fs, _ = url_to_fs(host_dir)
        if not fs.exists(host_dir):
            logger.warning("Host dir %s does not exist, skipping", host_dir)
            continue
        for entry in sorted(fs.ls(host_dir)):
            if isinstance(entry, dict):
                entry = entry["name"]
            if entry.endswith(".pkl"):
                paths.append(entry)
    paths.sort()
    return paths


def _spill_records(spill_dir: str, env_name: str, records: list[RolloutWithAdvantage], index: _SpillIndex) -> None:
    """Append records to a spill shard on local disk."""
    env_dir = os.path.join(spill_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)

    if env_name not in index.env_shards:
        index.env_shards[env_name] = []
        index.env_counts[env_name] = 0

    shard_idx = len(index.env_shards[env_name])
    shard_path = os.path.join(env_dir, f"shard_{shard_idx:04d}.pkl")
    with open(shard_path, "wb") as f:
        pickle.dump(records, f)

    index.env_shards[env_name].append(shard_path)
    index.env_counts[env_name] += len(records)


def _pass2_sample_and_pack(
    spill_index: _SpillIndex,
    manifest: SamplingManifest,
    config: MaterializerConfig,
    seed: int,
    pad_token_id: int,
) -> MaterializationManifest:
    """Pass 2: sample from spill files and pack into training batches."""
    rng = np.random.default_rng(seed)

    # Load all spill records into per-env lists with usage tracking
    env_rollouts: dict[str, list[tuple[RolloutWithAdvantage, int]]] = {}
    for env_name, shard_paths in spill_index.env_shards.items():
        env_rollouts[env_name] = []
        for shard_path in shard_paths:
            with open(shard_path, "rb") as f:
                records: list[RolloutWithAdvantage] = pickle.load(f)
            for record in records:
                env_rollouts[env_name].append((record, 0))  # (record, usage_count)

    fs, _ = url_to_fs(config.output_dir)
    fs.makedirs(config.output_dir, exist_ok=True)

    batch_paths = []
    total_batches = config.steps_per_phase

    for batch_idx in range(total_batches):
        sampled = _sample_balanced(env_rollouts, config.global_batch_size, config.alpha, config.max_samples, rng)
        if len(sampled) < config.global_batch_size:
            logger.warning(
                "Batch %d: only got %d samples (wanted %d). Using what we have.",
                batch_idx,
                len(sampled),
                config.global_batch_size,
            )
            if not sampled:
                raise RuntimeError(f"No samples available for batch {batch_idx}. Materializer cannot continue.")

        training_batch = create_training_batch_from_rollouts(
            sampled,
            max_tokens=config.max_seq_len,
            pad_token_id=pad_token_id,
            pad_to_multiple=config.pad_to_multiple,
        )

        batch_path = f"{config.output_dir}/batch_{batch_idx:06d}.pkl"
        with fs.open(batch_path, "wb") as f:
            pickle.dump(training_batch, f)
        batch_paths.append(batch_path)

        if (batch_idx + 1) % 10 == 0:
            logger.info("Packed %d/%d training batches", batch_idx + 1, total_batches)

    return MaterializationManifest(
        phase_id=manifest.phase_id,
        policy_version=manifest.policy_version,
        num_input_rollout_files=sum(len(v) for v in spill_index.env_shards.values()),
        num_rollout_groups=spill_index.total_groups,
        num_individual_rollouts=spill_index.total_individual,
        num_training_batches=len(batch_paths),
        global_batch_size=config.global_batch_size,
        max_seq_len=config.max_seq_len,
        batch_paths=batch_paths,
    )


def _sample_balanced(
    env_rollouts: dict[str, list[tuple[RolloutWithAdvantage, int]]],
    batch_size: int,
    alpha: float,
    max_samples: int,
    rng: np.random.Generator,
) -> list[RolloutWithAdvantage]:
    """Sample rollouts with balanced environment sampling and recency bias.

    Mirrors the sampling policy from ReplayBuffer.sample_rollouts().
    """
    # Filter to envs with available (under max_samples) rollouts
    available_envs = {}
    for env_name, rollouts in env_rollouts.items():
        available = [(i, r) for i, (r, count) in enumerate(rollouts) if count < max_samples]
        if available:
            available_envs[env_name] = available

    if not available_envs:
        return []

    # Build flat choice array for balanced env sampling
    env_choices = []
    for env_name, available in available_envs.items():
        env_choices.extend([env_name] * len(available))

    total_available = len(env_choices)
    actual_batch_size = min(batch_size, total_available)

    env_choices_arr = np.array(env_choices)
    selected_envs = rng.choice(env_choices_arr, size=actual_batch_size, replace=False)

    # Count per-env selections
    from collections import defaultdict

    env_count: dict[str, int] = defaultdict(int)
    for env_name in selected_envs:
        env_count[env_name] += 1

    sampled: list[RolloutWithAdvantage] = []
    for env_name, count in env_count.items():
        available = available_envs[env_name]
        indices_in_available = np.arange(len(available))
        weights = (indices_in_available + 1).astype(np.float64) ** alpha
        weights /= weights.sum()

        chosen = rng.choice(len(available), p=weights, size=min(count, len(available)), replace=False)
        for c in chosen:
            original_idx, record = available[c]
            sampled.append(record)
            # Increment usage count in-place
            env_rollouts[env_name][original_idx] = (record, env_rollouts[env_name][original_idx][1] + 1)

    return sampled
