# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable stage checkpoints for Zephyr pipelines.

A pipeline that runs for many hours loses all completed stages when one stage
fails. A stage checkpoint records the shard references and the counter totals
after each completed stage, so a later execution starts at the first incomplete
stage.

The checkpoint identity comes from a caller-supplied key. The key selects a
stable execution directory, thus the intermediate data of a failed execution
stays readable. The physical plan fingerprint guards against a resume across
different pipelines.
"""

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cloudpickle
from rigging.filesystem import StoragePath, atomic_rename

from zephyr.plan import PhysicalPlan, Shard
from zephyr.shuffle import ListShard, MemChunk, _scatter_meta_path
from zephyr.stage_io import PickleDiskChunk
from zephyr.worker_context import Aggregation, CounterEntry
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)

STAGE_CHECKPOINT_VERSION = 1
# Bump when the on-disk layout of the referenced intermediate data changes.
_STAGE_CHECKPOINT_PLAN_VERSION = "zephyr-stage-checkpoint-v1-parquet-scatter-v1"


@dataclass(frozen=True)
class ZephyrStageCheckpoint:
    """Stable identity for resume from completed Zephyr stages.

    Zephyr keeps the execution directory after a failed execution when this
    value is present. A later execution with the same key loads the newest
    valid stage manifest. A successful execution deletes the directory.

    Attributes:
        key: Stable identity for one logical pipeline. Use a different key for
            each ``execute()`` call in a multi-pipeline job.
    """

    key: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("ZephyrStageCheckpoint.key must not be empty")


@dataclass(frozen=True)
class StageCheckpointManifest:
    version: int
    plan_fingerprint: str
    stage_index: int
    shards: list[Shard]
    completed_totals: dict[tuple[str | None, str, Aggregation], CounterEntry]


def checkpoint_execution_id(checkpoint: ZephyrStageCheckpoint) -> str:
    digest = hashlib.sha256(checkpoint.key.encode()).hexdigest()[:24]
    return f"checkpoint-{digest}"


def plan_fingerprint(plan: PhysicalPlan) -> str:
    digest = hashlib.sha256()
    digest.update(_STAGE_CHECKPOINT_PLAN_VERSION.encode())
    digest.update(cloudpickle.dumps(plan))
    return digest.hexdigest()


def _stage_checkpoint_path(prefix: str, execution_id: str, stage_index: int) -> str:
    return f"{prefix}/{execution_id}/checkpoints/stage-{stage_index:04d}.pkl"


def _checkpoint_reference_paths(shards: list[Shard], prefix: str, execution_id: str) -> list[str]:
    """Return commit-marker paths that make a stage checkpoint readable."""
    execution_dir = f"{prefix}/{execution_id}/"
    paths: set[str] = set()
    seen_refs: set[int] = set()
    for shard in shards:
        if not isinstance(shard, ListShard):
            raise TypeError(f"Stage checkpoint does not support shard type {type(shard).__name__}")
        for ref in shard.refs:
            ref_id = id(ref)
            if ref_id in seen_refs:
                continue
            seen_refs.add(ref_id)
            if isinstance(ref, PickleDiskChunk):
                paths.add(ref.path)
                continue
            if isinstance(ref, MemChunk):
                for item in ref.items:
                    if isinstance(item, str) and item.startswith(execution_dir) and item.endswith("/"):
                        paths.add(_scatter_meta_path(item))
    return sorted(paths)


def _missing_checkpoint_references(paths: list[str]) -> list[str]:
    if not paths:
        return []

    def exists(path: str) -> bool:
        return StoragePath(path).exists()

    with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
        present = pool.map(exists, paths)
        return [path for path, path_exists in zip(paths, present, strict=True) if not path_exists]


def write_stage_checkpoint(
    *,
    prefix: str,
    execution_id: str,
    fingerprint: str,
    stage_index: int,
    shards: list[Shard],
    completed_totals: dict[tuple[str | None, str, Aggregation], CounterEntry],
) -> None:
    manifest = StageCheckpointManifest(
        version=STAGE_CHECKPOINT_VERSION,
        plan_fingerprint=fingerprint,
        stage_index=stage_index,
        shards=shards,
        completed_totals=completed_totals,
    )
    path = _stage_checkpoint_path(prefix, execution_id, stage_index)
    ensure_parent_dir(path)
    payload = cloudpickle.dumps(manifest)
    with atomic_rename(path) as temp_path:
        StoragePath(temp_path).write_bytes(payload)
    logger.info("Saved stage checkpoint %d to %s (%d bytes)", stage_index, path, len(payload))


def load_stage_checkpoint(
    *,
    prefix: str,
    execution_id: str,
    fingerprint: str,
    num_stages: int,
) -> StageCheckpointManifest | None:
    """Return the newest readable manifest, or None when no checkpoint exists.

    Raises:
        ValueError: If every stored manifest is unusable, or one manifest
            belongs to a different plan or checkpoint version.
    """
    pattern = StoragePath(f"{prefix}/{execution_id}/checkpoints/stage-*.pkl")
    paths = sorted((str(path) for path in pattern.glob()), reverse=True)
    if not paths:
        return None

    errors: list[str] = []
    for path in paths:
        try:
            manifest = cloudpickle.loads(StoragePath(path).read_bytes())
        except Exception as e:
            logger.warning("Cannot read stage checkpoint %s: %s", path, e)
            errors.append(f"{path}: {e}")
            continue

        if not isinstance(manifest, StageCheckpointManifest):
            errors.append(f"{path}: unknown manifest type {type(manifest).__name__}")
            continue
        if manifest.version != STAGE_CHECKPOINT_VERSION:
            raise ValueError(
                f"Stage checkpoint {path} has version {manifest.version}; expected {STAGE_CHECKPOINT_VERSION}"
            )
        if manifest.plan_fingerprint != fingerprint:
            raise ValueError(f"Stage checkpoint {path} does not match the current physical plan")
        if not 0 <= manifest.stage_index < num_stages:
            errors.append(f"{path}: stage index {manifest.stage_index} is outside the current plan")
            continue

        references = _checkpoint_reference_paths(manifest.shards, prefix, execution_id)
        missing = _missing_checkpoint_references(references)
        if missing:
            logger.warning("Stage checkpoint %s has %d missing references", path, len(missing))
            errors.append(f"{path}: {len(missing)} missing references")
            continue

        logger.info(
            "Loaded stage checkpoint %d from %s (%d references validated)",
            manifest.stage_index,
            path,
            len(references),
        )
        return manifest

    details = "; ".join(errors)
    raise ValueError(f"No valid stage checkpoint exists for {execution_id}: {details}")
