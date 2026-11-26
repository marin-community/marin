# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curation artifacts bound into one versioned set of tasks.

A ``TaskStore`` records each source's curation artifact and the SHA-256 of its canonical
``manifest.json``; it copies no task data. ``to_dataset`` reads every admitted task as a
Zephyr dataset and fails on a source whose manifest changed.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import pyarrow.parquet as pq
from finestore.rl.curation import CurationMode
from marin.execution.artifact import Artifact, read_record
from marin.execution.lazy import ArtifactStep, StepContext
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import IntendedUse
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.curation_catalog import MANIFEST_FILENAME

TASK_STORE_PREFIX = "rl-data/task-store"
TASK_ID_COLUMN = "task_id"
TASK_JSON_COLUMN = "task_json"
PARQUET_BATCH_ROWS = 256


@dataclass(frozen=True)
class TaskRow:
    """One admitted task and the source it came from."""

    source_id: str
    task_id: str
    task_json: str

    @property
    def task(self) -> TaskSpec:
        return TaskSpec.model_validate_json(self.task_json)


class CuratedSource(BaseModel):
    """A curation artifact; its tasks are the admitted rows that its manifest records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: str
    manifest_sha256: str
    """SHA-256 of the canonical ``manifest.json``; the admitted files themselves are not hashed."""
    intended_use: IntendedUse
    """Whether the source's tasks are for training or evaluation, as its manifest records."""

    def task_files(self) -> tuple[str, ...]:
        manifest = _read_manifest(self.root)
        if canonical_sha256(manifest) != self.manifest_sha256:
            raise ValueError(f"Curated source manifest changed after the task store pinned it: {self.root}")
        return admitted_task_files(self.root, manifest)


@dataclass(frozen=True)
class _TaskFile:
    source_id: str
    path: str

    def rows(self) -> Iterator[TaskRow]:
        for task_id, task_json in curated_rows(self.path):
            yield TaskRow(self.source_id, task_id, task_json)


class TaskStore(Artifact):
    """A versioned set of curation artifacts whose admitted tasks read as one dataset."""

    sources: dict[str, CuratedSource]

    @classmethod
    def open(cls, name: str, version: str) -> "TaskStore":
        """Load the store record ``name@version`` from the configured artifact prefix.

        Raises ``FileNotFoundError`` when the prefix holds no record for it.
        """
        path = task_store_path(name, version)
        if read_record(path) is None:
            raise FileNotFoundError(f"No task store {name}@{version} at {path}")
        return cls.raw_load(path)

    def select_sources(self, *source_ids: str) -> "TaskStore":
        """The store restricted to ``source_ids``, in that order."""
        missing = [source_id for source_id in source_ids if source_id not in self.sources]
        if missing:
            raise KeyError(f"Task store has no sources: {', '.join(missing)}")
        return TaskStore(sources={source_id: self.sources[source_id] for source_id in source_ids})

    def to_dataset(self) -> Dataset[TaskRow]:
        """Every admitted task, one shard per task file, in source and file order.

        The manifests are checked here, before any worker starts.
        """
        files = [
            _TaskFile(source_id, path) for source_id, source in self.sources.items() for path in source.task_files()
        ]
        return Dataset.from_list(files).flat_map(_TaskFile.rows)


@dataclass(frozen=True)
class TaskStoreConfig:
    sources: dict[str, str]
    """Source ID to curation artifact root."""


def build_task_store(config: TaskStoreConfig) -> TaskStore:
    """Pin each source by its current manifest; only SAMPLE and FULL curation admits tasks."""
    sources = {}
    for source_id, root in sorted(config.sources.items()):
        manifest = _read_manifest(root)
        if manifest["mode"] not in {CurationMode.SAMPLE, CurationMode.FULL}:
            raise ValueError(f"Task store sources need SAMPLE/FULL curation: {source_id} is {manifest['mode']}")
        sources[source_id] = CuratedSource(
            root=root, manifest_sha256=canonical_sha256(manifest), intended_use=manifest["intended_use"]
        )
    return TaskStore(sources=sources)


def task_store_step(name: str, version: str, *, sources: Mapping[str, ArtifactStep]) -> ArtifactStep[TaskStore]:
    """Bind curation artifacts, keyed by source ID, to the store ``name@version``."""

    def build_config(ctx: StepContext) -> TaskStoreConfig:
        return TaskStoreConfig({source_id: ctx.artifact_path(step) for source_id, step in sources.items()})

    step = ArtifactStep(
        name=f"{TASK_STORE_PREFIX}/{name}",
        version=version,
        artifact_type=TaskStore,
        deps=tuple(sources.values()),
        run=build_task_store,
        build_config=build_config,
    )
    # A fixed version keeps the sources it was first built with: another set fails instead of
    # silently reusing the older record.
    return replace(step, expected_fingerprint=step.fingerprint())


def task_store_path(name: str, version: str) -> str:
    """Where ``task_store_step(name, version, ...)`` writes its record under the artifact prefix."""
    return prefix_join(marin_prefix(), f"{TASK_STORE_PREFIX}/{name}/{version}")


def admitted_task_files(root: str, manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """The admitted task files that a curation artifact's manifest records, in name order.

    Recipe curation records its views under ``datasets`` and its counts under
    ``admission_counts``; a prebuilt import keeps ``final/`` under its root and counts
    ``admitted_rows``. Raises when the manifest counts admitted tasks but their files are
    gone, as happens when the prefix that held them expired.
    """
    datasets = manifest.get("datasets")
    final = datasets["final"] if datasets is not None else str(StoragePath(root) / "final")
    files = tuple(str(path) for path in sorted((StoragePath(final) / "*.parquet").glob(), key=str))
    counts = manifest.get("admission_counts")
    admitted = counts.get("admitted", 0) if counts is not None else manifest["admitted_rows"]
    if admitted and not files:
        raise ValueError(f"{root} admitted {admitted} tasks, but {final} holds no task files")
    return files


def curated_rows(path: str) -> Iterator[tuple[str, str]]:
    """The ``(task_id, task_json)`` rows of one admitted task file."""
    with StoragePath(path).open("rb") as stream:
        columns = [TASK_ID_COLUMN, TASK_JSON_COLUMN]
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=columns):
            yield from zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True)


def _read_manifest(root: str) -> dict[str, Any]:
    return json.loads((StoragePath(root) / MANIFEST_FILENAME).read_text())
