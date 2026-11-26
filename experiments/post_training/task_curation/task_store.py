# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated sources and released task files bound into one versioned set of tasks.

A ``TaskStore`` records where each source's tasks live and what pins them; it copies no task data.
A curated source is pinned by its canonical ``manifest.json`` and a released file by its bytes.
``to_dataset`` reads every task as a Zephyr dataset and fails on a source whose pin changed.
"""

import hashlib
import io
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal

import pyarrow.parquet as pq
from marin.execution.artifact import Artifact, read_record
from marin.execution.lazy import ArtifactStep, StepContext
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.curation_catalog import MANIFEST_FILENAME

TASK_STORE_PREFIX = "rl-data/task-store"
TASK_ID_COLUMN = "task_id"
TASK_JSON_COLUMN = "task_json"
TASK_SPEC_COLUMN = "task_spec"
PARQUET_BATCH_ROWS = 256


@dataclass(frozen=True)
class TaskRow:
    """One task and the source it came from."""

    source_id: str
    task_id: str
    task_spec: str
    """The TaskSpec JSON as the source stores it."""

    @property
    def task(self) -> TaskSpec:
        return TaskSpec.model_validate_json(self.task_spec)


class CuratedSource(BaseModel):
    """A curation artifact; its tasks are the admitted rows under ``final/``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["curated"] = "curated"
    root: str
    manifest_sha256: str
    """SHA-256 of the artifact's canonical ``manifest.json``. The ``final/`` files are not hashed."""

    def task_files(self) -> tuple[str, ...]:
        manifest = read_manifest(self.root)
        if canonical_sha256(manifest) != self.manifest_sha256:
            raise ValueError(f"Curated source manifest changed after the task store pinned it: {self.root}")
        return admitted_task_files(self.root, manifest)

    def rows(self, source_id: str, path: str) -> Iterator[TaskRow]:
        for task_id, task_json in curated_rows(path):
            yield TaskRow(source_id, task_id, task_json)


class ReleasedFile(BaseModel):
    """One published Parquet file with a ``task_spec`` column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["released"] = "released"
    uri: str
    sha256: str
    """SHA-256 of the file's bytes."""

    def task_files(self) -> tuple[str, ...]:
        return (self.uri,)

    def rows(self, source_id: str, path: str) -> Iterator[TaskRow]:
        parquet = _released_parquet(path, self.sha256)
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=[TASK_SPEC_COLUMN]):
            for raw in batch.column(0).to_pylist():
                yield TaskRow(source_id, json.loads(raw)["id"], raw)


SourceEntry = Annotated[CuratedSource | ReleasedFile, Field(discriminator="kind")]


@dataclass(frozen=True)
class _TaskFile:
    source_id: str
    entry: CuratedSource | ReleasedFile
    path: str

    def rows(self) -> Iterator[TaskRow]:
        return self.entry.rows(self.source_id, self.path)


class TaskStore(Artifact):
    """A versioned set of sources whose tasks read as one dataset."""

    sources: dict[str, SourceEntry]

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
        """Every task of every source, one shard per task file, in source and file order.

        Curated manifests are checked here, released files when a worker reads them.
        """
        files = [
            _TaskFile(source_id, entry, path) for source_id, entry in self.sources.items() for path in entry.task_files()
        ]
        return Dataset.from_list(files).flat_map(_TaskFile.rows)


@dataclass(frozen=True)
class TaskStoreConfig:
    curated: dict[str, str]
    """Source ID to curation artifact root."""
    released: dict[str, str]
    """Released file URI to the SHA-256 of its bytes."""


def build_task_store(config: TaskStoreConfig) -> TaskStore:
    """Pin each curated source by its current manifest and each released file by its declared digest."""
    if overlap := sorted(config.curated.keys() & config.released.keys()):
        raise ValueError(f"Task store sources are both curated and released: {', '.join(overlap)}")
    sources: dict[str, CuratedSource | ReleasedFile] = {
        source_id: CuratedSource(root=root, manifest_sha256=manifest_sha256(root))
        for source_id, root in sorted(config.curated.items())
    }
    sources.update({uri: ReleasedFile(uri=uri, sha256=sha256) for uri, sha256 in config.released.items()})
    return TaskStore(sources=sources)


def task_store_step(
    name: str, version: str, *, curated: Mapping[str, ArtifactStep], released: Mapping[str, str]
) -> ArtifactStep[TaskStore]:
    """Bind curated source artifacts and released files, keyed by URI with their SHA-256, to a store."""

    def build_config(ctx: StepContext) -> TaskStoreConfig:
        return TaskStoreConfig(
            curated={source_id: ctx.artifact_path(step) for source_id, step in curated.items()},
            released=dict(released),
        )

    step = ArtifactStep(
        name=f"{TASK_STORE_PREFIX}/{name}",
        version=version,
        artifact_type=TaskStore,
        deps=tuple(curated.values()),
        run=build_task_store,
        build_config=build_config,
    )
    # A fixed version keeps the sources it was first built with: another set fails instead of
    # silently reusing the older record.
    return replace(step, expected_fingerprint=step.fingerprint())


def task_store_path(name: str, version: str) -> str:
    """Where ``task_store_step(name, version, ...)`` writes its record under the artifact prefix."""
    return prefix_join(marin_prefix(), f"{TASK_STORE_PREFIX}/{name}/{version}")


def read_manifest(root: str) -> dict[str, Any]:
    return json.loads((StoragePath(root) / MANIFEST_FILENAME).read_text())


def manifest_sha256(root: str) -> str:
    return canonical_sha256(read_manifest(root))


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


def _released_parquet(uri: str, sha256: str) -> pq.ParquetFile:
    """The released file at ``uri``, after checking its bytes against ``sha256``."""
    contents = StoragePath(uri).read_bytes()
    if hashlib.sha256(contents).hexdigest() != sha256:
        raise ValueError(f"Released source SHA-256 mismatch: {uri}")
    return pq.ParquetFile(io.BytesIO(contents))
