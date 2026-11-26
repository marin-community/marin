# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated sources and released task files bound into one versioned set of tasks.

A ``TaskStore`` records where each source's tasks live and the SHA-256 that pins them; it copies
no task data. ``to_dataset`` reads every task as a Zephyr dataset and fails on a source that
changed after the store pinned it.
"""

import hashlib
import io
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from zephyr.dataset import Dataset

from experiments.post_training.task_curation.curation_catalog import MANIFEST_FILENAME

TASK_STORE_PREFIX = "rl-data/task-store"
CURATED_TASK_FILES = "final/*.parquet"
CURATED_COLUMNS = ("task_id", "task_json")
RELEASED_COLUMN = "task_spec"
PARQUET_BATCH_ROWS = 256


class SourceKind(StrEnum):
    CURATED = "curated"
    """A curation artifact; its tasks are the admitted rows under ``final/``."""
    RELEASED = "released"
    """One published Parquet file with a ``task_spec`` column."""


class SourceEntry(BaseModel):
    """Where one source's tasks live and the digest that pins them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SourceKind
    path: str
    """The curation artifact root, or the released file."""
    sha256: str
    """SHA-256 of the canonical ``manifest.json`` of a curated source, or of a released file's bytes."""


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


class TaskStore(Artifact):
    """A versioned set of sources whose tasks read as one dataset."""

    sources: dict[str, SourceEntry]

    def select_sources(self, *source_ids: str) -> "TaskStore":
        """The store restricted to ``source_ids``, in that order."""
        missing = [source_id for source_id in source_ids if source_id not in self.sources]
        if missing:
            raise KeyError(f"Task store has no sources: {', '.join(missing)}")
        return TaskStore(sources={source_id: self.sources[source_id] for source_id in source_ids})

    def to_dataset(self) -> Dataset[TaskRow]:
        """Every task of every source, one shard per task file, in source and file order.

        Curated sources are checked against their manifest pins here; released files are checked
        when a worker reads them.
        """
        files = [(source_id, entry, path) for source_id, entry in self.sources.items() for path in _task_files(entry)]
        return Dataset.from_list(files).flat_map(lambda item: _file_rows(*item))


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
    sources = {
        source_id: SourceEntry(kind=SourceKind.CURATED, path=root, sha256=_manifest_sha256(root))
        for source_id, root in sorted(config.curated.items())
    }
    sources.update(
        {uri: SourceEntry(kind=SourceKind.RELEASED, path=uri, sha256=sha256) for uri, sha256 in config.released.items()}
    )
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

    return ArtifactStep(
        name=f"{TASK_STORE_PREFIX}/{name}",
        version=version,
        artifact_type=TaskStore,
        deps=tuple(curated.values()),
        run=build_task_store,
        build_config=build_config,
    )


def _manifest_sha256(root: str) -> str:
    return canonical_sha256(json.loads((StoragePath(root) / MANIFEST_FILENAME).read_text()))


def _task_files(entry: SourceEntry) -> tuple[str, ...]:
    if entry.kind == SourceKind.RELEASED:
        return (entry.path,)
    if _manifest_sha256(entry.path) != entry.sha256:
        raise ValueError(f"Curated source changed after the task store pinned it: {entry.path}")
    return tuple(str(path) for path in sorted((StoragePath(entry.path) / CURATED_TASK_FILES).glob(), key=str))


def _file_rows(source_id: str, entry: SourceEntry, path: str) -> Iterator[TaskRow]:
    if entry.kind == SourceKind.CURATED:
        with StoragePath(path).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=CURATED_COLUMNS):
                for row in batch.to_pylist():
                    yield TaskRow(source_id, row["task_id"], row["task_json"])
        return
    contents = StoragePath(path).read_bytes()
    if hashlib.sha256(contents).hexdigest() != entry.sha256:
        raise ValueError(f"Released file changed after the task store pinned it: {path}")
    parquet = pq.ParquetFile(io.BytesIO(contents))
    for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=[RELEASED_COLUMN]):
        for raw in batch.column(0).to_pylist():
            yield TaskRow(source_id, json.loads(raw)["id"], raw)
