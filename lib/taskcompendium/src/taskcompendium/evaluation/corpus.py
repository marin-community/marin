# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and verify immutable evaluation task corpora."""

import hashlib
import json
from bisect import bisect_left
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import pyarrow.parquet as pq
from pydantic import Field, field_validator, model_validator
from rigging.filesystem.path_validation import validate_relative_file_path
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.evaluation.spec import (
    ChatGeneration,
    EvaluationItem,
    EvaluationModel,
    EvaluationProtocol,
    ResponseSelection,
    TextGeneration,
)
from taskcompendium.models import TaskSpec, TextMessage

MANIFEST_FILENAME = "evaluation_manifest.json"
PROTOCOL_FILENAME = "protocol.json"
type ItemBuilder = Callable[[TaskSpec], EvaluationItem]


class CorpusFile(EvaluationModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def check_path(cls, value: str) -> str:
        validate_relative_file_path(value)
        return value


class EvaluationShard(EvaluationModel):
    tasks: CorpusFile
    items: CorpusFile
    rows: int = Field(ge=0)
    first_source_row: str
    last_source_row: str


class EvaluationManifest(EvaluationModel):
    schema_version: Literal[1] = 1
    source_dataset: str
    source_revision: str
    importer_revision: str
    intended_use: Literal["eval"] = "eval"
    protocol_hash: str
    conversion_manifest_sha256: str
    rows: int = Field(gt=0)
    shards: tuple[EvaluationShard, ...]

    @model_validator(mode="after")
    def check_shards(self) -> "EvaluationManifest":
        if sum(shard.rows for shard in self.shards) != self.rows or any(shard.rows == 0 for shard in self.shards):
            raise ValueError("Evaluation manifest shards do not cover its population")
        ordered = sorted(self.shards, key=lambda shard: source_order(shard.first_source_row))
        for shard in ordered:
            if source_order(shard.first_source_row) > source_order(shard.last_source_row):
                raise ValueError("Evaluation shard source bounds are reversed")
        for previous, current in pairwise(ordered):
            if source_order(previous.last_source_row) >= source_order(current.first_source_row):
                raise ValueError("Evaluation source shards overlap or repeat source rows")
        return self


@dataclass(frozen=True)
class EvaluationCase:
    task: TaskSpec
    item: EvaluationItem


@dataclass(frozen=True)
class TaskRow:
    """Queryable task annotations and its location in an immutable corpus."""

    task_id: str
    source_row: str
    tags: tuple[str, ...]
    labels: dict[str, str]
    shard: int
    position: int


TASK_INDEX_COLUMNS = ("task_id", "source_dataset", "source_revision", "source_row", "tags", "labels")
ITEM_INDEX_COLUMNS = ("task_id", "source_row")


def parquet_rows(
    path: StoragePath, *, columns: tuple[str, ...] | None = None, positions: Collection[int] | None = None
) -> Iterator[dict[str, Any]]:
    """Read bounded batches, skipping row groups without requested row positions."""
    selected = sorted(positions) if positions is not None else None
    payload = columns is None or bool({"task_json", "evaluation_json"}.intersection(columns))
    batch_size = 1 if payload else 4096
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        start = 0
        for group in range(parquet.num_row_groups):
            end = start + parquet.metadata.row_group(group).num_rows
            wanted = None
            if selected is not None:
                wanted = frozenset(selected[bisect_left(selected, start) : bisect_left(selected, end)])
                if not wanted:
                    start = end
                    continue
            offset = start
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, row_groups=[group]):
                if wanted is None:
                    yield from batch.to_pylist()
                else:
                    for index in range(len(batch)):
                        if offset + index in wanted:
                            yield batch.slice(index, 1).to_pylist()[0]
                offset += len(batch)
            start = end


def source_order(locator: str) -> tuple[str, int]:
    file, index = locator.rsplit(":", 1)
    ordinal = int(index)
    if not file or ordinal < 0:
        raise ValueError(f"Invalid source locator: {locator}")
    return file, ordinal


def chat_item(task: TaskSpec, *, response_selection: ResponseSelection = "combine") -> EvaluationItem:
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        operation=ChatGeneration(response_selection=response_selection),
    )


def file_sha256(path: StoragePath) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_task(
    row: Mapping[str, Any], *, source_dataset: str, source_revision: str, importer_revision: str
) -> TaskSpec:
    if row["task_json"] is None or row["normalization_changes"]:
        raise ValueError(f"Evaluation release contains a rejected or rewritten row: {row['source_row']}")
    task = TaskSpec.model_validate_json(row["task_json"])
    if (task.id, task.source.dataset, task.source.revision, task.source.row) != (
        row["task_id"],
        row["source_dataset"],
        row["source_revision"],
        row["source_row"],
    ):
        raise ValueError("Evaluation canonical row changed its task identity or provenance")
    if (task.source.dataset, task.source.revision, task.source.importer_revision) != (
        source_dataset,
        source_revision,
        importer_revision,
    ):
        raise ValueError("Evaluation task provenance differs from its corpus")
    if "eval" not in task.tags:
        raise ValueError(f"Evaluation task lacks its eval tag: {task.id}")
    if tuple(row["tags"]) != task.tags or dict(row["labels"]) != task.labels:
        raise ValueError(f"Evaluation annotations differ from its canonical task: {task.id}")
    return task


def validate_item(item: EvaluationItem, protocol: EvaluationProtocol) -> None:
    if item.operation.kind != protocol.operation:
        raise ValueError(f"Evaluation operation differs from its protocol: {item.task_id}")
    source_order(item.source_row)
    policy = protocol.context_policy
    if policy is not None and policy.output_budget is not None and item.reference_text is None:
        raise ValueError(f"Evaluation output budget requires reference text: {item.task_id}")
    if policy is not None and policy.prefix_probe is not None and item.prompt_characters is None:
        raise ValueError(f"Evaluation prefix probe requires a source character count: {item.task_id}")


def text_prompt(task: TaskSpec) -> str:
    """Return the canonical raw prompt, requiring exactly one user text message."""
    events = task.context.events
    if len(events) != 1 or not isinstance(events[0], TextMessage) or events[0].role != "user":
        raise ValueError(f"Raw evaluation requires one canonical user text message: {task.id}")
    return events[0].content


def validate_task_item(task: TaskSpec, item: EvaluationItem) -> None:
    if not isinstance(item.operation, TextGeneration):
        return
    text_prompt(task)


def evaluation_protocol(root: str) -> tuple[EvaluationProtocol, EvaluationManifest]:
    """Read the protocol bound to a completed immutable corpus."""
    path = StoragePath(root)
    protocol = EvaluationProtocol.model_validate_json((path / PROTOCOL_FILENAME).read_text())
    manifest = EvaluationManifest.model_validate_json((path / MANIFEST_FILENAME).read_text())
    if protocol.fingerprint != manifest.protocol_hash or manifest.rows != protocol.expected_population:
        raise ValueError("Evaluation manifest does not match its protocol")
    conversion_bytes = (path / "manifest.json").read_bytes()
    if hashlib.sha256(conversion_bytes).hexdigest() != manifest.conversion_manifest_sha256:
        raise ValueError("Evaluation conversion provenance changed after publication")
    conversion = json.loads(conversion_bytes)
    if (
        conversion["source_dataset"] != manifest.source_dataset
        or conversion["source_revision"] != manifest.source_revision
        or conversion["recipe_revision"] != manifest.importer_revision
        or conversion["intended_use"] != "eval"
        or conversion["input_rows"] != manifest.rows
        or conversion["converted_rows"] != manifest.rows
        or conversion["rejections"]
    ):
        raise ValueError("Evaluation manifest differs from its conversion provenance")
    return protocol, manifest


def _shard_items(path: StoragePath, shard: EvaluationShard, protocol: EvaluationProtocol) -> Iterator[EvaluationItem]:
    if file_sha256(path / shard.items.path) != shard.items.sha256:
        raise ValueError(f"Evaluation corpus file changed after publication: {shard.items.path}")
    previous: tuple[str, int] | None = None
    count = 0
    first = last = ""
    for row in parquet_rows(path / shard.items.path):
        item = EvaluationItem.model_validate_json(row["evaluation_json"])
        validate_item(item, protocol)
        if (item.task_id, item.source_row) != (row["task_id"], row["source_row"]):
            raise ValueError("Evaluation sidecar row changed its task identity")
        order = source_order(item.source_row)
        if previous is not None and order <= previous:
            raise ValueError("Evaluation corpus is not in source order")
        previous = order
        first = first or item.source_row
        last = item.source_row
        count += 1
        yield item
    if count != shard.rows:
        raise ValueError("Evaluation shard population differs from its manifest")
    if (first, last) != (shard.first_source_row, shard.last_source_row):
        raise ValueError("Evaluation shard source bounds differ from its manifest")


def evaluation_items(root: str) -> Iterator[EvaluationItem]:
    """Stream verified sidecars for eligibility scans without reading canonical task payloads."""
    protocol, manifest = evaluation_protocol(root)
    path = StoragePath(root)
    for shard in sorted(manifest.shards, key=lambda shard: source_order(shard.first_source_row)):
        yield from _shard_items(path, shard, protocol)


class EvaluationCorpus:
    """One immutable read session, retaining small annotations and verified file identities."""

    def __init__(self, root: str):
        self.path = StoragePath(root)
        self.protocol, self.manifest = evaluation_protocol(root)
        self.shards = sorted(self.manifest.shards, key=lambda shard: source_order(shard.first_source_row))
        self._rows: tuple[TaskRow, ...] | None = None

    def rows(self) -> tuple[TaskRow, ...]:
        """Verify the complete identity join without reading either JSON payload column."""
        if self._rows is not None:
            return self._rows
        rows = []
        for index, shard in enumerate(self.shards):
            for file in (shard.tasks, shard.items):
                if file_sha256(self.path / file.path) != file.sha256:
                    raise ValueError(f"Evaluation corpus file changed after publication: {file.path}")
            previous = None
            first = last = ""
            count = 0
            for position, (task, item) in enumerate(
                zip(
                    parquet_rows(self.path / shard.tasks.path, columns=TASK_INDEX_COLUMNS),
                    parquet_rows(self.path / shard.items.path, columns=ITEM_INDEX_COLUMNS),
                    strict=True,
                )
            ):
                if (task["task_id"], task["source_row"], task["source_dataset"], task["source_revision"]) != (
                    item["task_id"],
                    item["source_row"],
                    self.manifest.source_dataset,
                    self.manifest.source_revision,
                ):
                    raise ValueError("Evaluation sidecar does not match its canonical row")
                order = source_order(task["source_row"])
                if previous is not None and order <= previous:
                    raise ValueError("Evaluation corpus is not in source order")
                previous = order
                first = first or task["source_row"]
                last = task["source_row"]
                count += 1
                rows.append(
                    TaskRow(
                        task["task_id"], task["source_row"], tuple(task["tags"]), dict(task["labels"]), index, position
                    )
                )
            if count != shard.rows or (first, last) != (shard.first_source_row, shard.last_source_row):
                raise ValueError("Evaluation shard population or source bounds differ from its manifest")
        self._rows = tuple(rows)
        return self._rows

    def cases(self, *, source_rows: Collection[str] | None = None) -> Iterator[EvaluationCase]:
        """Decode bounded selected payloads after verifying their complete projected identity join."""
        selected = frozenset(source_rows) if source_rows is not None else None
        rows = self.rows()
        for index, shard in enumerate(self.shards):
            requested = [row for row in rows if row.shard == index and (selected is None or row.source_row in selected)]
            if not requested:
                continue
            positions = [row.position for row in requested]
            for projected, task_row, item_row in zip(
                requested,
                parquet_rows(self.path / shard.tasks.path, positions=positions),
                parquet_rows(self.path / shard.items.path, positions=positions),
                strict=True,
            ):
                task = canonical_task(
                    task_row,
                    source_dataset=self.manifest.source_dataset,
                    source_revision=self.manifest.source_revision,
                    importer_revision=self.manifest.importer_revision,
                )
                item = EvaluationItem.model_validate_json(item_row["evaluation_json"])
                validate_item(item, self.protocol)
                if (task.id, task.source.row) != (item.task_id, item.source_row) or (
                    task.id,
                    task.source.row,
                    task.tags,
                    task.labels,
                ) != (projected.task_id, projected.source_row, projected.tags, projected.labels):
                    raise ValueError("Evaluation sidecar does not match its canonical task")
                validate_task_item(task, item)
                yield EvaluationCase(task, item)


def task_rows(root: str, *, tags: Collection[str] = (), labels: Mapping[str, str] | None = None) -> Iterator[TaskRow]:
    """Query projected task annotations without decoding canonical tasks or evaluation inputs."""
    for row in EvaluationCorpus(root).rows():
        if all(tag in row.tags for tag in tags) and all(
            row.labels.get(key) == value for key, value in (labels or {}).items()
        ):
            yield row


def evaluation_cases(root: str, *, source_rows: Collection[str] | None = None) -> Iterator[EvaluationCase]:
    """Verify corpus identities and decode only requested canonical tasks and evaluation inputs."""
    yield from EvaluationCorpus(root).cases(source_rows=source_rows)
