# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read and verify immutable evaluation task corpora."""

import hashlib
import json
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from rigging.filesystem.path_validation import validate_relative_file_path
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_parquet

from taskcompendium.evaluation.spec import (
    ChatGeneration,
    EvaluationItem,
    EvaluationModel,
    EvaluationProtocol,
    ResponseSelection,
    TextGeneration,
)
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage

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


def source_order(locator: str) -> tuple[str, int]:
    file, index = locator.rsplit(":", 1)
    return file, int(index)


def chat_item(task: TaskSpec, *, response_selection: ResponseSelection = "combine") -> EvaluationItem:
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        source_ordinal=source_order(task.source.row)[1],
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
    return task


def validate_item(item: EvaluationItem, protocol: EvaluationProtocol) -> None:
    if item.operation.kind != protocol.operation:
        raise ValueError(f"Evaluation operation differs from its protocol: {item.task_id}")
    if item.source_ordinal != source_order(item.source_row)[1]:
        raise ValueError(f"Evaluation ordinal differs from its source locator: {item.task_id}")
    policy, selection = protocol.context_policy, item.context_selection
    if (policy is None) != (selection is None) or (
        policy is not None and selection is not None and policy.kind != selection.kind
    ):
        raise ValueError(f"Evaluation context metadata differs from its policy: {item.task_id}")


def validate_task_item(task: TaskSpec, item: EvaluationItem) -> None:
    if not isinstance(item.operation, TextGeneration):
        return
    context = task.context
    if (
        not isinstance(context, ConversationInput)
        or len(context.events) != 1
        or not isinstance(context.events[0], TextMessage)
        or context.events[0].role != "user"
        or context.events[0].content != item.operation.context
    ):
        raise ValueError(f"Raw evaluation prompt differs from its canonical task: {task.id}")


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
    for row in load_parquet(str(path / shard.items.path)):
        item = EvaluationItem.model_validate_json(row["evaluation_json"])
        validate_item(item, protocol)
        if item.task_id != row["task_id"]:
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


def evaluation_cases(root: str, *, source_rows: Collection[str] | None = None) -> Iterator[EvaluationCase]:
    """Verify complete shard joins while decoding task payloads only for the requested source rows."""
    protocol, manifest = evaluation_protocol(root)
    path = StoragePath(root)
    selected = frozenset(source_rows) if source_rows is not None else None
    for shard in sorted(manifest.shards, key=lambda shard: source_order(shard.first_source_row)):
        if file_sha256(path / shard.tasks.path) != shard.tasks.sha256:
            raise ValueError(f"Evaluation corpus file changed after publication: {shard.tasks.path}")
        for task_row, item in zip(
            load_parquet(str(path / shard.tasks.path)), _shard_items(path, shard, protocol), strict=True
        ):
            if (
                task_row["task_id"],
                task_row["source_row"],
                task_row["source_dataset"],
                task_row["source_revision"],
            ) != (item.task_id, item.source_row, manifest.source_dataset, manifest.source_revision):
                raise ValueError("Evaluation sidecar does not match its canonical row")
            if selected is not None and item.source_row not in selected:
                continue
            task = canonical_task(
                task_row,
                source_dataset=manifest.source_dataset,
                source_revision=manifest.source_revision,
                importer_revision=manifest.importer_revision,
            )
            if (task.id, task.source.row) != (item.task_id, item.source_row):
                raise ValueError("Evaluation sidecar does not match its canonical task")
            validate_task_item(task, item)
            yield EvaluationCase(task, item)
