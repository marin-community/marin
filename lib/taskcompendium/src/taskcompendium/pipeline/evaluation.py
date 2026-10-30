# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish population-preserving evaluation corpora beside canonical task conversion rows."""

import hashlib
import json
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import Any, Literal

import pyarrow as pa
from pydantic import Field, field_validator, model_validator
from rigging.filesystem.path_validation import validate_relative_file_path
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet
from zephyr.writers import write_parquet_file

from taskcompendium.evaluation import (
    ChatGeneration,
    EvaluationItem,
    EvaluationModel,
    EvaluationProtocol,
    ResponseSelection,
    TextGeneration,
)
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import SourceFileOverride
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from taskcompendium.pipeline.source_processing import ConversionResult, SourceProcessingMode, run_source_pipeline

ITEM_SCHEMA = pa.schema([("task_id", pa.string()), ("evaluation_json", pa.string())])
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


def _canonical_task(
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


def _validate_item(item: EvaluationItem, protocol: EvaluationProtocol) -> None:
    if item.operation.kind != protocol.operation:
        raise ValueError(f"Evaluation operation differs from its protocol: {item.task_id}")
    if item.source_ordinal != source_order(item.source_row)[1]:
        raise ValueError(f"Evaluation ordinal differs from its source locator: {item.task_id}")
    policy, selection = protocol.context_policy, item.context_selection
    if (policy is None) != (selection is None) or (
        policy is not None and selection is not None and policy.kind != selection.kind
    ):
        raise ValueError(f"Evaluation context metadata differs from its policy: {item.task_id}")


def _validate_task_item(task: TaskSpec, item: EvaluationItem) -> None:
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


def _publish_shard(
    path: str,
    *,
    output_path: str,
    protocol: EvaluationProtocol,
    item_builder: ItemBuilder,
    source_dataset: str,
    source_revision: str,
    importer_revision: str,
) -> dict:
    source = StoragePath(path)
    destination = StoragePath(output_path) / "evaluation" / source.name
    count = 0
    first = last = ""

    def rows() -> Iterator[dict[str, str]]:
        nonlocal count, first, last
        for row in load_parquet(path):
            task = _canonical_task(
                row, source_dataset=source_dataset, source_revision=source_revision, importer_revision=importer_revision
            )
            item = item_builder(task)
            _validate_item(item, protocol)
            _validate_task_item(task, item)
            if (item.task_id, item.source_row) != (task.id, task.source.row):
                raise ValueError(f"Evaluation sidecar changed task provenance: {task.id}")
            if count and source_order(item.source_row) <= source_order(last):
                raise ValueError("Evaluation source rows must be in original source order")
            first = first or item.source_row
            last = item.source_row
            count += 1
            yield {"task_id": task.id, "evaluation_json": item.model_dump_json()}

    write_parquet_file(rows(), str(destination), schema=ITEM_SCHEMA)
    return EvaluationShard(
        tasks=CorpusFile(path="normalize/" + source.name, sha256=file_sha256(source)),
        items=CorpusFile(path="evaluation/" + source.name, sha256=file_sha256(destination)),
        rows=count,
        first_source_row=first,
        last_source_row=last,
    ).model_dump(mode="json")


def publish_evaluation_corpus(
    recipe: SourceRecipe,
    protocol: EvaluationProtocol,
    item_builder: ItemBuilder,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    *,
    source_overrides: Mapping[str, SourceFileOverride] | None = None,
) -> EvaluationManifest:
    """Convert all selected rows and publish a manifest only after population and sidecar checks pass.

    Conversion keeps duplicates at different source positions. Grader controls and old/new semantic
    comparison are independent checks; this manifest certifies conversion integrity, not grader parity.
    """
    if recipe.intended_use is not IntendedUse.EVAL:
        raise ValueError("An evaluation corpus requires an evaluation source recipe")
    output = StoragePath(output_path)
    if (output / MANIFEST_FILENAME).exists():
        raise FileExistsError(f"Evaluation corpus already published: {output_path}")
    # Without the release marker these are incomplete outputs from an earlier
    # conversion or sidecar attempt. Keep executor bookkeeping and staged inputs.
    for directory in ("normalize", "evaluation"):
        path = output / directory
        if path.exists():
            path.rmtree()
    for filename in ("manifest.json", PROTOCOL_FILENAME):
        path = output / filename
        if path.exists():
            path.rm()
    result = run_source_pipeline(
        recipe,
        context,
        source_input,
        output_path,
        mode=SourceProcessingMode.QUICK,
        canonical_source=recipe.name,
        source_overrides=source_overrides,
    )
    assert isinstance(result, ConversionResult)
    if (
        result.rejections
        or result.input_rows != protocol.expected_population
        or result.converted_rows != result.input_rows
    ):
        raise ValueError(
            f"Evaluation population mismatch: expected {protocol.expected_population}, "
            f"read {result.input_rows}, converted {result.converted_rows}, rejections={result.rejections}"
        )
    paths = sorted(str(path) for path in (StoragePath(result.normalized_path) / "*.parquet").glob())
    published = context.execute(
        Dataset.from_list(paths).map(
            partial(
                _publish_shard,
                output_path=output_path,
                protocol=protocol,
                item_builder=item_builder,
                source_dataset=recipe.source.dataset,
                source_revision=recipe.source.revision,
                importer_revision=recipe.version,
            )
        ),
        name="evaluation-sidecars",
    )
    shards = sorted(
        (shard for row in published.results if (shard := EvaluationShard.model_validate(row)).rows),
        key=lambda shard: source_order(shard.first_source_row),
    )
    manifest = EvaluationManifest(
        source_dataset=recipe.source.dataset,
        source_revision=recipe.source.revision,
        importer_revision=recipe.version,
        protocol_hash=protocol.fingerprint,
        conversion_manifest_sha256=file_sha256(output / "manifest.json"),
        rows=result.input_rows,
        shards=tuple(shards),
    )
    (output / PROTOCOL_FILENAME).write_text(protocol.model_dump_json(indent=2))
    (output / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
    return manifest


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
        _validate_item(item, protocol)
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
            task = _canonical_task(
                task_row,
                source_dataset=manifest.source_dataset,
                source_revision=manifest.source_revision,
                importer_revision=manifest.importer_revision,
            )
            if (task.id, task.source.row) != (item.task_id, item.source_row):
                raise ValueError("Evaluation sidecar does not match its canonical task")
            _validate_task_item(task, item)
            yield EvaluationCase(task, item)
