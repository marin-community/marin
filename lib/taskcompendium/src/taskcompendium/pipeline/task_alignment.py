# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score vector Parquet batches and join curriculum assignments to task instances."""

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import pyarrow as pa
from finestore.schema import arrow_schema
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.readers import load_parquet
from zephyr.worker_context import zephyr_worker_ctx

from taskcompendium.models import ConversationInput, format_conversation
from taskcompendium.pipeline.curriculum_alignment import (
    CurriculumVectors,
    TaskMatches,
    rank_task_batch,
)
from taskcompendium.pipeline.curriculum_models import EmbeddingSpec
from taskcompendium.pipeline.semantics import DescriptionStatus, TaskDescription
from taskcompendium.pipeline.task_embeddings import text_record

PREVIEW_CHARACTERS = 8192


class TaskAlignment(BaseModel):
    source_id: str
    source_key: str
    task_id: str
    task_hash: str
    content_hash: str
    intended_use: str
    population: str
    description_status: DescriptionStatus
    reason: str
    description: str
    task_preview: str
    instruction_characters: int
    preview_truncated: bool
    matches: TaskMatches | None


@dataclass(frozen=True)
class AlignmentSourceCounts:
    source_id: str
    source_key: str
    population: str
    intended_use: str
    task_count: int
    ready_count: int
    insufficient_context_count: int


def _source_counts(source_id: str, records: Iterator[dict]) -> AlignmentSourceCounts:
    row = next(records)
    counts = Counter([row["description_status"]])
    counts.update(record["description_status"] for record in records)
    return AlignmentSourceCounts(
        source_id=source_id,
        source_key=row["source_key"],
        population=row["population"],
        intended_use=row["intended_use"],
        task_count=counts.total(),
        ready_count=counts["ready"],
        insufficient_context_count=counts["insufficient_context"],
    )


def count_alignment_sources(context: ZephyrContext, *, paths: Sequence[str]) -> list[AlignmentSourceCounts]:
    """Count task outcomes without reading instruction or match columns."""
    result = context.execute(
        Dataset.from_list(list(paths))
        .load_parquet(columns=["source_id", "source_key", "population", "intended_use", "description_status"])
        .group_by(key=lambda row: row["source_id"], reducer=_source_counts, num_output_shards=context.max_workers),
        name="count-alignment-sources",
    )
    return result.results


@dataclass(frozen=True)
class _JoinInput:
    path: str
    kind: str


def _join_records(source: _JoinInput) -> Iterator[dict]:
    for row in load_parquet(source.path):
        yield {"content_hash": row["content_hash"], "kind": source.kind, "payload": json.dumps(row, ensure_ascii=False)}


def _described_tasks(content_hash: str, records: Iterator[dict]) -> Iterator[dict]:
    description = None
    for record in records:
        row = json.loads(record["payload"])
        if record["kind"] == "description":
            if description is not None:
                raise ValueError(f"Multiple descriptions for content hash {content_hash}")
            description = row
            continue
        if description is None:
            raise ValueError(f"No description for content hash {content_hash}")
        yield {**row, **description}


def _rank_batches(
    batches: Iterator[list[dict]],
    _shard: ShardInfo,
    *,
    shared_key: str,
) -> Iterator[dict]:
    curriculum: CurriculumVectors = zephyr_worker_ctx().get_shared(shared_key)
    for batch in batches:
        rankings = rank_task_batch(curriculum, np.asarray([row["embedding"] for row in batch], dtype=np.float32))
        for row, matches in zip(batch, rankings, strict=True):
            counters.current_stage().update_counter("alignment/scored_embeddings", 1)
            yield {"embedding_key": row["embedding_key"], "matches": matches.model_dump(mode="json")}


def _description_records(source: _JoinInput, *, spec: EmbeddingSpec) -> Iterator[dict]:
    for row in load_parquet(source.path):
        if source.kind == "alignment":
            key = row["embedding_key"]
        elif row["status"] == DescriptionStatus.READY:
            key = text_record(row["description"], spec)["embedding_key"]
        else:
            key = f"unavailable:{row['content_hash']}"
        yield {"embedding_key": key, "kind": source.kind, "payload": json.dumps(row, ensure_ascii=False)}


def _matched_descriptions(embedding_key: str, records: Iterator[dict]) -> Iterator[dict]:
    matches = None
    for record in records:
        row = json.loads(record["payload"])
        if record["kind"] == "alignment":
            if matches is not None:
                raise ValueError(f"Multiple alignments for embedding key {embedding_key}")
            matches = row["matches"]
            continue
        if row["status"] == DescriptionStatus.READY and matches is None:
            raise ValueError(f"Alignment is missing task vector {embedding_key}")
        yield {**row, "matches": matches}


def _task_alignment(row: dict) -> dict:
    public = json.loads(row["public_input"])
    instructions = format_conversation(ConversationInput.model_validate(public["context"]).events)
    result = TaskAlignment(
        **{
            name: row[name]
            for name in (
                "source_id",
                "source_key",
                "task_id",
                "task_hash",
                "content_hash",
                "intended_use",
                "population",
                "reason",
                "description",
                "matches",
            )
        },
        description_status=row["status"],
        task_preview=instructions[:PREVIEW_CHARACTERS],
        instruction_characters=len(instructions),
        preview_truncated=len(instructions) > PREVIEW_CHARACTERS,
    )
    counters.current_stage().update_counter("alignment/tasks", 1)
    counters.current_stage().update_counter(
        "alignment/subject_matches", len(result.matches.subjects) if result.matches else 0
    )
    return result.model_dump(mode="json")


def align_tasks(
    context: ZephyrContext,
    *,
    input_paths: Sequence[str],
    description_paths: Sequence[str],
    vector_paths: Sequence[str],
    curriculum: CurriculumVectors,
    curriculum_key: str,
    spec: EmbeddingSpec,
    output_root: str,
    batch_size: int = 256,
) -> list[str]:
    """Score each task vector once, then join scores to descriptions and task instances."""
    shared_key = f"curriculum-{curriculum_key}"
    context.put(shared_key, curriculum)
    matches_type = arrow_schema(TaskAlignment).field("matches").type
    rankings = context.execute(
        Dataset.from_list(list(vector_paths))
        .load_parquet()
        .window(batch_size)
        .map_shard(partial(_rank_batches, shared_key=shared_key))
        .write_parquet(
            str(StoragePath(output_root) / "rankings/part-{shard:05d}.parquet"),
            schema=pa.schema([("embedding_key", pa.string()), ("matches", matches_type)]),
        ),
        name="score-task-vectors",
    )
    description_inputs = [
        *(_JoinInput(path, "alignment") for path in rankings.results),
        *(_JoinInput(path, "description") for path in description_paths),
    ]
    descriptions = context.execute(
        Dataset.from_list(description_inputs)
        .flat_map(partial(_description_records, spec=spec))
        .group_by(
            key=lambda row: row["embedding_key"],
            sort_by=lambda row: row["kind"],
            reducer=_matched_descriptions,
            num_output_shards=context.max_workers,
        )
        .write_parquet(
            str(StoragePath(output_root) / "matched-descriptions/part-{shard:05d}.parquet"),
            schema=pa.schema([("content_hash", pa.string()), *arrow_schema(TaskDescription), ("matches", matches_type)]),
        ),
        name="join-description-scores",
    )
    inputs = [
        *(_JoinInput(path, "task") for path in input_paths),
        *(_JoinInput(path, "description") for path in descriptions.results),
    ]
    result = context.execute(
        Dataset.from_list(inputs)
        .flat_map(_join_records)
        .group_by(
            key=lambda row: row["content_hash"],
            sort_by=lambda row: row["kind"],
            reducer=_described_tasks,
            num_output_shards=context.max_workers,
        )
        .map(_task_alignment)
        .write_parquet(
            str(StoragePath(output_root) / "assignments/part-{shard:05d}.parquet"), schema=arrow_schema(TaskAlignment)
        ),
        name="task-curriculum-alignment",
    )
    return result.results
