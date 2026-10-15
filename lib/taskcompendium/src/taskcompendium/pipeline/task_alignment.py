# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join task descriptions and cached vectors into streaming curriculum assignments."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
from finestore.reader import ReadView
from finestore.rl.curriculum import EmbeddingSpec
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
from taskcompendium.pipeline.semantics import DescriptionStatus
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


def _align_batches(
    batches: Iterator[list[dict]],
    _shard: ShardInfo,
    *,
    shared_key: str,
    cache_root: str,
    spec: EmbeddingSpec,
) -> Iterator[dict]:
    curriculum: CurriculumVectors = zephyr_worker_ctx().get_shared(shared_key)
    view = ReadView(cache_root)
    for batch in batches:
        ready = [row for row in batch if row["status"] == DescriptionStatus.READY]
        requests = [text_record(row["description"], spec)["embedding_key"] for row in ready]
        cached = view.read_blobs(requests)
        missing = set(requests) - set(cached)
        if missing:
            raise ValueError(f"Alignment is missing {len(missing)} cached task vectors")
        rankings = iter(
            rank_task_batch(
                curriculum,
                np.stack([np.frombuffer(cached[key], dtype="<f4") for key in requests]),
            )
            if ready
            else []
        )
        for row in batch:
            assigned = next(rankings) if row["status"] == DescriptionStatus.READY else None
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
                    )
                },
                description_status=row["status"],
                description=row["description"],
                task_preview=instructions[:PREVIEW_CHARACTERS],
                instruction_characters=len(instructions),
                preview_truncated=len(instructions) > PREVIEW_CHARACTERS,
                matches=assigned,
            )
            counters.current_stage().update_counter("alignment/tasks", 1)
            counters.current_stage().update_counter(
                "alignment/subject_matches", len(assigned.subjects) if assigned else 0
            )
            yield result.model_dump(mode="json")


def align_tasks(
    context: ZephyrContext,
    *,
    input_paths: Sequence[str],
    description_paths: Sequence[str],
    curriculum: CurriculumVectors,
    curriculum_key: str,
    cache_root: str,
    spec: EmbeddingSpec,
    output_root: str,
    batch_size: int = 256,
) -> list[str]:
    """Preserve all task instances while joining descriptions by public-input hash."""
    shared_key = f"curriculum-{curriculum_key}"
    context.put(shared_key, curriculum)
    inputs = [
        *(_JoinInput(path, "task") for path in input_paths),
        *(_JoinInput(path, "description") for path in description_paths),
    ]
    result = context.execute(
        Dataset.from_list(inputs)
        .flat_map(_join_records)
        .group_by(key=lambda row: row["content_hash"], sort_by=lambda row: row["kind"], reducer=_described_tasks)
        .window(batch_size)
        .map_shard(
            partial(
                _align_batches,
                shared_key=shared_key,
                cache_root=cache_root,
                spec=spec,
            )
        )
        .write_parquet(
            str(StoragePath(output_root) / "assignments/part-{shard:05d}.parquet"), schema=arrow_schema(TaskAlignment)
        ),
        name="task-curriculum-alignment",
    )
    return result.results
