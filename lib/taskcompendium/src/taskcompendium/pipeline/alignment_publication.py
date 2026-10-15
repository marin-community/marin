# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write Atlas task Parquet after checking every selected task has one alignment row."""

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from finestore.rl.curriculum import AlignmentSource
from finestore.schema import arrow_schema
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from taskcompendium.pipeline.task_alignment import TaskAlignment

IDENTITY_FIELDS = ("source_id", "source_key", "task_id", "task_hash", "content_hash", "population", "intended_use")


@dataclass(frozen=True)
class PublicationData:
    paths: list[str]
    sources: list[AlignmentSource]
    unique_content_count: int
    unique_embedding_count: int


def _tagged_rows(item: tuple[str, str]) -> Iterator[dict]:
    path, kind = item
    for row in load_parquet(path):
        yield {"source_id": row["source_id"], "task_id": row["task_id"], "kind": kind, "row": json.dumps(row)}


def _covered_task(identifier: tuple[str, str], records: Iterator[dict]) -> dict:
    selected = {}
    for record in records:
        kind = record["kind"]
        if kind in selected:
            raise ValueError(f"Duplicate {kind} row for task {identifier}")
        selected[kind] = json.loads(record["row"])
    if set(selected) != {"input", "match"}:
        raise ValueError(f"Incomplete alignment coverage for task {identifier}: {sorted(selected)}")
    expected, actual = selected["input"], selected["match"]
    if any(expected[field] != actual[field] for field in IDENTITY_FIELDS):
        raise ValueError(f"Alignment lineage differs from input for task {identifier}")
    match = TaskAlignment.model_validate(actual)
    if (match.description_status == "ready") != bool(match.matches and match.matches.subjects):
        raise ValueError(f"Description and matching coverage disagree for task {identifier}")
    return match.model_dump(mode="json")


def _source_counts(source_id: str, records: Iterator[dict]) -> dict:
    counts = Counter()
    identity = None
    for row in records:
        current = (row["source_key"], row["population"], row["intended_use"])
        if identity is not None and identity != current:
            raise ValueError(f"Source {source_id} mixes input populations or manifests")
        identity = current
        counts[row["description_status"]] += 1
    assert identity is not None
    return {
        "source_id": source_id,
        "source_key": identity[0],
        "population": identity[1],
        "intended_use": identity[2],
        "task_count": counts.total(),
        "ready_count": counts["ready"],
        "insufficient_context_count": counts["insufficient_context"],
    }


def _count(source_id: str, records: Iterator[dict]) -> dict:
    return {"source_id": source_id, "count": sum(1 for _ in records)}


def publish_alignments(
    context: ZephyrContext,
    *,
    input_paths: Sequence[str],
    alignment_paths: Sequence[str],
    output_root: str,
) -> PublicationData:
    """Join inputs to scores by source/task ID, reject gaps or duplicates, and write Parquet.

    Returned paths and population counts form the Atlas release manifest.
    """
    tagged = [*((path, "input") for path in input_paths), *((path, "match") for path in alignment_paths)]
    result = context.execute(
        Dataset.from_list(tagged)
        .flat_map(_tagged_rows)
        .group_by(key=lambda row: (row["source_id"], row["task_id"]), reducer=_covered_task)
        .write_parquet(
            str(StoragePath(output_root) / "tasks/part-{shard:05d}.parquet"), schema=arrow_schema(TaskAlignment)
        ),
        name="publish-curriculum-matches",
    )
    data = Dataset.from_list(result.results).load_parquet()
    counts = context.execute(data.group_by(key=lambda row: row["source_id"], reducer=_source_counts))
    unique = data.deduplicate(key=lambda row: row["content_hash"])
    per_source = context.execute(
        data.deduplicate(key=lambda row: (row["source_id"], row["content_hash"])).group_by(
            key=lambda row: row["source_id"], reducer=_count
        )
    )
    unique_counts = {row["source_id"]: row["count"] for row in per_source.results}
    global_count = context.execute(unique.count()).results[0]
    embedding_count = context.execute(
        data.filter(lambda row: row["description_status"] == "ready")
        .deduplicate(key=lambda row: row["description"])
        .count()
    ).results[0]
    return PublicationData(
        result.results,
        [AlignmentSource(**row, unique_content_count=unique_counts[row["source_id"]]) for row in counts.results],
        global_count,
        embedding_count,
    )
