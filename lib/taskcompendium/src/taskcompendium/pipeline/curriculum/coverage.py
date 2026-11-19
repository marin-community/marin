# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Precompute Atlas statistics without scanning task rows while serving views."""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial

from finestore.schema import arrow_schema
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from taskcompendium.pipeline.curriculum.models import (
    CurriculumConfidenceStatistic,
    CurriculumNode,
    CurriculumStatistic,
)
from taskcompendium.pipeline.curriculum.task_alignment import TaskAlignment
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

# Approximate nearest-node coverage; retained scores remain available for review.
SUBJECT_DISTANCE = 0.70
CAPABILITY_DISTANCE = 0.70
CONFIDENCE_LEVELS = (
    ("All (nearest)", 0.30, 0.30),
    ("Broad", 0.55, 0.60),
    ("Moderate", 0.55, 0.65),
    ("Strict", 0.55, 0.70),
    ("Original (#10157)", 0.55, 0.75),
)
SOURCE_SAMPLE_SIZE = 100
AREA_SAMPLE_SIZE = 10
STATISTIC_FIELDS = ("source_id", "metric", "subject_id", "node_id")


@dataclass(frozen=True)
class AtlasData:
    paths: list[str]
    sample_paths: list[str]
    sample_task_count: int
    confidence_paths: list[str] = field(default_factory=list)


def _task_coverage(
    row: dict, *, ancestors: dict[tuple[str, str], list[str]], subject_distance: float, capability_distance: float
) -> dict:
    placements = set()
    capabilities = set()
    for match in (row["matches"] or {}).get("subjects", []):
        if match["membership_similarity"] < 1 - subject_distance:
            continue
        subject = match["subject_id"]
        placements.add((subject, "subject:" + subject))
        if match["candidates"][0]["similarity"] >= 1 - capability_distance:
            capabilities.add(subject)
            placements.update((subject, node) for node in ancestors[subject, match["capability_id"]])
    outcome = row["description_status"]
    if outcome == "ready":
        outcome = "assigned" if capabilities else "subject_only" if placements else "out_of_scope"
    return {
        "source_id": row["source_id"],
        "outcome": outcome,
        "placements": sorted(placements),
        "capabilities": sorted(capabilities),
    }


def _task_metrics(row: dict) -> Iterator[tuple[str, str, str]]:
    yield "denominator", "", ""
    yield row["outcome"], "", ""
    yield from (("placement", subject, node) for subject, node in row["placements"])
    yield from (("capability", subject, "subject:" + subject) for subject in row["capabilities"])


def _partition_statistics(records: Iterator[dict], _shard: ShardInfo) -> Iterator[dict]:
    counts = Counter()
    for row in records:
        counts.update((row["source_id"], *metric) for metric in _task_metrics(row))
    for key, count in counts.items():
        yield dict(zip(STATISTIC_FIELDS, key, strict=True), instances=count)


def _statistic_key(row: dict) -> tuple:
    return tuple(row[field] for field in STATISTIC_FIELDS)


def _sum_statistics(key: tuple, records: Iterator[dict]) -> dict:
    instances = 0
    for row in records:
        instances += row["instances"]
    return dict(zip(STATISTIC_FIELDS, key, strict=True), instances=instances)


def _partition_confidence_statistics(
    records: Iterator[dict],
    _shard: ShardInfo,
    *,
    ancestors: dict[tuple[str, str], list[str]],
    levels: Sequence[tuple[str, float, float]],
) -> Iterator[dict]:
    counts: dict[tuple, list[int]] = {}
    for row in records:
        for index, (_, subject_similarity, capability_similarity) in enumerate(levels):
            coverage = _task_coverage(
                row,
                ancestors=ancestors,
                subject_distance=1 - subject_similarity,
                capability_distance=1 - capability_similarity,
            )
            for metric in _task_metrics(coverage):
                key = row["source_id"], *metric
                if key not in counts:
                    counts[key] = [0] * len(levels)
                counts[key][index] += 1
    for key, instances in counts.items():
        yield dict(zip(STATISTIC_FIELDS, key, strict=True), instances=instances)


def _sum_confidence_statistics(key: tuple, records: Iterator[dict]) -> dict:
    first = next(records)
    instances = first["instances"].copy()
    for row in records:
        instances = [left + right for left, right in zip(instances, row["instances"], strict=True)]
    return dict(zip(STATISTIC_FIELDS, key, strict=True), instances=instances)


def _sample_groups(
    row: dict,
    *,
    ancestors: dict[tuple[str, str], list[str]],
    subject_distance: float,
    capability_distance: float,
    confidence_levels: Sequence[tuple[str, float, float]],
) -> Iterator[dict]:
    yield {"kind": "source", "scope_id": row["source_id"], "task": row}
    levels = confidence_levels or (("", 1 - subject_distance, 1 - capability_distance),)
    for index, (_, subject_similarity, capability_similarity) in enumerate(levels):
        for subject, node in _task_coverage(
            row,
            ancestors=ancestors,
            subject_distance=1 - subject_similarity,
            capability_distance=1 - capability_similarity,
        )["placements"]:
            yield {"kind": "area", "scope_id": f"{index}/{subject}/{node}", "task": row}


def _sample_tasks(
    key: tuple[str, str],
    rows: Iterator[dict],
    *,
    source_size: int,
    area_size: int,
    sampling_source_ids: Mapping[str, str],
) -> Iterator[dict]:
    _, selected = seeded_sample(
        rows,
        size=source_size if key[0] == "source" else area_size,
        key=lambda row: seeded_order(sampling_source_ids[row["task"]["source_id"]] + "\0" + row["task"]["task_id"], 0),
    )
    yield from selected


def build_curriculum_coverage(
    context: ZephyrContext,
    *,
    task_paths: Sequence[str],
    nodes: Sequence[CurriculumNode],
    output_root: str,
    subject_distance: float,
    capability_distance: float,
    source_sample_size: int,
    area_sample_size: int,
    sampling_source_ids: Mapping[str, str],
    confidence_levels: Sequence[tuple[str, float, float]] = (),
) -> AtlasData:
    """Export full-population counts and bounded, overlapping task samples."""
    by_id = {(node.subject_id, node.node_id): node for node in nodes}
    ancestors = {}
    for node in nodes:
        path = [node.node_id]
        current = node
        while current.parent_id is not None:
            current = by_id[current.subject_id, current.parent_id]
            path.append(current.node_id)
        ancestors[node.subject_id, node.node_id] = path
    result = context.execute(
        Dataset.from_list(list(task_paths))
        .load_parquet(columns=["source_id", "description_status", "matches"])
        .map(
            partial(
                _task_coverage,
                ancestors=ancestors,
                subject_distance=subject_distance,
                capability_distance=capability_distance,
            )
        )
        .map_shard(_partition_statistics)
        .group_by(key=_statistic_key, reducer=_sum_statistics, num_output_shards=context.max_workers)
        .write_parquet(
            str(StoragePath(output_root) / "statistics/part-{shard:05d}.parquet"),
            schema=arrow_schema(CurriculumStatistic),
        ),
        name="curriculum-coverage-statistics",
    )
    confidence_paths = []
    if confidence_levels:
        confidence = context.execute(
            Dataset.from_list(list(task_paths))
            .load_parquet(columns=["source_id", "description_status", "matches"])
            .map_shard(partial(_partition_confidence_statistics, ancestors=ancestors, levels=confidence_levels))
            .group_by(key=_statistic_key, reducer=_sum_confidence_statistics, num_output_shards=context.max_workers)
            .write_parquet(
                str(StoragePath(output_root) / "confidence/part-{shard:05d}.parquet"),
                schema=arrow_schema(CurriculumConfidenceStatistic),
            ),
            name="curriculum-confidence-statistics",
        )
        confidence_paths = confidence.results
    sample = partial(
        _sample_tasks,
        source_size=source_sample_size,
        area_size=area_sample_size,
        sampling_source_ids=sampling_source_ids,
    )
    samples = context.execute(
        Dataset.from_list(list(task_paths))
        .load_parquet()
        .flat_map(
            partial(
                _sample_groups,
                ancestors=ancestors,
                subject_distance=subject_distance,
                capability_distance=capability_distance,
                confidence_levels=confidence_levels,
            )
        )
        .group_by(
            key=lambda row: (row["kind"], row["scope_id"]),
            reducer=sample,
            combiner=sample,
            num_output_shards=context.max_workers,
        )
        .map(lambda row: row["task"])
        .deduplicate(key=lambda row: (row["source_id"], row["task_id"]), num_output_shards=context.max_workers)
        .write_parquet(
            str(StoragePath(output_root) / "samples/part-{shard:05d}.parquet"), schema=arrow_schema(TaskAlignment)
        ),
        name="curriculum-review-samples",
    )
    sample_count = context.execute(Dataset.from_list(samples.results).load_parquet(columns=["task_id"]).count())
    return AtlasData(result.results, samples.results, sample_count.results[0], confidence_paths)
