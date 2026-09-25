# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the production cluster-verification markers in normalized shard order.

The word 3-gram rule and text limits reproduce the v11-c075-restored hero run
from PR 8405. Prefix truncation and cluster splits can change removal decisions.
"""

import logging
import time
from collections.abc import Iterator
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_dedup import (
    ClusterDedupParams,
    find_duplicates,
)
from marin.processing.classification.deduplication.cluster_text import (
    CLUSTER_TEXT_SUBDIRECTORY,
    CLUSTER_TEXT_SUCCESS_FILENAME,
    ClusterTextData,
    ClusterTextShard,
    read_cluster_text_manifest,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsArtifact,
    VerifiedFuzzyDupsPerSource,
)

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_verify"
# Grouping map inputs and bounding reducers limits the shuffle fan-in while
# retaining enough map tasks to use the cluster.
DEFAULT_FILES_PER_TASK = 32
DEFAULT_REDUCE_SHARDS = 2048

# A reduce attempt opens every mapper's chunk, so brief object-store failures
# can affect several shards in one wave.
DEFAULT_MAX_SHARD_FAILURES = 20
_SHARED_SHARDS_KEY = "fuzzy_cluster_verify_shards"

_SIZE_BINS = (2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384, 65536)


CLUSTER_DUPLICATE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_doc", pa.bool_(), nullable=False),
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("dup_representative_id", pa.string(), nullable=False),
        pa.field("dup_representative_source_tag", pa.string(), nullable=False),
        pa.field("dup_containment", pa.float32(), nullable=False),
        pa.field("dup_jaccard", pa.float32(), nullable=False),
        pa.field("dup_novel_tokens", pa.int32(), nullable=False),
        pa.field("dup_comparisons", pa.int32(), nullable=False),
    ]
)


class ClusterVerificationLimits(BaseModel):
    """Production text limits that determine verification batches."""

    model_config = ConfigDict(frozen=True)

    maximum_document_chars: int = Field(default=8 * 1024 * 1024, ge=1)
    maximum_cluster_chars: int = Field(default=256 * 1024 * 1024, ge=1)


class ClusterVerifiedFuzzyDupsAttrData(VerifiedFuzzyDupsArtifact):
    """Sparse markers from whole-cluster containment verification."""

    producer: Literal["pipeline", "cluster"] = "cluster"
    version: str = "v1"
    rule: ClusterDedupParams
    limits: ClusterVerificationLimits = ClusterVerificationLimits()

    @model_validator(mode="after")
    def _cluster_producer(self) -> "ClusterVerifiedFuzzyDupsAttrData":
        if self.producer != "cluster":
            raise ValueError("Cluster verified-marker artifacts require producer='cluster'")
        return self


def _size_bin(size: int) -> str:
    for edge in _SIZE_BINS:
        if size < edge:
            return f"{edge:06d}"
    return "999999"


def _solve_text_shards(
    paths: list[str], params: ClusterDedupParams, limits: ClusterVerificationLimits
) -> Iterator[dict[str, Any]]:
    for path in paths:
        yield from solve_text_shard(path, params, limits)


def solve_text_shard(
    path: str, params: ClusterDedupParams, limits: ClusterVerificationLimits = ClusterVerificationLimits()
) -> Iterator[dict[str, Any]]:
    """Solve every cluster in one grouped text file.

    The file is written sorted by ``cluster_key``, so a cluster is a contiguous
    run and the whole file streams without holding more than one cluster.
    """
    started = time.monotonic()
    columns = ["cluster_key", "dup_cluster_id", "id", "text", "file_idx"]
    clusters = 0
    duplicates = 0
    documents = 0
    chars = 0
    pending: list[dict[str, Any]] = []
    current: str | None = None

    def batches(members: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        """Split the final buffer longest-first, as in the production run."""
        if sum(len(row["text"]) for row in members) <= limits.maximum_cluster_chars:
            yield members
            return
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/oversized_clusters", 1)
        ordered = sorted(members, key=lambda row: -len(row["text"]))
        batch: list[dict[str, Any]] = []
        size = 0
        for row in ordered:
            if batch and size + len(row["text"]) > limits.maximum_cluster_chars:
                yield batch
                batch, size = [], 0
            batch.append(row)
            size += len(row["text"])
        if batch:
            yield batch

    def solve(members: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        nonlocal clusters
        clusters += 1
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/cluster_size/{_size_bin(len(members))}", 1)
        if len(members) < 2:
            counters.pipeline.update_counter(f"{COUNTER_PREFIX}/singleton_groups", 1)
            return
        for chunk in batches(members):
            yield from _solve_batch(chunk)

    def _solve_batch(members: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        nonlocal duplicates
        if len(members) < 2:
            return
        cluster = [row["text"] for row in members]
        shards: dict[int, ClusterTextShard] = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)
        for removal in find_duplicates(cluster, params):
            member = members[removal.member_index]
            representative = members[removal.representative_index]
            duplicates += 1
            yield {
                "file_idx": member["file_idx"],
                "id": member["id"],
                "dup_doc": True,
                "dup_cluster_id": member["dup_cluster_id"],
                "dup_representative_id": representative["id"],
                "dup_representative_source_tag": shards[representative["file_idx"]].source_tag,
                "dup_containment": removal.containment,
                "dup_jaccard": removal.jaccard,
                "dup_novel_tokens": removal.novel_tokens,
                "dup_comparisons": removal.comparisons,
            }

    # Production flushes after the existing buffer exceeds the limit. Changing
    # this to a check before the next append changes which documents can match.
    pending_chars = 0
    with StoragePath(path).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(columns=columns, batch_size=8192):
            for row in batch.to_pylist():
                documents += 1
                chars += len(row["text"])
                if len(row["text"]) > limits.maximum_document_chars:
                    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/truncated_documents", 1)
                    row["text"] = row["text"][: limits.maximum_document_chars]
                if row["cluster_key"] != current:
                    if pending:
                        yield from solve(pending)
                    pending = []
                    pending_chars = 0
                    current = row["cluster_key"]
                elif pending_chars > limits.maximum_cluster_chars:
                    # Still inside one cluster, but it no longer fits. Solve what
                    # is buffered and keep going: members split across flushes
                    # lose the chance to match each other, which is the same
                    # trade the upstream member-count split already makes.
                    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/mid_cluster_flushes", 1)
                    yield from _solve_batch(pending)
                    pending = []
                    pending_chars = 0
                pending.append(row)
                pending_chars += len(row["text"])
    if pending:
        yield from solve(pending)

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/documents", documents)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/text_chars", chars)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/clusters", clusters)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/duplicates", duplicates)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/solve_seconds_milli", int((time.monotonic() - started) * 1000))


def _write_markers(file_idx: int, records: Iterator[dict[str, Any]], output_path: str) -> dict[str, Any]:
    """Write one shard's markers into the co-partitioned attribute tree."""
    shards: dict[int, ClusterTextShard] = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)
    shard = shards[file_idx]
    path = prefix_join(_attr_dir(output_path, shard.source_tag), shard.basename)
    rows = ({field.name: record[field.name] for field in CLUSTER_DUPLICATE_SCHEMA} for record in records)
    result = write_parquet_file(rows, path, schema=CLUSTER_DUPLICATE_SCHEMA)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/source/{shard.source_tag}/markers", result["count"])
    return {**result, "file_idx": file_idx, "markers": result["count"]}


def _attr_dir(output_path: str, source_tag: str) -> str:
    return prefix_join(output_path, f"outputs/{source_tag}")


def verify_cluster_text(
    *,
    cluster_text: str,
    output_path: str,
    params: ClusterDedupParams = ClusterDedupParams(),
    limits: ClusterVerificationLimits = ClusterVerificationLimits(),
    max_workers: int | None = None,
    worker_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    files_per_task: int = DEFAULT_FILES_PER_TASK,
    reduce_shards: int = DEFAULT_REDUCE_SHARDS,
    max_shard_failures: int = DEFAULT_MAX_SHARD_FAILURES,
) -> ClusterVerifiedFuzzyDupsAttrData:
    """Solve a materialized cluster-text dataset and write duplicate markers.

    Args:
        cluster_text: Root of the grouped text: ``text/*.parquet`` beside the
            ``manifest.json`` that names every normalized shard.
        output_path: Root of the attribute tree to write.
        params: The duplicate rule, recorded on the result.
        limits: Text limits and batch boundaries, recorded on the result.
        max_workers: Worker limit. Defaults to one worker per map task.
        worker_resources: Shape of one worker.
        map_task_resources: Shape of one solving task.
        reduce_task_resources: Shape of one marker-writing task.
        files_per_task: Grouped text files solved by one map task. Every reducer
            reads every map task's chunk, so the shuffle costs the product of the
            two counts: one file per task made 65,536 mappers against 8,192
            reducers, and each reducer opened all 65,536 chunks to find its
            120 MB slice. Grouping the map side divides that product without
            changing the result.
        reduce_shards: Reduce tasks. Markers are ~380 bytes and total about a
            terabyte, so a reducer holds a few hundred megabytes whatever this
            is; it exists to bound the fan-in, not to fit memory.
        max_shard_failures: Attempts one shard gets before the pipeline aborts.
            Zephyr defaults to 3, which is too few here: one reduce attempt
            opens every mapper's chunk, so the wave issues millions of requests
            and a brief object-store outage lands on several shards at once. The
            first production run died that way with seven shards out of 2,048
            exhausted, having already written most of its markers.

    Returns:
        The marker attribute tree, one directory per source key.
    """
    success_path = prefix_join(cluster_text, CLUSTER_TEXT_SUCCESS_FILENAME)
    if not StoragePath(success_path).exists():
        raise FileNotFoundError(f"Cluster-text artifact is incomplete: {success_path} is absent")
    manifest = read_cluster_text_manifest(cluster_text)
    if not manifest.shards:
        raise ValueError(f"{cluster_text} manifest names no normalized shards")
    shards = {shard.file_idx: shard for shard in manifest.shards}

    text_dir = prefix_join(cluster_text, CLUSTER_TEXT_SUBDIRECTORY)
    paths = sorted(str(path) for path in StoragePath(prefix_join(text_dir, "*.parquet")).glob())
    if files_per_task < 1:
        raise ValueError(f"files_per_task must be at least 1, got {files_per_task}")
    groups = [paths[start : start + files_per_task] for start in range(0, len(paths), files_per_task)]
    reduce_shards = max(1, min(len(shards), reduce_shards))
    logger.info(
        "Solving %d grouped text files as %d map tasks into %d reduce tasks with %s",
        len(paths),
        len(groups),
        reduce_shards,
        params.model_dump_json(),
    )

    context = ZephyrContext(
        name="fuzzy-cluster-verify",
        resources=worker_resources,
        max_workers=max_workers or max(1, len(groups)),
        max_shard_failures=max_shard_failures,
    )
    context.put(_SHARED_SHARDS_KEY, shards)
    pipeline = (
        Dataset.from_list(groups)
        .flat_map(lambda group: _solve_text_shards(group, params, limits))
        .group_by(
            key=lambda record: record["file_idx"],
            reducer=lambda file_idx, records: _write_markers(file_idx, records, output_path),
            sort_by=lambda record: record["id"],
            num_output_shards=reduce_shards,
        )
    )
    outcome = context.execute(
        pipeline,
        verbose=True,
        map_task_resources=map_task_resources,
        reduce_task_resources=reduce_task_resources,
    )

    # Consumers resolve an attribute tree through its source manifest, the same
    # way they resolve every other co-partitioned Datakit output.
    source_tags = {shard.source_key: shard.source_tag for shard in manifest.shards}
    attr_dirs = {source_key: _attr_dir(output_path, source_tag) for source_key, source_tag in source_tags.items()}
    write_copartitioned_source_manifest(output_path=output_path, attr_dirs=attr_dirs)
    markers = sum(result["markers"] for result in outcome.results)
    output_counters: dict[str, int | float] = dict(outcome.counters)
    output_counters.setdefault(f"{COUNTER_PREFIX}/documents", 0)
    output_counters[f"{COUNTER_PREFIX}/markers"] = markers
    output_counters[f"{COUNTER_PREFIX}/text_files"] = len(paths)
    logger.info("Wrote %d duplicate markers from %d grouped text files", markers, len(paths))
    return ClusterVerifiedFuzzyDupsAttrData(
        rule=params,
        limits=limits,
        sources={
            source_key: VerifiedFuzzyDupsPerSource(attr_dir=attr_dirs[source_key], source_tag=source_tag)
            for source_key, source_tag in source_tags.items()
        },
        counters=output_counters,
    )


def cluster_verify_step(
    *,
    name: str,
    cluster_text: StepSpec,
    params: ClusterDedupParams = ClusterDedupParams(),
    limits: ClusterVerificationLimits = ClusterVerificationLimits(),
    max_workers: int = 64,
    worker_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    files_per_task: int = DEFAULT_FILES_PER_TASK,
    reduce_shards: int = DEFAULT_REDUCE_SHARDS,
    max_shard_failures: int = DEFAULT_MAX_SHARD_FAILURES,
) -> StepSpec:
    """Create verification with the duplicate rule and text limits in its identity."""
    return StepSpec(
        name=name,
        deps=[cluster_text],
        hash_attrs={
            "version": 1,
            "rule": params.model_dump(mode="json"),
            "limits": limits.model_dump(mode="json"),
        },
        fn=lambda output_path: verify_cluster_text(
            cluster_text=read_artifact(cluster_text.output_path, ClusterTextData).path,
            output_path=output_path,
            params=params,
            limits=limits,
            max_workers=max_workers,
            worker_resources=worker_resources,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
            files_per_task=files_per_task,
            reduce_shards=reduce_shards,
            max_shard_failures=max_shard_failures,
        ),
    )
