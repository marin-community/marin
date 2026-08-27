# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solve every materialized fuzzy-duplicate cluster and write duplicate markers.

Reads the cluster-grouped candidate text, solves each cluster with the
containment rule in :mod:`~marin.processing.classification.deduplication.cluster_dedup`,
and writes the decisions into a co-partitioned attribute tree that the store
filters on. The text arrives already grouped, so this stage is an embarrassingly
parallel map: no memory store, no re-join, and no cross-cluster communication.

Output rows follow the shape the store already consumes::

    {id, dup_doc, dup_cluster_id, dup_representative_id,
     dup_representative_source_key, dup_representative_kind,
     dup_member_containment, dup_jaccard, dup_comparisons, ...}

One file per normalized shard, named after it, so the tree is co-partitioned
with the normalized data and with every other attribute tree.

The result records the cluster rule and the source mapping needed by the store.
"""

import logging
import time
from collections.abc import Iterator
from typing import Any, Literal

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import model_validator
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.processing.classification.deduplication.cluster_dedup import (
    ClusterDedupParams,
    ClusterDocument,
    find_duplicates,
)
from marin.processing.classification.deduplication.cluster_text import (
    CLUSTER_TEXT_SUBDIRECTORY,
    CLUSTER_TEXT_SUCCESS_FILENAME,
    read_cluster_text_manifest,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VERIFIED_FUZZY_DUPLICATE_SCHEMA,
    RepresentativeKind,
    VerifiedFuzzyDupsArtifact,
    VerifiedFuzzyDupsPerSource,
)

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_verify"
# N-gram preparation expands text into Python and NumPy objects. This cap is
# paired with the CLI's 12 GiB default task memory.
MAXIMUM_CLUSTER_CHARS = 256 * 1024 * 1024
# Grouping map inputs and bounding reducers limits the shuffle fan-in while
# retaining enough map tasks to use the cluster.
DEFAULT_FILES_PER_TASK = 32
DEFAULT_REDUCE_SHARDS = 2048

# A reduce attempt opens every mapper's chunk, so brief object-store failures
# can affect several shards in one wave.
DEFAULT_MAX_SHARD_FAILURES = 20
_SHARED_SHARDS_KEY = "fuzzy_cluster_verify_shards"

_SIZE_BINS = (2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384, 65536)


class ClusterVerifiedFuzzyDupsAttrData(VerifiedFuzzyDupsArtifact):
    """Sparse markers from whole-cluster containment verification."""

    producer: Literal["pipeline", "cluster"] = "cluster"
    version: str = "v1"
    rule: ClusterDedupParams

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


def _solve_text_shards(paths: list[str], params: ClusterDedupParams) -> Iterator[dict[str, Any]]:
    for path in paths:
        yield from solve_text_shard(path, params)


def solve_text_shard(path: str, params: ClusterDedupParams) -> Iterator[dict[str, Any]]:
    """Solve every cluster in one grouped text file.

    The file is written sorted by ``cluster_key``, so a cluster is a contiguous
    run and the whole file streams without holding more than one cluster.
    """
    started = time.monotonic()
    columns = ["cluster_key", "dup_cluster_id", "id", "text", "text_truncated", "file_idx"]
    clusters = 0
    duplicates = 0
    documents = 0
    chars = 0
    pending: list[dict[str, Any]] = []
    current: str | None = None

    def record_cluster(size: int) -> None:
        nonlocal clusters
        clusters += 1
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/cluster_size/{_size_bin(size)}", 1)
        if size < 2:
            counters.pipeline.update_counter(f"{COUNTER_PREFIX}/singleton_groups", 1)

    def _solve_batch(members: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        nonlocal duplicates
        eligible = [member for member in members if not member["text_truncated"]]
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/truncated_documents_skipped", len(members) - len(eligible))
        if len(eligible) < 2:
            return
        cluster = [ClusterDocument(id=row["id"], text=row["text"]) for row in eligible]
        shards: dict[int, tuple[str, str, str]] = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)
        for removal in find_duplicates(cluster, params):
            member = eligible[removal.member_index]
            representative = eligible[removal.representative_index]
            duplicates += 1
            yield {
                "file_idx": member["file_idx"],
                "id": member["id"],
                "dup_doc": True,
                "dup_cluster_id": member["dup_cluster_id"],
                "dup_representative_id": representative["id"],
                "dup_representative_source_key": shards[representative["file_idx"]][0],
                "dup_representative_kind": RepresentativeKind.CLUSTER_LONGEST.value,
                "dup_shared_lsh_buckets": 0,
                "dup_member_containment": removal.containment,
                "dup_jaccard": removal.jaccard,
                "dup_comparisons": removal.comparisons,
                "dup_under_tokenized": False,
                "dup_char_jaccard": None,
                "dup_local_line_count_ratio": None,
            }

    pending_chars = 0
    cluster_members = 0
    cluster_flushed = False
    with StoragePath(path).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(columns=columns, batch_size=8192):
            for row in batch.to_pylist():
                documents += 1
                chars += len(row["text"])
                if row["cluster_key"] != current:
                    if pending:
                        yield from _solve_batch(pending)
                    if current is not None:
                        record_cluster(cluster_members)
                    pending = []
                    pending_chars = 0
                    cluster_members = 0
                    cluster_flushed = False
                    current = row["cluster_key"]
                elif pending and pending_chars + len(row["text"]) > MAXIMUM_CLUSTER_CHARS:
                    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/mid_cluster_flushes", 1)
                    if not cluster_flushed:
                        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/oversized_clusters", 1)
                        cluster_flushed = True
                    yield from _solve_batch(pending)
                    pending = []
                    pending_chars = 0
                pending.append(row)
                pending_chars += len(row["text"])
                cluster_members += 1
    if pending:
        yield from _solve_batch(pending)
    if current is not None:
        record_cluster(cluster_members)

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/documents", documents)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/text_chars", chars)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/clusters", clusters)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/duplicates", duplicates)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/solve_seconds_milli", int((time.monotonic() - started) * 1000))


def _write_markers(file_idx: int, records: Iterator[dict[str, Any]], output_path: str) -> dict[str, Any]:
    """Write one shard's markers into the co-partitioned attribute tree."""
    shards: dict[int, tuple[str, str, str]] = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)
    _, source_tag, basename = shards[file_idx]
    path = prefix_join(_attr_dir(output_path, source_tag), basename)
    rows = ({field.name: record[field.name] for field in VERIFIED_FUZZY_DUPLICATE_SCHEMA} for record in records)
    result = write_parquet_file(rows, path, schema=VERIFIED_FUZZY_DUPLICATE_SCHEMA)
    return {**result, "file_idx": file_idx, "markers": result["count"]}


def _attr_dir(output_path: str, source_tag: str) -> str:
    return prefix_join(output_path, f"outputs/{source_tag}")


def verify_cluster_text(
    *,
    cluster_text: str,
    output_path: str,
    params: ClusterDedupParams,
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
    shards = {shard.file_idx: (shard.source_key, shard.source_tag, shard.basename) for shard in manifest.shards}

    text_dir = prefix_join(cluster_text, CLUSTER_TEXT_SUBDIRECTORY)
    paths = sorted(str(path) for path in StoragePath(prefix_join(text_dir, "*.parquet")).glob())
    if not paths:
        raise FileNotFoundError(f"No grouped text files under {text_dir}")
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
        max_workers=max_workers or len(groups),
        max_shard_failures=max_shard_failures,
    )
    context.put(_SHARED_SHARDS_KEY, shards)
    pipeline = (
        Dataset.from_list(groups)
        .flat_map(lambda group: _solve_text_shards(group, params))
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
    output_counters[f"{COUNTER_PREFIX}/markers"] = markers
    output_counters[f"{COUNTER_PREFIX}/text_files"] = len(paths)
    logger.info("Wrote %d duplicate markers from %d grouped text files", markers, len(paths))
    return ClusterVerifiedFuzzyDupsAttrData(
        rule=params,
        sources={
            source_key: VerifiedFuzzyDupsPerSource(attr_dir=attr_dirs[source_key], source_tag=source_tag)
            for source_key, source_tag in source_tags.items()
        },
        counters=output_counters,
    )
