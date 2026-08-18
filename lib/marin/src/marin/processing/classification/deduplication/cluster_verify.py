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
     dup_representative_source_tag, dup_containment, dup_jaccard,
     dup_novel_tokens, dup_comparisons}

One file per normalized shard, named after it, so the tree is co-partitioned
with the normalized data and with every other attribute tree.

The result is a
:class:`~marin.processing.classification.deduplication.verify_fuzzy_dups.VerifiedFuzzyDupsAttrData`
that records ``rule`` in place of the pipeline verifier's ``verification`` and
``local_representatives``: this rule has no local-representative stage and no
sampled comparison budget, so it cannot honestly fill those fields. The store
reads one artifact type whichever rule wrote it.
"""

import logging
import time
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import MAX_IRIS_WORKER_REPLICAS, ZephyrContext
from zephyr.worker_context import zephyr_worker_ctx
from zephyr.writers import write_parquet_file

from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.execution.artifact import write_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_dedup import (
    ClusterDedupParams,
    ClusterDocument,
    find_duplicates,
    prepare,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsAttrData,
    VerifiedFuzzyDupsPerSource,
)

logger = logging.getLogger(__name__)

CLUSTER_VERIFY_STAGE_VERSION = 1
COUNTER_PREFIX = "fuzzy/cluster_verify"
CLUSTER_TEXT_MANIFEST_FILENAME = "manifest.json"
CLUSTER_TEXT_SUBDIRECTORY = "text"
# Solving holds an n-gram set per document, so peak memory runs roughly seven
# times the cluster's text: a 14 GiB shard drove one task to 165 GiB of RSS and
# took the worker down. 256 MB of text per batch keeps that peak near 2 GB,
# comfortably inside a task budget, and no observed cluster below the cap needs
# splitting at all.
MAXIMUM_CLUSTER_CHARS = 256 * 1024 * 1024
# One document's n-grams are built as one Python object per word plus one per
# n-gram, roughly twenty times the text in allocator overhead, and a batch cap
# cannot split a single row. A 14 GiB shard held documents large enough to drive
# one task to 186 GiB of RSS inside ``ngram_hashes``. Duplicate detection reads
# containment from the head of the text, so a cap costs no signal a rule can use.
MAXIMUM_DOCUMENT_CHARS = 8 * 1024 * 1024
# The reduce writes one file per normalized shard. Beyond this many output
# shards the per-task bookkeeping costs more than the parallelism buys.
MAXIMUM_OUTPUT_SHARDS = 8192
# Every reducer reads every map task's chunk, so the shuffle costs the product of
# the two counts. Solving one grouped file per task put 65,536 mappers against
# 8,192 reducers: 537 million Parquet footer reads to move markers that total
# about a terabyte. Grouping 32 files per task and halving the reduce side twice
# leaves 2,048 x 2,048, which is 128 times less work for the same result.
#
# 2,048 is deliberate rather than as small as possible. A cluster-text solve is
# CPU bound and cw-us-east-02a offers roughly 4,100 single-CPU task slots, so
# 2,048 mappers still fill the machine in one wave. Fewer mappers would shrink an
# already small fan-in while lengthening the expensive stage.
DEFAULT_FILES_PER_TASK = 32
DEFAULT_REDUCE_SHARDS = 2048
_SHARED_SHARDS_KEY = "fuzzy_cluster_verify_shards"

_MARKER_SCHEMA = pa.schema(
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

DEFAULT_THRESHOLDS = (0.60, 0.65, 0.70, 0.80)

_SIZE_BINS = (2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536)


class ClusterTextShard(BaseModel):
    """One normalized shard, as the cluster-text manifest names it.

    ``file_idx`` is the key every grouped text row carries back, and the pair of
    ``source_tag`` and ``basename`` places that row's markers in the attribute
    tree beside the normalized shard the row came from.
    """

    file_idx: int
    source_key: str
    source_tag: str
    basename: str


class ClusterTextManifest(BaseModel):
    """The shard manifest written beside a materialized cluster-text dataset."""

    version: str
    shards: list[ClusterTextShard]


def read_cluster_text_manifest(cluster_text: str) -> ClusterTextManifest:
    """Read the shard manifest of a materialized cluster-text dataset."""
    path = StoragePath(prefix_join(cluster_text, CLUSTER_TEXT_MANIFEST_FILENAME))
    return ClusterTextManifest.model_validate_json(path.read_bytes())


def _size_bin(size: int) -> str:
    for edge in _SIZE_BINS:
        if size < edge:
            return f"{edge:06d}"
    return "999999"


def _solve_text_shards(
    paths: list[str], params: ClusterDedupParams, thresholds: tuple[float, ...]
) -> Iterator[dict[str, Any]]:
    """Solve several grouped text files in one map task.

    Each file is solved independently, exactly as one task per file did. Only the
    shuffle sees the difference: one chunk per task instead of one per file.
    """
    for path in paths:
        yield from solve_text_shard(path, params, thresholds)


def solve_text_shard(
    path: str, params: ClusterDedupParams, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
) -> Iterator[dict[str, Any]]:
    """Solve every cluster in one grouped text file.

    The file is written sorted by ``cluster_key``, so a cluster is a contiguous
    run and the whole file streams without holding more than one cluster.
    """
    started = time.monotonic()
    columns = ["cluster_key", "dup_cluster_id", "id", "text", "file_idx", "source_tag"]
    clusters = 0
    duplicates = 0
    documents = 0
    chars = 0
    pending: list[dict[str, Any]] = []
    current: str | None = None

    def batches(members: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        """Split a cluster into byte-bounded batches, longest document first.

        The upstream split caps a cluster at 100,000 members, which does not
        cap its bytes: the same member count spans 220 MB in one cluster and
        2.4 GB in another, because document size correlates strongly inside a
        cluster. Solving builds an n-gram set per document, so peak memory runs
        several times the text and an unbounded cluster takes the worker down.
        Sorting longest-first keeps the best representative in the first batch,
        which is where most members find their match.
        """
        if sum(len(row["text"]) for row in members) <= MAXIMUM_CLUSTER_CHARS:
            yield members
            return
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/oversized_clusters", 1)
        ordered = sorted(members, key=lambda row: -len(row["text"]))
        batch: list[dict[str, Any]] = []
        size = 0
        for row in ordered:
            if batch and size + len(row["text"]) > MAXIMUM_CLUSTER_CHARS:
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
        """Emit one record per removed document, marking every threshold that removed it.

        Building the n-gram sets dominates the cost and does not depend on the
        threshold, so the cover runs once per threshold over one ``prepare``.
        Measured on real clusters, a cover is 15% of a single-threshold solve
        while the n-gram build is half of it, so four thresholds cost about 45%
        more than one against four separate passes over the same text.

        The record carries a bitmask rather than being emitted once per
        threshold, which keeps the shuffle at the size of a single threshold's
        markers. The reduce expands the mask into one ordinary marker tree per
        threshold, so the mask never leaves this job.

        Removal is not monotonic in the threshold, so the mask cannot be
        collapsed to the strictest threshold that fired. Lowering the bar can
        remove a member's only representative and let the member survive: a
        cluster was measured dropping one document at 0.50 and keeping it at
        0.45, for exactly that reason.
        """
        nonlocal duplicates
        if len(members) < 2:
            return
        cluster = [
            ClusterDocument(id=row["id"], text=row["text"], file_idx=row["file_idx"], source_tag=row["source_tag"])
            for row in members
        ]
        prepared = prepare(cluster, params)
        marks: dict[int, dict[str, Any]] = {}
        for bit, threshold in enumerate(thresholds):
            rule = params.model_copy(update={"minimum_containment": threshold})
            for removal in find_duplicates(cluster, rule, prepared=prepared):
                record: dict[str, Any] | None = marks.get(removal.member_index)
                if record is None:
                    member = members[removal.member_index]
                    representative = members[removal.representative_index]
                    record = {
                        "file_idx": member["file_idx"],
                        "id": member["id"],
                        "dup_doc": True,
                        "dup_thresholds": 0,
                        "dup_cluster_id": member["dup_cluster_id"],
                        "dup_representative_id": representative["id"],
                        "dup_representative_source_tag": representative["source_tag"],
                        "dup_containment": removal.containment,
                        "dup_jaccard": removal.jaccard,
                        "dup_novel_tokens": removal.novel_tokens,
                        "dup_comparisons": removal.comparisons,
                    }
                    marks[removal.member_index] = record
                record["dup_thresholds"] |= 1 << bit
                counters.pipeline.update_counter(f"{COUNTER_PREFIX}/duplicates/{threshold:.2f}", 1)
        duplicates += len(marks)
        yield from marks.values()

    # The file is written sorted by cluster_key, so a cluster is a contiguous
    # run and only one cluster is ever resident. That alone does not bound
    # memory: a single cluster can hold gigabytes of text, and one did -- a
    # 14 GiB shard drove a task to 185 GiB of RSS and was OOM-killed. Flushing
    # mid-cluster once the buffer passes the budget is what actually bounds it,
    # because the buffer is what grows, not the solver.
    pending_chars = 0
    with StoragePath(path).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(columns=columns, batch_size=8192):
            for row in batch.to_pylist():
                documents += 1
                chars += len(row["text"])
                if len(row["text"]) > MAXIMUM_DOCUMENT_CHARS:
                    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/truncated_documents", 1)
                    row["text"] = row["text"][:MAXIMUM_DOCUMENT_CHARS]
                if row["cluster_key"] != current:
                    if pending:
                        yield from solve(pending)
                    pending = []
                    pending_chars = 0
                    current = row["cluster_key"]
                elif pending_chars > MAXIMUM_CLUSTER_CHARS:
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


def _write_markers(
    file_idx: int,
    records: Iterator[dict[str, Any]],
    output_path: str,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    """Write this shard's markers, one ordinary attribute tree per threshold.

    Each record carries the bitmask of thresholds that removed its document.
    Expanding it here rather than in the map keeps the shuffle at one
    threshold's size, and every tree written is the plain marker shape the store
    already reads: rows present means drop, and ``dup_doc`` is always true.
    """
    shards: dict[int, ClusterTextShard] = zephyr_worker_ctx().get_shared(_SHARED_SHARDS_KEY)
    shard = shards[file_idx]
    # The reducer streams its records once, and each threshold keeps a different
    # subset, so the shard is held in memory. A shard's markers are a few tens of
    # thousands of ~380 byte rows, well under a task budget.
    held = list(records)
    written: dict[str, int] = {}
    for bit, threshold in enumerate(thresholds):
        kept = [record for record in held if record["dup_thresholds"] & (1 << bit)]
        path = prefix_join(_attr_dir(output_path, threshold, shard.source_tag), shard.basename)
        rows = ({field.name: record[field.name] for field in _MARKER_SCHEMA} for record in kept)
        write_parquet_file(rows, path, schema=_MARKER_SCHEMA)
        written[f"{threshold:.2f}"] = len(kept)
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/markers/{threshold:.2f}", len(kept))
    return {"file_idx": file_idx, "markers": written}


def threshold_directory(threshold: float) -> str:
    """Name of the marker tree a threshold writes, as the store is pointed at it."""
    return f"markers@{threshold:.2f}"


def _attr_dir(output_path: str, threshold: float, source_tag: str) -> str:
    return prefix_join(output_path, f"{threshold_directory(threshold)}/outputs/{source_tag}")


def verify_cluster_text(
    *,
    cluster_text: str,
    output_path: str,
    params: ClusterDedupParams,
    max_workers: int | None = None,
    worker_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    text_file_limit: int | None = None,
    files_per_task: int = DEFAULT_FILES_PER_TASK,
    reduce_shards: int = DEFAULT_REDUCE_SHARDS,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> VerifiedFuzzyDupsAttrData:
    """Solve a materialized cluster-text dataset and write duplicate markers.

    Args:
        cluster_text: Root of the grouped text: ``text/*.parquet`` beside the
            ``manifest.json`` that names every normalized shard.
        output_path: Root of the attribute tree to write.
        params: The duplicate rule, recorded on the result.
        max_workers: Worker limit. Defaults to one worker per grouped text file.
        worker_resources: Shape of one worker.
        map_task_resources: Shape of one solving task.
        reduce_task_resources: Shape of one marker-writing task.
        text_file_limit: Solve only the first N grouped text files, for a smoke
            run over part of a materialized dataset. The source manifest still
            names every source, so a partial output stays readable.
        files_per_task: Grouped text files solved by one map task. Every reducer
            reads every map task's chunk, so the shuffle costs the product of the
            two counts: one file per task made 65,536 mappers against 8,192
            reducers, and each reducer opened all 65,536 chunks to find its
            120 MB slice. Grouping the map side divides that product without
            changing the result.
        reduce_shards: Reduce tasks. Markers are ~380 bytes and total about a
            terabyte, so a reducer holds a few hundred megabytes whatever this
            is; it exists to bound the fan-in, not to fit memory.

    Returns:
        The marker attribute tree, one directory per source key.
    """
    manifest = read_cluster_text_manifest(cluster_text)
    if not manifest.shards:
        raise ValueError(f"{cluster_text} manifest names no normalized shards")
    shards = {shard.file_idx: shard for shard in manifest.shards}

    text_dir = prefix_join(cluster_text, CLUSTER_TEXT_SUBDIRECTORY)
    paths = sorted(str(path) for path in StoragePath(prefix_join(text_dir, "*.parquet")).glob())
    if not paths:
        raise FileNotFoundError(f"No grouped text files under {text_dir}")
    if text_file_limit is not None:
        paths = paths[:text_file_limit]
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
        max_workers=max_workers or min(len(groups), MAX_IRIS_WORKER_REPLICAS),
    )
    context.put(_SHARED_SHARDS_KEY, shards)
    pipeline = (
        Dataset.from_list(groups)
        .flat_map(lambda group: _solve_text_shards(group, params, thresholds))
        .group_by(
            key=lambda record: record["file_idx"],
            reducer=lambda file_idx, records: _write_markers(file_idx, records, output_path, thresholds),
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

    # Each threshold is a self-contained attribute tree: its own source manifest
    # and its own artifact record, so pointing the store at one is repointing a
    # path. Consumers resolve a tree through its manifest, the same way they
    # resolve every other co-partitioned Datakit output.
    source_tags = {shard.source_key: shard.source_tag for shard in manifest.shards}
    output_counters: dict[str, int | float] = dict(outcome.counters)
    output_counters[f"{COUNTER_PREFIX}/text_files"] = len(paths)
    results: dict[float, VerifiedFuzzyDupsAttrData] = {}
    for threshold in thresholds:
        tree = prefix_join(output_path, threshold_directory(threshold))
        attr_dirs = {key: _attr_dir(output_path, threshold, tag) for key, tag in source_tags.items()}
        write_copartitioned_source_manifest(output_path=tree, attr_dirs=attr_dirs)
        markers = sum(result["markers"][f"{threshold:.2f}"] for result in outcome.results)
        counters_for_tree = dict(output_counters)
        counters_for_tree[f"{COUNTER_PREFIX}/markers"] = markers
        results[threshold] = VerifiedFuzzyDupsAttrData(
            rule=params.model_copy(update={"minimum_containment": threshold}),
            sources={
                key: VerifiedFuzzyDupsPerSource(attr_dir=attr_dirs[key], source_tag=tag)
                for key, tag in source_tags.items()
            },
            counters=counters_for_tree,
        )
        write_artifact(results[threshold], tree)
        logger.info("Threshold %.2f: %d markers under %s", threshold, markers, tree)

    # The caller's own threshold names the artifact this returns; the rest stand
    # on disk for a later choice.
    primary = params.minimum_containment
    if primary not in results:
        primary = thresholds[0]
    return results[primary]


def cluster_verify_step(
    *,
    name: str,
    cluster_text_step: StepSpec,
    params: ClusterDedupParams,
    max_workers: int | None = None,
    worker_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a step that solves one materialized cluster-text dataset.

    The rule enters ``hash_attrs``, so moving a threshold moves the output path
    and the store below it rebuilds against the markers the new rule wrote.
    """
    return StepSpec(
        name=name,
        deps=[cluster_text_step],
        hash_attrs={
            "artifact_version": CLUSTER_VERIFY_STAGE_VERSION,
            "rule": params.model_dump(mode="json"),
        },
        fn=lambda output_path: verify_cluster_text(
            cluster_text=cluster_text_step.output_path,
            output_path=output_path,
            params=params,
            max_workers=max_workers,
            worker_resources=worker_resources,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
        ),
        override_output_path=override_output_path,
    )
