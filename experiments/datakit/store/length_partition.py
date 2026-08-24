# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Partition an existing clustered token store by document length."""

import dataclasses
import json
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from itertools import pairwise

import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import CacheLedger, CacheMetadata, TreeCache, _merge_sharded_ledgers
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from pydantic import BaseModel
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, format_shard_path

from experiments.datakit.store.bucket_writer import write_bucket_cache
from experiments.datakit.store.datakit_store import BucketCacheStats, ClusteredStoreData

DOCUMENT_LENGTH_THRESHOLD = 65_536
READ_CHUNK_ELEMENTS = 16 * 1024 * 1024
SOURCE_STORE_PATH = "gs://marin-us-central2/datakit/store_8ac06c74"
INPUT_IDS = "input_ids"


class DocumentLengthBucket(StrEnum):
    LTE_64K = "lte_64k"
    GT_64K = "gt_64k"


class LengthBucketCacheStats(BaseModel):
    cluster_id: int
    quality_bucket: int
    length_bucket: DocumentLengthBucket
    path: str
    total_elements: int
    total_tokens: int
    n_shards: int


class LengthPartitionedStoreData(Artifact):
    source_store: str
    cluster_view: int
    split: str
    length_threshold: int
    buckets: list[LengthBucketCacheStats]
    source_names: list[str]
    tokenizer: str
    counters: dict[str, int]


@dataclasses.dataclass(frozen=True)
class LengthPartitionConfig:
    source_path: str
    output_path: str
    worker_resources: ResourceConfig
    max_workers: int


@dataclasses.dataclass(frozen=True)
class _SourceLeaf:
    cluster: int
    quality: int
    path: str


@dataclasses.dataclass(frozen=True)
class _PartitionTask:
    leaves: list[_SourceLeaf]
    task: int
    total_tasks: int


@dataclasses.dataclass(frozen=True)
class _WrittenShard:
    cluster: int
    quality: int
    length_bucket: DocumentLengthBucket
    path: str
    rows: int
    tokens: int


def _source_leaf_groups(source: ClusteredStoreData) -> list[list[_SourceLeaf]]:
    def bucket_leaves(bucket: BucketCacheStats) -> list[tuple[str, _SourceLeaf]]:
        ledger = CacheLedger.load(bucket.path)
        if ledger.layout != "sharded":
            return [(bucket.path, _SourceLeaf(bucket.cluster_id, bucket.quality_bucket, bucket.path))]
        return [
            (
                shard,
                _SourceLeaf(bucket.cluster_id, bucket.quality_bucket, prefix_join(bucket.path, shard)),
            )
            for shard in ledger.finished_shards
        ]

    grouped: dict[str, list[_SourceLeaf]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=min(32, len(source.buckets))) as executor:
        for leaves in executor.map(bucket_leaves, source.buckets):
            for shard, leaf in leaves:
                grouped[shard].append(leaf)
    return [grouped[shard] for shard in sorted(grouped)]


def _selected_token_chunks(store, ends: np.ndarray, selected: np.ndarray) -> Iterator[np.ndarray]:
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1]
    boundaries = np.concatenate(([0], np.flatnonzero(selected[1:] != selected[:-1]) + 1, [len(selected)]))
    for first, last in pairwise(boundaries):
        if not selected[first]:
            continue
        token_start = int(starts[first])
        token_stop = int(ends[last - 1])
        for start in range(token_start, token_stop, READ_CHUNK_ELEMENTS):
            stop = min(start + READ_CHUNK_ELEMENTS, token_stop)
            yield np.asarray(store.data[start:stop].read().result(), dtype=np.int32)


def _partition_leaf(
    leaf: _SourceLeaf,
    *,
    output_path: str,
    task: int,
    total_tasks: int,
    attempt: str,
) -> list[_WrittenShard]:
    cache = TreeCache.load(leaf.path, {INPUT_IDS: np.array([0], dtype=np.int32)})
    input_ids = cache.jagged_array_tree()[INPUT_IDS]
    ends = np.asarray(input_ids.offsets[1 : len(cache) + 1].read().result(), dtype=np.int64)
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1]
    lengths = ends - starts
    is_long = lengths > DOCUMENT_LENGTH_THRESHOLD

    written = []
    for length_bucket, selected in (
        (DocumentLengthBucket.LTE_64K, ~is_long),
        (DocumentLengthBucket.GT_64K, is_long),
    ):
        selected_lengths = lengths[selected]
        if not len(selected_lengths):
            continue
        bucket_root = prefix_join(
            output_path,
            f"cluster={leaf.cluster}/quality={leaf.quality}/length={length_bucket.value}",
        )
        pattern = prefix_join(bucket_root, f"part-{{shard:05d}}-of-{{total:05d}}-attempt-{attempt}")
        cache_path = format_shard_path(pattern, task, total_tasks)
        ledger = write_bucket_cache(
            cache_path,
            _selected_token_chunks(input_ids, ends, selected),
            selected_lengths,
        )
        written.append(
            _WrittenShard(
                cluster=leaf.cluster,
                quality=leaf.quality,
                length_bucket=length_bucket,
                path=cache_path,
                rows=ledger.total_num_rows,
                tokens=ledger.field_counts[INPUT_IDS],
            )
        )
    return written


def _sidecar_path(output_path: str, task: int, total_tasks: int) -> str:
    pattern = prefix_join(output_path, "_done/shard-{shard:05d}-of-{total:05d}.json")
    return format_shard_path(pattern, task, total_tasks)


def _write_sidecar(path: str, shards: list[_WrittenShard]) -> None:
    payload = json.dumps([dataclasses.asdict(shard) for shard in shards], sort_keys=True)
    with atomic_rename(path) as temporary_path:
        StoragePath(temporary_path).write_text(payload)


def _load_sidecar(path: str) -> list[_WrittenShard]:
    return [
        _WrittenShard(
            cluster=item["cluster"],
            quality=item["quality"],
            length_bucket=DocumentLengthBucket(item["length_bucket"]),
            path=item["path"],
            rows=item["rows"],
            tokens=item["tokens"],
        )
        for item in json.loads(StoragePath(path).read_text())
    ]


def _partition_task(task_input: _PartitionTask, *, output_path: str) -> str:
    sidecar = _sidecar_path(output_path, task_input.task, task_input.total_tasks)
    if StoragePath(sidecar).exists():
        counters.pipeline.update_counter("datakit_length_partition/tasks_resumed", 1)
        return sidecar

    attempt = uuid.uuid4().hex
    written = [
        shard
        for leaf in task_input.leaves
        for shard in _partition_leaf(
            leaf,
            output_path=output_path,
            task=task_input.task,
            total_tasks=task_input.total_tasks,
            attempt=attempt,
        )
    ]
    counters.pipeline.update_counter("datakit_length_partition/tasks_written", 1)
    counters.pipeline.update_counter("datakit_length_partition/docs_out", sum(shard.rows for shard in written))
    counters.pipeline.update_counter("datakit_length_partition/tokens_out", sum(shard.tokens for shard in written))
    _write_sidecar(sidecar, written)
    return sidecar


def _merge_buckets(shards: list[_WrittenShard], output_path: str) -> list[LengthBucketCacheStats]:
    grouped: dict[tuple[int, int, DocumentLengthBucket], list[_WrittenShard]] = defaultdict(list)
    for shard in shards:
        grouped[(shard.cluster, shard.quality, shard.length_bucket)].append(shard)

    buckets = []
    metadata = CacheMetadata.empty()
    for (cluster, quality, length_bucket), bucket_shards in sorted(grouped.items()):
        bucket_root = prefix_join(
            output_path,
            f"cluster={cluster}/quality={quality}/length={length_bucket.value}",
        )
        bucket_shards.sort(key=lambda shard: shard.path)
        paths = [shard.path for shard in bucket_shards]
        ledgers = [
            CacheLedger(
                total_num_rows=shard.rows,
                shard_rows={},
                finished_shards=[],
                field_counts={},
                metadata=metadata,
            )
            for shard in bucket_shards
        ]
        field_counts = [{INPUT_IDS: shard.tokens} for shard in bucket_shards]
        ledger = _merge_sharded_ledgers(bucket_root, paths, ledgers, field_counts, metadata)
        buckets.append(
            LengthBucketCacheStats(
                cluster_id=cluster,
                quality_bucket=quality,
                length_bucket=length_bucket,
                path=bucket_root,
                total_elements=ledger.total_num_rows,
                total_tokens=ledger.field_counts[INPUT_IDS],
                n_shards=len(bucket_shards),
            )
        )
    return buckets


def partition_store_by_length(config: LengthPartitionConfig) -> LengthPartitionedStoreData:
    source = read_artifact(config.source_path, ClusteredStoreData)
    leaf_groups = _source_leaf_groups(source)
    tasks = [
        _PartitionTask(leaves=leaves, task=task, total_tasks=len(leaf_groups)) for task, leaves in enumerate(leaf_groups)
    ]
    context = ZephyrContext(
        resources=config.worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
        max_workers=min(config.max_workers, len(tasks)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=config.output_path),
        name="datakit-length-partition",
    )
    outcome = context.execute(
        Dataset.from_list(tasks).map(partial(_partition_task, output_path=config.output_path)),
        verbose=True,
        map_task_resources=config.worker_resources,
    )
    sidecars = [str(path) for path in outcome.results]
    with ThreadPoolExecutor(max_workers=min(64, len(sidecars))) as executor:
        written = [shard for shards in executor.map(_load_sidecar, sidecars) for shard in shards]
    buckets = _merge_buckets(written, config.output_path)
    bucket_docs = {length_bucket: 0 for length_bucket in DocumentLengthBucket}
    bucket_tokens = {length_bucket: 0 for length_bucket in DocumentLengthBucket}
    for bucket in buckets:
        bucket_docs[bucket.length_bucket] += bucket.total_elements
        bucket_tokens[bucket.length_bucket] += bucket.total_tokens
    return LengthPartitionedStoreData(
        path=config.output_path,
        source_store=config.source_path,
        cluster_view=source.cluster_view,
        split=source.split,
        length_threshold=DOCUMENT_LENGTH_THRESHOLD,
        buckets=buckets,
        source_names=source.source_names,
        tokenizer=source.tokenizer,
        counters={
            "docs_lte_64k": bucket_docs[DocumentLengthBucket.LTE_64K],
            "docs_gt_64k": bucket_docs[DocumentLengthBucket.GT_64K],
            "tokens_lte_64k": bucket_tokens[DocumentLengthBucket.LTE_64K],
            "tokens_gt_64k": bucket_tokens[DocumentLengthBucket.GT_64K],
        },
    )


SOURCE_STORE = ArtifactStep.adopt(
    "datakit/store/june-67b-a2b",
    "2026.08.24",
    source=SOURCE_STORE_PATH,
)


def build() -> ArtifactStep[LengthPartitionedStoreData]:
    def build_config(ctx: StepContext) -> LengthPartitionConfig:
        return LengthPartitionConfig(
            source_path=ctx.artifact_path(SOURCE_STORE),
            output_path=ctx.output_path,
            worker_resources=ctx.runtime_arg("worker_resources"),
            max_workers=ctx.runtime_arg("max_workers"),
        )

    return ArtifactStep(
        name="datakit/store/june-67b-a2b-length64k",
        version="2026.08.24",
        artifact_type=LengthPartitionedStoreData,
        run=partition_store_by_length,
        build_config=build_config,
        deps=(SOURCE_STORE,),
        runtime_args={
            "worker_resources": ResourceConfig(cpu=2, ram="16g", disk="16g", preemptible=False),
            "max_workers": 512,
        },
    )


if __name__ == "__main__":
    experiment_main(build)()
