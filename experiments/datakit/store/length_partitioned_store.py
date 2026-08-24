# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a clustered token store with an added document-length bucket."""

import dataclasses
import json
import multiprocessing
import os
import tempfile
import time
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from enum import StrEnum
from functools import partial
from typing import TypedDict

import numpy as np
from fray.types import ResourceConfig
from levanter.store.cache import CacheLedger, CacheMetadata, _merge_sharded_ledgers
from marin.datakit.decon import DeconAttributes
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsAttrData,
)
from marin.processing.tokenize.attributes import TokenizedAttrData
from pydantic import BaseModel
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, format_shard_path

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.global_exact_dedup import GlobalExactDedupData
from experiments.datakit.store.bucket_writer import BucketSpillRun, write_bucket_cache_from_spills
from experiments.datakit.store.datakit_store import (
    DEFAULT_PARALLEL_BUCKET_WRITES,
    DEFAULT_PARTITION_PROCESSES,
    _FilterStats,
    _iter_surviving_docs,
    _per_source_shard_tuples,
    _resolve_dedup_attr_dir,
    _validate_cluster_view,
)

DOCUMENT_LENGTH_THRESHOLD = 65_536


class DocumentLengthBucket(StrEnum):
    LTE_64K = "lte_64k"
    GT_64K = "gt_64k"


class LengthBucketCacheStats(BaseModel):
    cluster_id: int
    quality_bucket: int
    length_bucket: DocumentLengthBucket
    path: DatakitArtifactPath
    total_elements: int
    total_tokens: int
    n_shards: int


class LengthPartitionedStoreData(BaseModel):
    version: str = "v1"
    cache_path: DatakitArtifactPath
    cluster_view: int
    bucket_edges: list[float]
    split: str
    length_threshold: int
    buckets: list[LengthBucketCacheStats]
    source_names: list[str]
    tokenizer: str
    counters: dict[str, int | float]


@dataclasses.dataclass(frozen=True)
class _TaskBucketStat:
    cluster: int
    quality: int
    length_bucket: DocumentLengthBucket
    task: int
    path: str
    rows: int
    tokens: int


@dataclasses.dataclass(frozen=True)
class _TaskCompletion:
    buckets: list[_TaskBucketStat]
    counters: dict[str, int | float]


class _StoreTask(TypedDict):
    specs: list[dict[str, str]]
    task: int
    total_tasks: int


class _TaskConfirmation(TypedDict):
    sidecar_path: str
    task: int
    resumed: int


@dataclasses.dataclass(frozen=True)
class _SpillBucketRun:
    cluster: int
    quality: int
    length_bucket: DocumentLengthBucket
    run: BucketSpillRun


@dataclasses.dataclass(frozen=True)
class _SpillShardResult:
    buckets: list[_SpillBucketRun]
    stats: _FilterStats
    tokens: int


def _document_length_bucket(length: int) -> DocumentLengthBucket:
    if length > DOCUMENT_LENGTH_THRESHOLD:
        return DocumentLengthBucket.GT_64K
    return DocumentLengthBucket.LTE_64K


def _spill_source_shard(
    spec: dict[str, str],
    cluster_col: str,
    scratch_dir: str,
    shard_index: int,
) -> _SpillShardResult:
    shard_dir = os.path.join(scratch_dir, f"shard-{shard_index:05d}")
    os.makedirs(shard_dir)
    buckets: dict[tuple[int, int, DocumentLengthBucket], list[np.ndarray]] = defaultdict(list)
    stats = _FilterStats()
    for document in _iter_surviving_docs(spec, cluster_col, stats=stats):
        length_bucket = _document_length_bucket(len(document.input_ids))
        buckets[(document.cluster, document.quality, length_bucket)].append(
            np.asarray(document.input_ids, dtype=np.int32)
        )

    results = []
    total_tokens = 0
    for (cluster, quality, length_bucket), documents in buckets.items():
        stem = os.path.join(shard_dir, f"cluster-{cluster}-quality-{quality}-length-{length_bucket}")
        data_path = f"{stem}.tokens.i32"
        lengths_path = f"{stem}.lengths.i64"
        lengths = np.fromiter((len(document) for document in documents), dtype=np.int64, count=len(documents))
        tokens = int(lengths.sum())
        with open(data_path, "wb") as stream:
            for document in documents:
                document.tofile(stream)
        lengths.tofile(lengths_path)
        total_tokens += tokens
        results.append(
            _SpillBucketRun(
                cluster=cluster,
                quality=quality,
                length_bucket=length_bucket,
                run=BucketSpillRun(
                    data_path=data_path,
                    lengths_path=lengths_path,
                    rows=len(documents),
                    tokens=tokens,
                ),
            )
        )
    return _SpillShardResult(buckets=results, stats=stats, tokens=total_tokens)


def _task_sidecar_path(output_path: str, task: int, total_tasks: int) -> str:
    pattern = prefix_join(output_path, "_done/shard-{shard:05d}-of-{total:05d}.json")
    return format_shard_path(pattern, task, total_tasks)


def _write_task_sidecar(path: str, completion: _TaskCompletion) -> None:
    payload = json.dumps(dataclasses.asdict(completion), sort_keys=True)
    with atomic_rename(path) as temporary_path:
        StoragePath(temporary_path).write_text(payload)


def _load_task_sidecar(path: str) -> _TaskCompletion:
    payload = json.loads(StoragePath(path).read_text())
    return _TaskCompletion(
        buckets=[
            _TaskBucketStat(
                cluster=item["cluster"],
                quality=item["quality"],
                length_bucket=DocumentLengthBucket(item["length_bucket"]),
                task=item["task"],
                path=item["path"],
                rows=item["rows"],
                tokens=item["tokens"],
            )
            for item in payload["buckets"]
        ],
        counters=payload["counters"],
    )


def _partition_and_write_task(
    task_input: _StoreTask,
    *,
    cluster_col: str,
    output_path: str,
    max_parallel_bucket_writes: int,
    partition_processes: int,
) -> _TaskConfirmation:
    task = task_input["task"]
    total_tasks = task_input["total_tasks"]
    batch_specs = task_input["specs"]
    sidecar_path = _task_sidecar_path(output_path, task, total_tasks)
    if StoragePath(sidecar_path).exists():
        counters.pipeline.update_counter("datakit_length_store/tasks_resumed", 1)
        return {"sidecar_path": sidecar_path, "task": task, "resumed": 1}

    initial_counters = counters.pipeline.get_counters()
    spill_runs: dict[tuple[int, int, DocumentLengthBucket], list[BucketSpillRun]] = defaultdict(list)
    bucket_tokens: dict[tuple[int, int, DocumentLengthBucket], int] = defaultdict(int)
    n_tokens = 0
    partition_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="datakit-length-store-") as scratch_dir:
        with ProcessPoolExecutor(
            max_workers=min(partition_processes, len(batch_specs)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            results = executor.map(
                _spill_source_shard,
                batch_specs,
                [cluster_col] * len(batch_specs),
                [scratch_dir] * len(batch_specs),
                range(len(batch_specs)),
            )
            for result in results:
                n_tokens += result.tokens
                for name, value in result.stats.counters().items():
                    counters.pipeline.update_counter(name.replace("datakit_store/", "datakit_length_store/"), value)
                for bucket in result.buckets:
                    key = (bucket.cluster, bucket.quality, bucket.length_bucket)
                    spill_runs[key].append(bucket.run)
                    bucket_tokens[key] += bucket.run.tokens
        partition_seconds = time.monotonic() - partition_started

        attempt = uuid.uuid4().hex

        def write_one(key: tuple[int, int, DocumentLengthBucket]) -> _TaskBucketStat:
            cluster, quality, length_bucket = key
            bucket_root = prefix_join(
                output_path,
                f"cluster={cluster}/quality={quality}/length={length_bucket}",
            )
            pattern = prefix_join(bucket_root, f"part-{{shard:05d}}-of-{{total:05d}}-attempt-{attempt}")
            cache_path = format_shard_path(pattern, task, total_tasks)
            ledger = write_bucket_cache_from_spills(cache_path, spill_runs[key])
            return _TaskBucketStat(
                cluster=cluster,
                quality=quality,
                length_bucket=length_bucket,
                task=task,
                path=cache_path,
                rows=ledger.total_num_rows,
                tokens=bucket_tokens[key],
            )

        ordered_keys = sorted(spill_runs, key=lambda key: bucket_tokens[key], reverse=True)
        bucket_stats = []
        write_started = time.monotonic()
        if ordered_keys:
            with ThreadPoolExecutor(max_workers=min(max_parallel_bucket_writes, len(ordered_keys))) as executor:
                futures = [executor.submit(write_one, key) for key in ordered_keys]
                for future in as_completed(futures):
                    bucket_stats.append(future.result())
        write_seconds = time.monotonic() - write_started
    counters.pipeline.update_counter("datakit_length_store/tokens_out", n_tokens)
    counters.pipeline.update_counter("datakit_length_store/tasks_written", 1)
    counters.pipeline.update_counter("datakit_length_store/bucket_caches_written", len(bucket_stats))
    for name, value in (
        ("datakit_length_store/partition_seconds_max", partition_seconds),
        ("datakit_length_store/write_seconds_max", write_seconds),
        ("datakit_length_store/task_seconds_max", partition_seconds + write_seconds),
    ):
        counters.pipeline.set_aggregation(name, counters.Aggregation.MAX)
        counters.pipeline.update_counter(name, value)
    final_counters = counters.pipeline.get_counters()
    task_counters = {
        name: value - initial_counters.get(name, 0)
        for name, value in final_counters.items()
        if name.startswith("datakit_length_store/") and value != initial_counters.get(name, 0)
    }
    bucket_stats.sort(key=lambda stat: (stat.cluster, stat.quality, stat.length_bucket))
    _write_task_sidecar(sidecar_path, _TaskCompletion(buckets=bucket_stats, counters=task_counters))
    return {"sidecar_path": sidecar_path, "task": task, "resumed": 0}


def _merge_buckets(task_bucket_stats: list[_TaskBucketStat], output_path: str) -> list[LengthBucketCacheStats]:
    grouped: dict[tuple[int, int, DocumentLengthBucket], list[_TaskBucketStat]] = defaultdict(list)
    for stat in task_bucket_stats:
        grouped[(stat.cluster, stat.quality, stat.length_bucket)].append(stat)

    metadata = CacheMetadata.empty()
    buckets = []
    for (cluster, quality, length_bucket), task_parts in sorted(grouped.items()):
        bucket_root = prefix_join(
            output_path,
            f"cluster={cluster}/quality={quality}/length={length_bucket}",
        )
        task_parts.sort(key=lambda stat: (stat.task, stat.path))
        paths = [stat.path for stat in task_parts]
        ledgers = [
            CacheLedger(
                total_num_rows=stat.rows,
                shard_rows={},
                finished_shards=[],
                field_counts={},
                metadata=metadata,
            )
            for stat in task_parts
        ]
        field_counts = [{"input_ids": stat.tokens} for stat in task_parts]
        ledger = _merge_sharded_ledgers(bucket_root, paths, ledgers, field_counts, metadata)
        buckets.append(
            LengthBucketCacheStats(
                cluster_id=cluster,
                quality_bucket=quality,
                length_bucket=length_bucket,
                path=bucket_root,
                total_elements=ledger.total_num_rows,
                total_tokens=ledger.field_counts["input_ids"],
                n_shards=len(task_parts),
            )
        )
    return buckets


def build_length_partitioned_store(
    *,
    tokenize: dict[str, TokenizedAttrData],
    decontam: dict[str, DeconAttributes],
    cluster_assign: dict[str, AssignmentAttrData],
    quality: dict[str, QualityScores],
    exact_dedup: GlobalExactDedupData,
    dedup: VerifiedFuzzyDupsAttrData,
    output_path: str,
    cluster_view: int = 40,
    split: str = "train",
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 4096,
    task_count: int | None = None,
    max_parallel_bucket_writes: int = DEFAULT_PARALLEL_BUCKET_WRITES,
    partition_processes: int = DEFAULT_PARTITION_PROCESSES,
) -> LengthPartitionedStoreData:
    """Build caches for each populated ``(cluster, quality, document length)`` bucket."""
    if not tokenize:
        raise ValueError("build_length_partitioned_store: tokenize is empty")
    for label, artifacts in (
        ("decontam", decontam),
        ("cluster_assign", cluster_assign),
        ("quality", quality),
    ):
        if set(artifacts) != set(tokenize):
            missing = sorted(set(tokenize) - set(artifacts))
            extra = sorted(set(artifacts) - set(tokenize))
            raise ValueError(f"{label} source set must equal tokenize: missing={missing!r}, extra={extra!r}")
    if task_count is not None and task_count < 1:
        raise ValueError(f"task_count must be >= 1, got {task_count}")
    if max_parallel_bucket_writes < 1:
        raise ValueError(f"max_parallel_bucket_writes must be >= 1, got {max_parallel_bucket_writes}")
    if partition_processes < 1:
        raise ValueError(f"partition_processes must be >= 1, got {partition_processes}")

    models = {(item.model_dir, item.calib_file, tuple(item.bucket_edges)) for item in quality.values()}
    if len(models) != 1:
        raise ValueError(f"build_length_partitioned_store: sources span multiple quality models: {sorted(models)}")
    bucket_edges = next(iter(quality.values())).bucket_edges
    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)

    source_keys = {}
    for source_name, tokenized in tokenize.items():
        source_key = tokenized.source_keys.get(split)
        if source_key is None:
            raise ValueError(f"{source_name}: tokenize has no source_key for split={split!r}")
        source_keys[source_name] = source_key
    expected_source_keys = set(source_keys.values())
    if len(expected_source_keys) != len(source_keys):
        raise ValueError(f"tokenize sources must use unique source keys for split={split!r}")
    for label, sources in (("exact_dedup", exact_dedup.sources), ("dedup", dedup.sources)):
        if set(sources) != expected_source_keys:
            missing = sorted(expected_source_keys - set(sources))
            extra = sorted(set(sources) - expected_source_keys)
            raise ValueError(f"{label} source set must equal tokenize source keys: missing={missing!r}, extra={extra!r}")

    for source_name, assignment in cluster_assign.items():
        source_key = source_keys[source_name]
        if assignment.source_key != source_key:
            raise ValueError(
                f"{source_name}: cluster_assign.source_key={assignment.source_key!r} "
                f"!= tokenize.source_keys[{split!r}]={source_key!r}"
            )

    def resolve_source_shards(source_name: str) -> list[dict[str, str]]:
        source_key = source_keys[source_name]
        dedup_attr_dir = _resolve_dedup_attr_dir(
            source_name=source_name,
            source_key=source_key,
            sources=dedup.sources,
            label="dedup",
        )
        exact_dedup_attr_dir = _resolve_dedup_attr_dir(
            source_name=source_name,
            source_key=source_key,
            sources=exact_dedup.sources,
            label="exact_dedup",
        )
        return _per_source_shard_tuples(
            source_name=source_name,
            tokenize=tokenize[source_name],
            decontam=decontam[source_name],
            cluster_assign=cluster_assign[source_name],
            quality=quality[source_name],
            exact_dedup_attr_dir=exact_dedup_attr_dir,
            dedup_attr_dir=dedup_attr_dir,
            split=split,
        )

    source_names = sorted(tokenize)
    with ThreadPoolExecutor(max_workers=min(32, len(source_names))) as executor:
        per_source_specs = executor.map(resolve_source_shards, source_names)
        shard_specs = [spec for source_specs in per_source_specs for spec in source_specs]
    if not shard_specs:
        raise ValueError("No input shards resolved -- nothing to do")

    resolved_task_count = len(shard_specs) if task_count is None else min(task_count, len(shard_specs))
    batched_specs = [shard_specs[index::resolved_task_count] for index in range(resolved_task_count)]
    if worker_resources is None:
        worker_resources = ResourceConfig(cpu=2, ram="16g", disk="16g")

    context = ZephyrContext(
        resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
        max_workers=min(max_workers, len(batched_specs)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=output_path),
        name="datakit-length-partitioned-store",
    )
    write_task = partial(
        _partition_and_write_task,
        cluster_col=cluster_col,
        output_path=output_path,
        max_parallel_bucket_writes=max_parallel_bucket_writes,
        partition_processes=partition_processes,
    )
    tasks: list[_StoreTask] = [
        {"specs": specs, "task": task, "total_tasks": resolved_task_count} for task, specs in enumerate(batched_specs)
    ]
    outcome = context.execute(
        Dataset.from_list(tasks).map(write_task),
        verbose=True,
        map_task_resources=worker_resources,
    )
    confirmations = [result for result in outcome.results if result is not None]
    sidecar_paths = [str(result["sidecar_path"]) for result in confirmations]
    with ThreadPoolExecutor(max_workers=min(64, len(sidecar_paths))) as executor:
        completions = list(executor.map(_load_task_sidecar, sidecar_paths))
    buckets = _merge_buckets(
        [bucket for completion in completions for bucket in completion.buckets],
        output_path,
    )

    durable_counters: dict[str, int | float] = defaultdict(int)
    for completion in completions:
        for name, value in completion.counters.items():
            if name.endswith("_max"):
                durable_counters[name] = max(durable_counters[name], value)
            else:
                durable_counters[name] += value
    artifact_counters = dict(outcome.counters)
    artifact_counters.update(durable_counters)
    for length_bucket in DocumentLengthBucket:
        suffix = length_bucket.value
        artifact_counters[f"datakit_length_store/docs_{suffix}"] = sum(
            bucket.total_elements for bucket in buckets if bucket.length_bucket == length_bucket
        )
        artifact_counters[f"datakit_length_store/tokens_{suffix}"] = sum(
            bucket.total_tokens for bucket in buckets if bucket.length_bucket == length_bucket
        )

    artifact = LengthPartitionedStoreData(
        cache_path=output_path,
        cluster_view=cluster_view,
        bucket_edges=bucket_edges,
        split=split,
        length_threshold=DOCUMENT_LENGTH_THRESHOLD,
        buckets=buckets,
        source_names=source_names,
        tokenizer=next(iter(tokenize.values())).tokenizer,
        counters=artifact_counters,
    )
    write_artifact(artifact, output_path)
    return artifact
