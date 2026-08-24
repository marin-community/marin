# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a clustered token store with an added document-length bucket."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from enum import StrEnum
from functools import partial
from typing import TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.store.cache import CacheLedger, CacheMetadata, _merge_sharded_ledgers
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import read_artifact, write_artifact
from pydantic import BaseModel
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, format_shard_path

from experiments.datakit.store.bucket_writer import BucketSpillRun, write_bucket_cache_from_spills
from experiments.datakit.store.datakit_store import (
    DEFAULT_PARALLEL_BUCKET_WRITES,
    DEFAULT_PARTITION_PROCESSES,
)

DOCUMENT_LENGTH_THRESHOLD = 65_536
SOURCE_STORE_PATH = "gs://marin-us-central2/datakit/store_8ac06c74"
OUTPUT_PATH = "gs://marin-us-central2/datakit/store/june-67b-a2b-length64k/2026.08.24"
QUALITY_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)
SPLIT = "train"
CLUSTER_VIEW = 40
SHARDS_PER_TASK = 16
MAX_WORKERS = 512


class _TokenizedAttrData(BaseModel):
    output_dirs: dict[str, str]
    source_main_dirs: dict[str, str]
    tokenizer: str


class _DeconAttributes(BaseModel):
    output_dir: str


class _AssignmentAttrData(BaseModel):
    output_dir: str
    source_main_dir: str
    k_train: int
    k_views: list[int]


class _QualityOutput(BaseModel):
    output_dir: str


class _FuzzyDupsPerSource(BaseModel):
    attr_dir: str


class _FuzzyDupsAttrData(BaseModel):
    sources: dict[str, _FuzzyDupsPerSource]


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
    quality_thresholds: list[float]
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


@dataclasses.dataclass
class _FilterStats:
    records_in: int = 0
    contaminated_dropped: int = 0
    dedup_noncanonical_dropped: int = 0
    records_out: int = 0

    def counters(self) -> dict[str, int]:
        return {
            "datakit_store/records_in": self.records_in,
            "datakit_store/contaminated_dropped": self.contaminated_dropped,
            "datakit_store/dedup_noncanonical_dropped": self.dedup_noncanonical_dropped,
            "datakit_store/records_out": self.records_out,
        }


@dataclasses.dataclass(frozen=True)
class _SurvivingDocument:
    cluster: int
    quality: int
    input_ids: np.ndarray


def _per_source_shard_tuples(
    *,
    source_name: str,
    tokenize: _TokenizedAttrData,
    decontam: _DeconAttributes,
    cluster_assign: _AssignmentAttrData,
    quality: _QualityOutput,
    dedup_attr_dir: str,
    split: str,
) -> list[dict[str, str]]:
    tokenize_dir = tokenize.output_dirs.get(split)
    if tokenize_dir is None:
        raise FileNotFoundError(f"{source_name}: tokenize has no split={split!r}")
    tokenize_shards = sorted(str(path) for path in StoragePath(f"{tokenize_dir.rstrip('/')}/*.parquet").glob())
    if not tokenize_shards:
        raise FileNotFoundError(f"{source_name}: no tokenize shards under {tokenize_dir}")

    return [
        {
            "tokenize": tokenize_path,
            "decontam": f"{decontam.output_dir.rstrip('/')}/{os.path.basename(tokenize_path)}",
            "cluster": f"{cluster_assign.output_dir.rstrip('/')}/{os.path.basename(tokenize_path)}",
            "quality": (
                f"{quality.output_dir.rstrip('/')}/" f"{os.path.basename(tokenize_path).replace('part-', 'data-', 1)}"
            ),
            "dedup": f"{dedup_attr_dir.rstrip('/')}/{os.path.basename(tokenize_path)}",
            "source_name": source_name,
            "basename": os.path.basename(tokenize_path),
        }
        for tokenize_path in tokenize_shards
    ]


def _read_columns(path: str, columns: list[str]) -> pa.Table:
    with StoragePath(path).open("rb") as stream:
        return pq.read_table(stream, columns=columns)


def _load_decontam_table(path: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", "attributes"])
    ids = table.column("id").combine_chunks()
    contaminated = np.asarray(table.column("attributes").combine_chunks().field("contaminated"), dtype=bool)
    return ids, contaminated


def _load_cluster_table(path: str, cluster_col: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", cluster_col])
    return table.column("id").combine_chunks(), np.asarray(table.column(cluster_col), dtype=np.int32)


def _load_quality_table(path: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", "score"])
    return table.column("id").combine_chunks(), np.asarray(table.column("score"), dtype=np.float64)


def _load_dedup_canonical(path: str) -> dict[str, bool]:
    if not StoragePath(path).exists():
        return {}
    with StoragePath(path).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows == 0:
            return {}
        table = parquet.read(columns=["id", "attributes"])
    ids = table.column("id").to_pylist()
    canonical = table.column("attributes").combine_chunks().field("is_cluster_canonical").to_pylist()
    return dict(zip(ids, canonical, strict=True))


def _quality_bucket(score: float) -> int:
    return sum(score >= threshold for threshold in QUALITY_THRESHOLDS)


def _iter_surviving_docs(
    spec: dict[str, str],
    cluster_col: str,
    *,
    stats: _FilterStats,
) -> Iterator[_SurvivingDocument]:
    decontam_ids, contaminated = _load_decontam_table(spec["decontam"])
    cluster_ids, cluster_values = _load_cluster_table(spec["cluster"], cluster_col)
    quality_ids, scores = _load_quality_table(spec["quality"])
    row_counts = (len(decontam_ids), len(cluster_ids), len(quality_ids))
    if len(set(row_counts)) != 1:
        raise RuntimeError(
            f"{spec['source_name']}/{spec['basename']}: dense-table row count mismatch "
            f"(decontam={row_counts[0]}, cluster={row_counts[1]}, quality={row_counts[2]})"
        )
    if not pc.all(pc.equal(decontam_ids, cluster_ids)).as_py():
        raise RuntimeError(f"{spec['source_name']}/{spec['basename']}: decontam/cluster id mismatch")
    if not pc.all(pc.equal(decontam_ids, quality_ids)).as_py():
        raise RuntimeError(f"{spec['source_name']}/{spec['basename']}: decontam/quality id mismatch")
    expected_ids = decontam_ids.to_pylist()
    dedup_canonical = _load_dedup_canonical(spec["dedup"])

    with StoragePath(spec["tokenize"]).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=8192, columns=["id", "input_ids"]):
            document_ids = batch.column("id").to_pylist()
            input_ids = batch.column("input_ids")
            for index, document_id in enumerate(document_ids):
                if row_index >= len(expected_ids) or document_id != expected_ids[row_index]:
                    raise RuntimeError(
                        f"{spec['source_name']}/{spec['basename']}: tokenize/decontam id mismatch "
                        f"at document {row_index}"
                    )
                stats.records_in += 1
                if contaminated[row_index]:
                    stats.contaminated_dropped += 1
                elif dedup_canonical.get(document_id) is False:
                    stats.dedup_noncanonical_dropped += 1
                else:
                    stats.records_out += 1
                    yield _SurvivingDocument(
                        cluster=int(cluster_values[row_index]),
                        quality=_quality_bucket(float(scores[row_index])),
                        input_ids=input_ids[index].values.to_numpy(),
                    )
                row_index += 1
    if row_index != len(expected_ids):
        raise RuntimeError(
            f"{spec['source_name']}/{spec['basename']}: tokenize rows ({row_index}) "
            f"!= decontam rows ({len(expected_ids)})"
        )


def _validate_cluster_view(cluster_assign: dict[str, _AssignmentAttrData], cluster_view: int) -> str:
    for source_name, assignment in cluster_assign.items():
        valid_views = {assignment.k_train, *assignment.k_views}
        if cluster_view not in valid_views:
            raise ValueError(f"cluster_view={cluster_view} not in {source_name}'s views {sorted(valid_views)}")
    return f"cluster_{cluster_view}"


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
    paths: dict[tuple[int, int, DocumentLengthBucket], tuple[str, str]] = {}
    rows: dict[tuple[int, int, DocumentLengthBucket], int] = defaultdict(int)
    tokens: dict[tuple[int, int, DocumentLengthBucket], int] = defaultdict(int)
    stats = _FilterStats()
    streams = {}
    with ExitStack() as stack:
        for document in _iter_surviving_docs(spec, cluster_col, stats=stats):
            input_ids = np.asarray(document.input_ids, dtype=np.int32)
            length_bucket = _document_length_bucket(len(input_ids))
            key = (document.cluster, document.quality, length_bucket)
            if key not in streams:
                cluster, quality, bucket = key
                stem = os.path.join(shard_dir, f"cluster-{cluster}-quality-{quality}-length-{bucket}")
                data_path = f"{stem}.tokens.i32"
                lengths_path = f"{stem}.lengths.i64"
                paths[key] = (data_path, lengths_path)
                streams[key] = (
                    stack.enter_context(open(data_path, "wb")),
                    stack.enter_context(open(lengths_path, "wb")),
                )
            data_stream, lengths_stream = streams[key]
            input_ids.tofile(data_stream)
            np.asarray([len(input_ids)], dtype=np.int64).tofile(lengths_stream)
            rows[key] += 1
            tokens[key] += len(input_ids)

    results = []
    for (cluster, quality, length_bucket), (data_path, lengths_path) in paths.items():
        key = (cluster, quality, length_bucket)
        results.append(
            _SpillBucketRun(
                cluster=cluster,
                quality=quality,
                length_bucket=length_bucket,
                run=BucketSpillRun(
                    data_path=data_path,
                    lengths_path=lengths_path,
                    rows=rows[key],
                    tokens=tokens[key],
                ),
            )
        )
    return _SpillShardResult(buckets=results, stats=stats, tokens=sum(tokens.values()))


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
    tokenize: dict[str, _TokenizedAttrData],
    decontam: dict[str, _DeconAttributes],
    cluster_assign: dict[str, _AssignmentAttrData],
    quality: dict[str, _QualityOutput],
    dedup: _FuzzyDupsAttrData,
    output_path: str,
    cluster_view: int = 40,
    split: str = "train",
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 4096,
    shards_per_task: int = 1,
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
    if shards_per_task < 1:
        raise ValueError(f"shards_per_task must be >= 1, got {shards_per_task}")
    if max_parallel_bucket_writes < 1:
        raise ValueError(f"max_parallel_bucket_writes must be >= 1, got {max_parallel_bucket_writes}")
    if partition_processes < 1:
        raise ValueError(f"partition_processes must be >= 1, got {partition_processes}")

    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)

    source_main_dirs = {}
    for source_name, tokenized in tokenize.items():
        source_main_dir = tokenized.source_main_dirs.get(split)
        if source_main_dir is None:
            raise ValueError(f"{source_name}: tokenize has no source_main_dir for split={split!r}")
        source_main_dirs[source_name] = source_main_dir

    for source_name, assignment in cluster_assign.items():
        source_main_dir = source_main_dirs[source_name]
        if assignment.source_main_dir != source_main_dir:
            raise ValueError(
                f"{source_name}: cluster_assign.source_main_dir={assignment.source_main_dir!r} "
                f"!= tokenize.source_main_dirs[{split!r}]={source_main_dir!r}"
            )

    def resolve_source_shards(source_name: str) -> list[dict[str, str]]:
        source_main_dir = source_main_dirs[source_name]
        dedup_source = dedup.sources.get(source_main_dir)
        if dedup_source is None:
            raise KeyError(f"{source_name}: dedup has no source {source_main_dir!r}")
        return _per_source_shard_tuples(
            source_name=source_name,
            tokenize=tokenize[source_name],
            decontam=decontam[source_name],
            cluster_assign=cluster_assign[source_name],
            quality=quality[source_name],
            dedup_attr_dir=dedup_source.attr_dir,
            split=split,
        )

    source_names = sorted(tokenize)
    with ThreadPoolExecutor(max_workers=min(32, len(source_names))) as executor:
        per_source_specs = executor.map(resolve_source_shards, source_names)
        shard_specs = [spec for source_specs in per_source_specs for spec in source_specs]
    if not shard_specs:
        raise ValueError("No input shards resolved -- nothing to do")

    batched_specs = [
        shard_specs[index : index + shards_per_task] for index in range(0, len(shard_specs), shards_per_task)
    ]
    resolved_task_count = len(batched_specs)
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
        quality_thresholds=list(QUALITY_THRESHOLDS),
        split=split,
        length_threshold=DOCUMENT_LENGTH_THRESHOLD,
        buckets=buckets,
        source_names=source_names,
        tokenizer=next(iter(tokenize.values())).tokenizer,
        counters=artifact_counters,
    )
    write_artifact(artifact, output_path)
    return artifact


def _source_stage_paths() -> tuple[dict[str, dict[str, str]], str]:
    executor_info = json.loads(StoragePath(prefix_join(SOURCE_STORE_PATH, ".executor_info")).read_text())
    dependencies = executor_info["dependencies"]
    stages: dict[str, dict[str, str]] = {stage: {} for stage in ("tokenize", "decontam", "cluster_assign", "quality")}
    dedup_path = ""
    for path in dependencies:
        if path.startswith(f"{SOURCE_STORE_PATH.rsplit('/', 1)[0]}/dedup_"):
            dedup_path = path
            continue
        for stage in stages:
            prefix = f"gs://marin-us-central2/datakit/{stage}/"
            if not path.startswith(prefix):
                continue
            source_and_hash = path.removeprefix(prefix)
            source_name, _, fingerprint = source_and_hash.rpartition("_")
            if not source_name or len(fingerprint) != 8:
                raise ValueError(f"Cannot parse {stage} dependency {path!r}")
            stages[stage][source_name] = path
            break
    source_names = set(stages["tokenize"])
    for stage, paths in stages.items():
        if set(paths) != source_names:
            raise ValueError(f"{stage} source set does not match tokenize")
    if not dedup_path:
        raise ValueError("Source store has no dedup dependency")
    return {name: {stage: paths[name] for stage, paths in stages.items()} for name in sorted(source_names)}, dedup_path


def _read_source_artifacts():
    source_paths, dedup_path = _source_stage_paths()

    def read_source(item: tuple[str, dict[str, str]]):
        source_name, paths = item
        return source_name, (
            read_artifact(paths["tokenize"], _TokenizedAttrData),
            read_artifact(paths["decontam"], _DeconAttributes),
            read_artifact(paths["cluster_assign"], _AssignmentAttrData),
            read_artifact(paths["quality"], _QualityOutput),
        )

    with ThreadPoolExecutor(max_workers=32) as executor:
        artifacts = dict(executor.map(read_source, source_paths.items()))
    return (
        {name: source[0] for name, source in artifacts.items()},
        {name: source[1] for name, source in artifacts.items()},
        {name: source[2] for name, source in artifacts.items()},
        {name: source[3] for name, source in artifacts.items()},
        read_artifact(dedup_path, _FuzzyDupsAttrData),
    )


def main() -> None:
    tokenize, decontam, cluster_assign, quality, dedup = _read_source_artifacts()
    build_length_partitioned_store(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        dedup=dedup,
        output_path=OUTPUT_PATH,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="32g", preemptible=False),
        max_workers=MAX_WORKERS,
        shards_per_task=SHARDS_PER_TASK,
    )


if __name__ == "__main__":
    main()
