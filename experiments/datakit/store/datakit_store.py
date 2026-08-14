# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build task-partitioned per-(cluster, quality) Levanter stores.

Each Zephyr task joins and filters a batch of co-partitioned input shards,
partitions surviving token arrays into append-only local SSD runs, then writes
one materialized cache per populated bucket. Final token writes align to the
TensorStore write grid.

Task outputs live under unique attempt paths. A small sidecar is committed last;
its presence makes the whole task visible to driver-side ledger aggregation.
Retries skip completed sidecars and ignore partial attempt directories.
"""

import dataclasses
import json
import logging
import multiprocessing
import os
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.store.cache import (
    CacheLedger,
    CacheMetadata,
    _merge_sharded_ledgers,
)
from marin.datakit.decon import DeconAttributes
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsAttrData,
    VerifiedFuzzyDupsPerSource,
)
from marin.processing.tokenize._core import CHUNK_INDEX_FIELD
from marin.processing.tokenize.attributes import TokenizedAttrData
from pydantic import BaseModel
from rigging.filesystem import StoragePath, atomic_rename, marin_temp_bucket, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset, format_shard_path
from zephyr.execution import ZephyrContext

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.global_exact_dedup import ExactDupsPerSource, GlobalExactDedupData
from experiments.datakit.store.bucket_writer import BucketSpillRun, write_bucket_cache_from_spills

logger = logging.getLogger(__name__)


class BucketCacheStats(BaseModel):
    """Per-(cluster, quality) Levanter cache stats inside :class:`ClusteredStoreData`."""

    cluster_id: int
    quality_bucket: int
    path: DatakitArtifactPath
    total_elements: int
    total_tokens: int
    n_shards: int


class ClusteredStoreData(BaseModel):
    """One Levanter cache per populated (cluster, quality) bucket.

    Persisted as ``<output_path>/artifact.json``. Load via
    ``read_artifact(output_path, ClusteredStoreData)``.
    """

    version: str = "v4"
    cache_path: DatakitArtifactPath
    cluster_view: int
    bucket_edges: list[float]
    split: str
    buckets: list[BucketCacheStats]
    source_names: list[str]
    tokenizer: str
    counters: dict[str, int | float]


def _per_source_shard_tuples(
    *,
    source_name: str,
    tokenize: TokenizedAttrData,
    decontam: DeconAttributes,
    cluster_assign: AssignmentAttrData,
    quality: QualityScores,
    exact_dedup_attr_dir: str,
    dedup_attr_dir: str,
    split: str,
) -> list[dict[str, str]]:
    """Align one source's co-partitioned attribute shards by basename."""
    tok_dir = tokenize.output_dirs.get(split)
    if tok_dir is None:
        raise FileNotFoundError(f"{source_name}: tokenize has no split={split!r}")
    tok_shards = sorted(str(m) for m in StoragePath(f"{tok_dir.rstrip('/')}/*.parquet").glob())
    if not tok_shards:
        raise FileNotFoundError(f"{source_name}: no tokenize shards under {tok_dir}")

    decon_dir = decontam.main_output_dir.rstrip("/")
    cluster_dir = cluster_assign.output_dir.rstrip("/")
    quality_dir = quality.main_output_dir.rstrip("/")
    exact_dedup_dir = exact_dedup_attr_dir.rstrip("/")
    dedup_dir = dedup_attr_dir.rstrip("/")
    return [
        {
            "tokenize": tok_path,
            "decontam": f"{decon_dir}/{os.path.basename(tok_path)}",
            "cluster": f"{cluster_dir}/{os.path.basename(tok_path)}",
            "quality": f"{quality_dir}/{os.path.basename(tok_path)}",
            "exact_dedup": f"{exact_dedup_dir}/{os.path.basename(tok_path)}",
            "dedup": f"{dedup_dir}/{os.path.basename(tok_path)}",
            "source_name": source_name,
            "basename": os.path.basename(tok_path),
        }
        for tok_path in tok_shards
    ]


def _read_columns(path: str, columns: list[str]) -> pa.Table:
    """Read Parquet through fsspec for compatibility with CoreWeave object storage."""
    with StoragePath(path).open("rb") as fh:
        return pq.read_table(fh, columns=columns)


def _load_decon_table(path: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", "contaminated"])
    ids = table.column("id").combine_chunks()
    contaminated = np.asarray(table.column("contaminated"), dtype=bool)
    return ids, contaminated


def _load_cluster_table(path: str, cluster_col: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", cluster_col])
    return table.column("id").combine_chunks(), np.asarray(table.column(cluster_col), dtype=np.int32)


def _load_quality_table(path: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", "quality_bucket"])
    return table.column("id").combine_chunks(), np.asarray(table.column("quality_bucket"), dtype=np.int32)


def _load_verified_duplicates(path: str) -> set[str]:
    """Return IDs that the full-text verifier marked as duplicates."""
    if not StoragePath(path).exists():
        return set()
    with StoragePath(path).open("rb") as fh:
        parquet = pq.ParquetFile(fh)
        if parquet.metadata.num_rows == 0:
            return set()
        table = parquet.read(columns=["id", "dup_doc"])
    ids = table.column("id").to_pylist()
    duplicate_flags = table.column("dup_doc")
    if duplicate_flags.null_count or pc.all(duplicate_flags).as_py() is not True:
        raise ValueError(f"{path} contains a verified fuzzy-duplicate row with dup_doc=False")
    return set(ids)


def _load_exact_duplicates(path: str) -> set[str]:
    """Return sparse exact-duplicate IDs; a missing shard has no duplicates."""
    if not StoragePath(path).exists():
        return set()
    return set(_read_columns(path, ["id"]).column("id").to_pylist())


def _validate_cluster_view(cluster_assign: dict[str, AssignmentAttrData], cluster_view: int) -> str:
    """Check that every assignment artifact materialized the selected view."""
    for name, assignment in cluster_assign.items():
        valid_views = {assignment.k_train, *assignment.k_views}
        if cluster_view not in valid_views:
            raise ValueError(
                f"cluster_view={cluster_view} not in {name}'s views "
                f"(k_train={assignment.k_train}, k_views={assignment.k_views})"
            )
    return f"cluster_{cluster_view}"


def _resolve_dedup_attr_dir(
    *,
    source_name: str,
    source_key: str,
    sources: Mapping[str, ExactDupsPerSource | VerifiedFuzzyDupsPerSource],
    label: str,
) -> str:
    entry = sources.get(source_key)
    if entry is None:
        raise KeyError(
            f"{source_name}: {label}.sources has no entry for source_key={source_key!r}. "
            f"Drop the source from the config or rebuild {label} with it included."
        )
    return entry.attr_dir


# Rows read from a tokenized shard at once during the positional join.
_TOKENIZE_BATCH_SIZE = 8192

# Bucket caches are independent. This bounds simultaneous 512 MiB write buffers
# while allowing TensorStore compression and object-store commits to overlap.
DEFAULT_PARALLEL_BUCKET_WRITES = 4
DEFAULT_PARTITION_PROCESSES = 1


@dataclasses.dataclass(frozen=True)
class _TaskBucketStat:
    """One task's materialized cache for a populated quality-topic bucket."""

    cluster: int
    quality: int
    task: int
    path: str
    rows: int
    tokens: int


@dataclasses.dataclass(frozen=True)
class _TaskCompletion:
    """Durable task result used for retry and driver-side aggregation."""

    buckets: list[_TaskBucketStat]
    counters: dict[str, int | float]


@dataclasses.dataclass
class _FilterStats:
    records_in: int = 0
    contaminated_dropped: int = 0
    exact_duplicate_dropped: int = 0
    fuzzy_duplicate_dropped: int = 0
    records_out: int = 0

    def counters(self) -> dict[str, int]:
        return {
            "datakit_store/records_in": self.records_in,
            "datakit_store/contaminated_dropped": self.contaminated_dropped,
            "datakit_store/exact_duplicate_dropped": self.exact_duplicate_dropped,
            "datakit_store/fuzzy_duplicate_dropped": self.fuzzy_duplicate_dropped,
            "datakit_store/records_out": self.records_out,
        }


@dataclasses.dataclass(frozen=True)
class _SurvivingDocument:
    cluster: int
    quality: int
    input_ids: np.ndarray


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
    run: BucketSpillRun


@dataclasses.dataclass(frozen=True)
class _SpillShardResult:
    buckets: list[_SpillBucketRun]
    stats: _FilterStats
    tokens: int


# ---------------------------------------------------------------------------
# Map side: join + filter -> task-local bucket partitions.
# ---------------------------------------------------------------------------


def _iter_tokenized_documents(path: str) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(doc_id, input_ids)`` per document from one tokenized shard.

    A document above the token limit of one Parquet row occupies several adjacent
    rows that share its ``id``, ordered by ``chunk_index`` (see
    :mod:`marin.processing.tokenize.attributes`). Those rows are joined back into
    one token array here, so this shard yields one document per source document
    and the positional join against the dense per-document tables holds.

    ``chunk_index == 0`` marks the first row of a document. A rule that instead
    started a document on a change of ``id`` would merge two adjacent documents
    that share an id, which some sources produce.

    Raises ``RuntimeError`` on a shard with no ``chunk_index`` column (written
    before the column existed) and on rows that do not run 0, 1, 2 ... within one
    id. Concatenating out-of-order rows would corrupt the token stream silently.
    """
    with StoragePath(path).open("rb") as fh:
        parquet = pq.ParquetFile(fh)
        if CHUNK_INDEX_FIELD not in parquet.schema_arrow.names:
            raise RuntimeError(
                f"{path}: tokenize shard has no {CHUNK_INDEX_FIELD} column. It predates the column, "
                "so its step identity does not match this code. Re-run tokenize for this source."
            )
        doc_id: str | None = None
        chunks: list[np.ndarray] = []
        for batch in parquet.iter_batches(
            batch_size=_TOKENIZE_BATCH_SIZE, columns=["id", CHUNK_INDEX_FIELD, "input_ids"]
        ):
            row_ids = batch.column("id").to_pylist()
            chunk_indices = batch.column(CHUNK_INDEX_FIELD).to_pylist()
            input_ids = batch.column("input_ids")
            for i, row_id in enumerate(row_ids):
                if chunk_indices[i] == 0:
                    if chunks:
                        assert doc_id is not None
                        yield doc_id, chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
                    doc_id, chunks = row_id, []
                elif row_id != doc_id or chunk_indices[i] != len(chunks):
                    raise RuntimeError(
                        f"{path}: row {i} is chunk {chunk_indices[i]} of {row_id}, but chunk "
                        f"{len(chunks)} of {doc_id} must come next"
                    )
                chunks.append(input_ids[i].values.to_numpy())
        if chunks:
            assert doc_id is not None
            yield doc_id, chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


def _iter_surviving_docs(
    spec: dict[str, str],
    cluster_col: str,
    *,
    stats: _FilterStats,
) -> Iterator[_SurvivingDocument]:
    """Join one shard's datasets and yield its surviving documents.

    Reads decon/cluster/quality densely and duplicate attributes sparsely. It
    streams tokenize in positional lockstep and drops filtered rows. It fails
    on missing or misaligned inputs.
    """
    decon_ids, contaminated = _load_decon_table(spec["decontam"])
    cluster_ids, cluster_vals = _load_cluster_table(spec["cluster"], cluster_col)
    # Quality parquets carry a precomputed, calibrated ``quality_bucket`` column
    # (fast-transformer scorer), consumed as-is -- no score->bucket mapping here.
    quality_ids, quality_buckets = _load_quality_table(spec["quality"])
    n_decon, n_cluster, n_quality = len(decon_ids), len(cluster_ids), len(quality_ids)
    if not (n_decon == n_cluster == n_quality):
        raise RuntimeError(
            f"{spec['source_name']}/{spec['basename']}: dense-table row count mismatch "
            f"(decon={n_decon}, cluster={n_cluster}, quality={n_quality}) -- co-partitioning broken"
        )
    # Equal row counts don't imply equal ID order. Verify the dense tables align
    # before routing positionally, and retain one ID sequence to check tokenize.
    where = f"{spec['source_name']}/{spec['basename']}"
    if not pc.all(pc.equal(decon_ids, cluster_ids)).as_py():
        raise RuntimeError(f"{where}: decon/cluster id mismatch -- co-partitioning broken")
    if not pc.all(pc.equal(decon_ids, quality_ids)).as_py():
        raise RuntimeError(f"{where}: decon/quality id mismatch -- co-partitioning broken")
    expected_ids = decon_ids.to_pylist()
    del decon_ids, cluster_ids, quality_ids
    exact_duplicates = _load_exact_duplicates(spec["exact_dedup"])
    verified_duplicates = _load_verified_duplicates(spec["dedup"])

    doc_idx = 0
    for doc_id, ids in _iter_tokenized_documents(spec["tokenize"]):
        if doc_idx >= n_decon:
            raise RuntimeError(
                f"{where}: tokenize holds more documents than decon rows ({n_decon}) -- co-partitioning broken"
            )
        stats.records_in += 1
        position, doc_idx = doc_idx, doc_idx + 1
        if doc_id != expected_ids[position]:
            raise RuntimeError(
                f"{where}: tokenize/decon id mismatch at document {position}: "
                f"{doc_id!r} != {expected_ids[position]!r} -- co-partitioning broken"
            )
        if contaminated[position]:
            stats.contaminated_dropped += 1
            continue
        # Verified attributes mark only the members a full-text comparison
        # confirmed, so membership alone decides.
        if doc_id in verified_duplicates:
            stats.fuzzy_duplicate_dropped += 1
            continue
        if doc_id in exact_duplicates:
            stats.exact_duplicate_dropped += 1
            continue
        stats.records_out += 1
        yield _SurvivingDocument(
            cluster=int(cluster_vals[position]),
            quality=int(quality_buckets[position]),
            input_ids=ids,
        )
    if doc_idx != n_decon:
        raise RuntimeError(
            f"{where}: tokenize documents ({doc_idx}) != decon rows ({n_decon}) -- co-partitioning broken"
        )


def _spill_source_shard(
    spec: dict[str, str],
    cluster_col: str,
    scratch_dir: str,
    shard_index: int,
) -> _SpillShardResult:
    """Join one source shard and write its bucket runs to local storage."""
    shard_dir = os.path.join(scratch_dir, f"shard-{shard_index:05d}")
    os.makedirs(shard_dir)
    buckets: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    stats = _FilterStats()
    for document in _iter_surviving_docs(spec, cluster_col, stats=stats):
        buckets[(document.cluster, document.quality)].append(np.asarray(document.input_ids, dtype=np.int32))

    results: list[_SpillBucketRun] = []
    total_tokens = 0
    for (cluster, quality), documents in buckets.items():
        stem = os.path.join(shard_dir, f"cluster-{cluster}-quality-{quality}")
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
        buckets=[_TaskBucketStat(**item) for item in payload["buckets"]],
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
    """Join, partition, and write one map task; commit its sidecar last."""
    task = task_input["task"]
    total_tasks = task_input["total_tasks"]
    batch_specs = task_input["specs"]
    sidecar_path = _task_sidecar_path(output_path, task, total_tasks)
    if StoragePath(sidecar_path).exists():
        counters.pipeline.update_counter("datakit_store/tasks_resumed", 1)
        return {"sidecar_path": sidecar_path, "task": task, "resumed": 1}

    initial_counters = counters.pipeline.get_counters()
    spill_runs: dict[tuple[int, int], list[BucketSpillRun]] = defaultdict(list)
    bucket_tokens: dict[tuple[int, int], int] = defaultdict(int)
    n_tokens = 0
    partition_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="datakit-store-") as scratch_dir:
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
                    counters.pipeline.update_counter(name, value)
                for bucket in result.buckets:
                    key = (bucket.cluster, bucket.quality)
                    spill_runs[key].append(bucket.run)
                    bucket_tokens[key] += bucket.run.tokens
        partition_seconds = time.monotonic() - partition_started

        attempt = uuid.uuid4().hex

        def write_one(key: tuple[int, int]) -> _TaskBucketStat:
            cluster, quality = key
            bucket_root = prefix_join(output_path, f"cluster={cluster}/quality={quality}")
            pattern = prefix_join(bucket_root, f"part-{{shard:05d}}-of-{{total:05d}}-attempt-{attempt}")
            cache_path = format_shard_path(pattern, task, total_tasks)
            ledger = write_bucket_cache_from_spills(cache_path, spill_runs[key])
            return _TaskBucketStat(
                cluster=cluster,
                quality=quality,
                task=task,
                path=cache_path,
                rows=ledger.total_num_rows,
                tokens=bucket_tokens[key],
            )

        ordered_keys = sorted(spill_runs, key=lambda key: bucket_tokens[key], reverse=True)
        bucket_stats: list[_TaskBucketStat] = []
        write_started = time.monotonic()
        if ordered_keys:
            with ThreadPoolExecutor(max_workers=min(max_parallel_bucket_writes, len(ordered_keys))) as executor:
                futures = [executor.submit(write_one, key) for key in ordered_keys]
                for future in as_completed(futures):
                    bucket_stats.append(future.result())
        write_seconds = time.monotonic() - write_started
    counters.pipeline.update_counter("datakit_store/tokens_out", n_tokens)
    counters.pipeline.update_counter("datakit_store/tasks_written", 1)
    counters.pipeline.update_counter("datakit_store/bucket_caches_written", len(bucket_stats))
    for name, value in (
        ("datakit_store/partition_seconds_max", partition_seconds),
        ("datakit_store/write_seconds_max", write_seconds),
        ("datakit_store/task_seconds_max", partition_seconds + write_seconds),
    ):
        counters.pipeline.set_aggregation(name, counters.Aggregation.MAX)
        counters.pipeline.update_counter(name, value)
    final_counters = counters.pipeline.get_counters()
    task_counters = {
        name: value - initial_counters.get(name, 0)
        for name, value in final_counters.items()
        if name.startswith("datakit_store/") and value != initial_counters.get(name, 0)
    }
    bucket_stats.sort(key=lambda stat: (stat.cluster, stat.quality))
    _write_task_sidecar(sidecar_path, _TaskCompletion(buckets=bucket_stats, counters=task_counters))
    return {"sidecar_path": sidecar_path, "task": task, "resumed": 0}


# ---------------------------------------------------------------------------
# Driver-side per-bucket ledger merge.
# ---------------------------------------------------------------------------


def _merge_per_bucket_ledgers(
    *,
    task_bucket_stats: list[_TaskBucketStat],
    output_path: str,
) -> list[BucketCacheStats]:
    """Write one per-bucket ledger over the completed tasks' leaf caches.

    Pure driver-side work: each stat already carries its cache's row + token
    counts, so we synthesize the minimal ``CacheLedger`` stubs ``_merge_sharded_ledgers``
    needs (it only reads ``total_num_rows``) and call it per bucket.
    """
    by_bucket: dict[tuple[int, int], list[_TaskBucketStat]] = defaultdict(list)
    for s in task_bucket_stats:
        by_bucket[(s.cluster, s.quality)].append(s)

    metadata = CacheMetadata.empty()
    buckets: list[BucketCacheStats] = []
    for key in sorted(by_bucket):
        cluster, quality = key
        bucket_root = prefix_join(output_path, f"cluster={cluster}/quality={quality}")
        task_parts = sorted(by_bucket[key], key=lambda s: (s.task, s.path))
        shard_paths = [s.path for s in task_parts]
        shard_ledgers = [
            CacheLedger(total_num_rows=s.rows, shard_rows={}, finished_shards=[], field_counts={}, metadata=metadata)
            for s in task_parts
        ]
        per_shard_field_counts = [{"input_ids": s.tokens} for s in task_parts]
        ledger = _merge_sharded_ledgers(bucket_root, shard_paths, shard_ledgers, per_shard_field_counts, metadata)
        total_tokens = ledger.field_counts.get("input_ids", 0)
        buckets.append(
            BucketCacheStats(
                cluster_id=cluster,
                quality_bucket=quality,
                path=bucket_root,
                total_elements=ledger.total_num_rows,
                total_tokens=total_tokens,
                n_shards=len(task_parts),
            )
        )
        logger.info(
            "cluster=%d quality=%d: docs=%d tokens=%d task_parts=%d -> %s",
            cluster,
            quality,
            ledger.total_num_rows,
            total_tokens,
            len(task_parts),
            bucket_root,
        )
    return buckets


# ---------------------------------------------------------------------------
# Driver entry point.
# ---------------------------------------------------------------------------


def build_clustered_store(
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
) -> ClusteredStoreData:
    """Build materialized caches for each populated ``(cluster, quality)`` bucket.

    Args:
        task_count: Split input shards round-robin across this many tasks. When
            omitted, each input shard becomes one task.
        max_parallel_bucket_writes: Bucket caches finalized concurrently per
            task. Each writer may hold one 512 MiB aligned token buffer.
        partition_processes: Source shards joined and partitioned concurrently
            into append-only local bucket runs inside each task.
    """
    if not tokenize:
        raise ValueError("build_clustered_store: tokenize is empty")
    for label, d in (("decontam", decontam), ("cluster_assign", cluster_assign), ("quality", quality)):
        if set(d) != set(tokenize):
            missing = sorted(set(tokenize) - set(d))
            extra = sorted(set(d) - set(tokenize))
            raise ValueError(f"{label} source set must equal tokenize: missing={missing!r}, extra={extra!r}")
    if task_count is not None and task_count < 1:
        raise ValueError(f"task_count must be >= 1, got {task_count}")
    if max_parallel_bucket_writes < 1:
        raise ValueError(f"max_parallel_bucket_writes must be >= 1, got {max_parallel_bucket_writes}")
    if partition_processes < 1:
        raise ValueError(f"partition_processes must be >= 1, got {partition_processes}")

    # Every source must share one quality model so bucket IDs are comparable.
    models = {(q.model_dir, q.calib_file, tuple(q.bucket_edges)) for q in quality.values()}
    if len(models) != 1:
        raise ValueError(f"build_clustered_store: sources span multiple quality models: {sorted(models)}")
    bucket_edges = next(iter(quality.values())).bucket_edges

    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)

    source_keys: dict[str, str] = {}
    for source_name, tok in tokenize.items():
        source_key = tok.source_keys.get(split)
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

    for source_name in sorted(tokenize):
        source_key = source_keys[source_name]
        cluster_asg = cluster_assign[source_name]
        if cluster_asg.source_key != source_key:
            raise ValueError(
                f"{source_name}: cluster_assign.source_key={cluster_asg.source_key!r} "
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
    batched_specs = [shard_specs[i::resolved_task_count] for i in range(resolved_task_count)]
    logger.info(
        "build_clustered_store: %d sources, %d input shards -> %d map-only tasks -> %s",
        len(tokenize),
        len(shard_specs),
        len(batched_specs),
        output_path,
    )

    if worker_resources is None:
        # Production supplies a dedicated worker shape through StoreConfig.
        worker_resources = ResourceConfig(cpu=2, ram="16g", disk="16g")

    ctx = ZephyrContext(
        resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
        max_workers=min(max_workers, len(batched_specs)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=output_path),
        name="datakit-clustered-store",
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
    ds = Dataset.from_list(tasks).map(write_task)
    outcome = ctx.execute(
        ds,
        verbose=True,
        map_task_resources=worker_resources,
    )
    confirmations = [r for r in outcome.results if r is not None]
    logger.info(
        "build_clustered_store: completed %d tasks (records_out=%d, tokens_out=%d)",
        len(confirmations),
        outcome.counters.get("datakit_store/records_out", 0),
        outcome.counters.get("datakit_store/tokens_out", 0),
    )

    sidecar_paths = [str(result["sidecar_path"]) for result in confirmations]
    with ThreadPoolExecutor(max_workers=min(64, len(sidecar_paths))) as executor:
        completions = list(executor.map(_load_task_sidecar, sidecar_paths))
    task_bucket_stats = [stat for completion in completions for stat in completion.buckets]
    buckets = _merge_per_bucket_ledgers(task_bucket_stats=task_bucket_stats, output_path=output_path)

    durable_counters: dict[str, int | float] = defaultdict(int)
    for completion in completions:
        for name, value in completion.counters.items():
            if name.endswith("_max"):
                durable_counters[name] = max(durable_counters[name], value)
            else:
                durable_counters[name] += value
    artifact_counters = dict(outcome.counters)
    artifact_counters.update(durable_counters)

    tokenizer = next(iter(tokenize.values())).tokenizer
    artifact = ClusteredStoreData(
        cache_path=output_path,
        cluster_view=cluster_view,
        bucket_edges=bucket_edges,
        split=split,
        buckets=buckets,
        source_names=sorted(tokenize),
        tokenizer=tokenizer,
        counters=artifact_counters,
    )
    write_artifact(artifact, output_path)
    return artifact
