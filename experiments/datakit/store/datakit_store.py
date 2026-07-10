# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit -> per-(cluster, quality) Levanter store: single map-side Zephyr pass.

Shape mirrors :mod:`marin.processing.classification.consolidate`, specialized
for the datakit attribute datasets that the global pipelines produce:

    tokenize        per-source ``{id, input_ids}``, dense, sorted by id
    decontam        per-source ``{id, attributes: {contaminated, ...}}``, dense
    cluster_assign  per-source ``{id, cluster_<K>, ...}``, dense
    quality         per-source ``{id, score: float}``, dense (flat schema;
                    output of fasttext quality classifier via
                    ``score_target_label`` path which writes a scalar
                    directly under ``output_field_name="score"``)
    dedup           per-source ``{id, attributes: {is_cluster_canonical, ...}}``,
                    SPARSE -- singletons omitted by ``compute_fuzzy_dups_attrs``

All five are co-partitioned with the source ``NormalizedData`` by basename, so
a single map-side pass joins them per shard with no shuffle. The pass:

1. Reads decon, cluster_assign, and quality fully (small columns; one shard's worth).
2. Reads dedup into ``{id -> is_canonical}`` if present.
3. Streams tokenize via ``ParquetFile.iter_batches`` in positional lockstep
   with the three dense attribute tables (sanity-asserts id alignment).
4. Drops contaminated rows; drops dedup-cluster non-canonicals (rows missing
   from dedup are singletons -> kept).
5. Maps each surviving doc's ``score`` into one of ``len(_QUALITY_THRESHOLDS) + 1``
   quality buckets (default 5; fixed cutoffs ``[0.2, 0.4, 0.6, 0.8]``), then
   routes the row by ``(cluster_<view>, quality_bucket)`` directly into one
   of up to ``K_clusters * K_quality`` lazily-opened ``SerialCacheWriter``
   instances under ``<output>/cluster=<C>/quality=<Q>/part-NNNNN-of-MMMMM``.
   Memory peak stays at ``N_open_writers * _BATCH_FLUSH * avg-doc-size``
   (~hundreds of MB worst case at 200 buckets) -- independent of input-shard size.
6. After ``ExitStack`` closes the writers (committing per-shard ledgers),
   loads each just-written ledger (``SerialCacheWriter`` already populates
   ``field_counts``) and yields it back to the driver.

After the Zephyr execute, the driver groups results by
``(cluster_id, quality_bucket)`` in a plain Python dict (no Zephyr group_by,
no shuffle) and calls ``_merge_sharded_ledgers`` per bucket -- each call only
writes the small ``<output>/cluster=<C>/quality=<Q>/shard_ledger.json``. No
second Zephyr context.
"""

import bisect
import contextlib
import dataclasses
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.store.cache import (
    CacheLedger,
    CacheMetadata,
    SerialCacheWriter,
    _merge_sharded_ledgers,
)
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo, format_shard_path
from zephyr.execution import ZephyrContext
from zephyr.writers import atomic_rename

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import LlmQualityOutput

logger = logging.getLogger(__name__)


# Fixed quality-bucket thresholds: score is bucketed via bisect_right
# against these cutoffs, yielding 0..(len(thresholds)) inclusive. With the
# default below: bucket 0 = [0, 0.2), 1 = [0.2, 0.4), 2 = [0.4, 0.6),
# 3 = [0.6, 0.8), 4 = [0.8, 1.0].
_QUALITY_THRESHOLDS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
N_QUALITY_BUCKETS = len(_QUALITY_THRESHOLDS) + 1


class BucketCacheStats(BaseModel):
    """Per-(cluster, quality) Levanter cache stats inside :class:`ClusteredStoreData`."""

    cluster_id: int
    quality_bucket: int
    path: str
    total_elements: int
    total_tokens: int
    n_shards: int


class ClusteredStoreData(BaseModel):
    """Outcome of :func:`build_clustered_store`: one Levanter cache per (cluster, quality) bucket.

    Persisted as ``<output_path>/artifact.json``. Load via
    ``read_artifact(output_path, ClusteredStoreData)``.

    Attributes:
        cache_path: Root directory. Each bucket's cache lives at
            ``cache_path/cluster=<C>/quality=<Q>/``.
        cluster_view: Cluster-K used to partition the store
            (column name ``cluster_<view>`` read from the assignment shards).
        quality_thresholds: Cutoffs used to bucket ``score`` via
            ``bisect_right``. Length == ``N_QUALITY_BUCKETS - 1``.
        split: Tokenize split fed into the store (e.g. ``"train"``).
        buckets: List of per-(cluster, quality) stats. Buckets that
            received zero records across every input shard are omitted.
        source_names: Registry names of the sources that contributed.
        tokenizer: Tokenizer name copied from the first tokenize source.
        counters: Aggregated zephyr counters across the join pass.
    """

    version: str = "v2"
    cache_path: str
    cluster_view: int
    quality_thresholds: list[float]
    split: str
    buckets: list[BucketCacheStats]
    source_names: list[str]
    tokenizer: str
    counters: dict[str, int | float]


# ---------------------------------------------------------------------------
# Shard input spec (per-source-shard 4-tuple of paths + global indexing).
# ---------------------------------------------------------------------------


def _per_source_shard_tuples(
    *,
    source_name: str,
    tokenize: TokenizedAttrData,
    decontam: DeconAttributes,
    cluster_assign: AssignmentAttrData,
    quality: LlmQualityOutput,
    dedup_attr_dir: str,
    split: str,
) -> list[dict[str, str]]:
    """Align the five datasets' parquet shards for one source by basename.

    Returns a list of
    ``{tokenize, decontam, cluster, quality, dedup, source_name, basename}``
    dicts -- one per source shard. ``dedup`` may be absent for source shards
    with zero non-singletons; worker handles that case.
    """
    tok_dir = tokenize.output_dirs.get(split)
    if tok_dir is None:
        raise FileNotFoundError(f"{source_name}: tokenize has no split={split!r}")
    tok_shards = sorted(str(m) for m in StoragePath(f"{tok_dir.rstrip('/')}/*.parquet").glob())
    if not tok_shards:
        raise FileNotFoundError(f"{source_name}: no tokenize shards under {tok_dir}")

    # We trust the datakit co-partitioning invariant: tokenize / decon /
    # cluster_assign / quality / dedup all mirror the source NormalizedData
    # basenames. Workers fail loud if a per-shard file is missing -- cheaper
    # than O(N_shards) serial GCS HEAD requests up front.
    decon_dir = decontam.output_dir.rstrip("/")
    cluster_dir = cluster_assign.output_dir.rstrip("/")
    quality_dir = quality.output_dir.rstrip("/")
    dedup_dir = dedup_attr_dir.rstrip("/")
    return [
        {
            "tokenize": tok_path,
            "decontam": f"{decon_dir}/{os.path.basename(tok_path)}",
            "cluster": f"{cluster_dir}/{os.path.basename(tok_path)}",
            # Quality's writer uses ``data-NNNNN-of-MMMMM.parquet`` (set in
            # cluster/quality/v0/all_sources_quality_llm.py) while every
            # other co-partitioned step uses ``part-NNNNN-of-MMMMM.parquet``.
            # Map the basename so the join stays purely basename-keyed
            # without invalidating the existing quality cache.
            "quality": f"{quality_dir}/{os.path.basename(tok_path).replace('part-', 'data-', 1)}",
            # ``dedup`` may legitimately be absent for shards with zero
            # non-singletons. Worker checks existence before opening.
            "dedup": f"{dedup_dir}/{os.path.basename(tok_path)}",
            "source_name": source_name,
            "basename": os.path.basename(tok_path),
        }
        for tok_path in tok_shards
    ]


# ---------------------------------------------------------------------------
# Per-shard map: join + filter + bucket-by-cluster + write Levanter shards.
# ---------------------------------------------------------------------------


# IDs are kept as ``pa.Array`` (compact: a single string buffer + offsets,
# ~10-20 bytes/row) and only used for a one-shot vectorized alignment check
# at start of the shard; the dedup/filter loop never sees them as Python
# objects. Values come back as numpy arrays (1 byte/bool, 4 byte/int32, 8
# byte/float) -- crucial on big nemotron shards where Python-list boxing
# of the full id+value columns was OOM'ing 8g workers (50 bytes/Python str
# x ~5M rows x 3 id cols = ~750 MB just on strings).


def _load_decon_table(path: str) -> tuple[pa.Array, np.ndarray]:
    """Return ``(ids, contaminated)`` for one decon shard. ids is pyarrow, contam is bool numpy."""
    table = pq.read_table(path, columns=["id", "attributes"])
    ids = table.column("id").combine_chunks()
    contaminated = np.asarray(
        table.column("attributes").combine_chunks().field("contaminated"),
        dtype=bool,
    )
    return ids, contaminated


def _load_cluster_table(path: str, cluster_col: str) -> tuple[pa.Array, np.ndarray]:
    """Return ``(ids, cluster)`` for one cluster-assign shard. ids is pyarrow, cluster is int32 numpy."""
    table = pq.read_table(path, columns=["id", cluster_col])
    ids = table.column("id").combine_chunks()
    cluster = np.asarray(table.column(cluster_col), dtype=np.int32)
    return ids, cluster


def _load_quality_table(path: str) -> tuple[pa.Array, np.ndarray]:
    """Return ``(ids, score)`` for one quality shard. ids is pyarrow, scores are float64 numpy.

    Quality parquets have a flat ``{id: string, score: double}`` schema --
    unlike decon/dedup which nest under ``attributes`` -- because the
    fasttext quality classifier's ``score_target_label`` path writes the
    extracted scalar directly via ``output_field_name``. Read the flat
    ``score`` column, no struct field access.
    """
    table = pq.read_table(path, columns=["id", "score"])
    ids = table.column("id").combine_chunks()
    score = np.asarray(table.column("score"), dtype=np.float64)
    return ids, score


def _load_dedup_canonical(path: str) -> dict[str, bool]:
    """Return ``{id -> is_cluster_canonical}`` for one dedup shard, or ``{}`` if absent.

    Dedup is sparse: ids missing from this dict are singletons (kept). Ids
    present are non-singleton cluster members; only the canonical one survives.
    """
    if not StoragePath(path).exists():
        return {}
    # Sources with zero non-singletons (e.g. ghalogs/public) get an empty
    # parquet stub from the dedup writer -- 176 bytes, num_rows=0, zero
    # data columns. Treat that as "no non-singletons" rather than letting
    # ``pq.read_table(columns=["id","attributes"])`` raise ArrowInvalid
    # because the projected columns don't exist in the empty schema.
    if pq.ParquetFile(path).metadata.num_rows == 0:
        return {}
    table = pq.read_table(path, columns=["id", "attributes"])
    ids = table.column("id").to_pylist()
    canonical = table.column("attributes").combine_chunks().field("is_cluster_canonical").to_pylist()
    return dict(zip(ids, canonical, strict=True))


@dataclasses.dataclass(frozen=True)
class _WrittenShard:
    """Slim summary of one per-(input_shard, cluster, quality) Levanter shard cache.

    Persisted to a per-input-shard sidecar in GCS (see ``_sidecar_path``);
    the driver scans those sidecars at end-of-pipeline rather than reading
    millions of these records back via zephyr's ``outcome.results`` (which
    OOMs the 1 GB-default zephyr coord at full-fleet scale: ~98K shards x
    ~50 non-empty buckets each ≈ 5M records).
    """

    cluster_id: int
    quality_bucket: int
    path: str
    total_num_rows: int
    field_counts: dict[str, int]


def _sidecar_path(output_path: str, shard_idx: int, total: int) -> str:
    """Return the per-input-shard sidecar path."""
    return f"{output_path.rstrip('/')}/_done/shard-{shard_idx:05d}-of-{total:05d}.json"


def _write_shard_sidecar(path: str, records: list[_WrittenShard]) -> None:
    """Persist per-shard sidecar atomically (write tmp, rename).

    Holding the full list of ``_WrittenShard`` records for one shard --
    at most ~200 buckets * ~250 bytes = ~50 KB -- on the worker is cheap.
    The driver reads these back via :func:`_load_shard_sidecar`.
    """
    payload = json.dumps([dataclasses.asdict(r) for r in records])
    tmp_path = f"{path}.tmp"
    StoragePath(tmp_path).write_text(payload)
    StoragePath(tmp_path).rename(path)


def _load_shard_sidecar(path: str) -> list[_WrittenShard] | None:
    """Load a per-shard sidecar if present; return ``None`` if missing.

    A missing sidecar means the shard has not been processed (or only
    partially processed and crashed before writing the sidecar); the worker
    re-runs it. ``atomic_rename`` on the Levanter caches makes the re-run
    safe: bucket dirs that already exist get overwritten.
    """
    if not StoragePath(path).exists():
        return None
    data = json.loads(StoragePath(path).read_text())
    return [_WrittenShard(**d) for d in data]


# Per-bucket pending-buffer flush threshold inside one map_shard task.
# Pending-memory bound = N_open_buckets * _BATCH_FLUSH * avg-doc-bytes.
# With ``input_ids`` stored as numpy int32 (4 bytes/token) and nemotron-
# style docs averaging ~few thousand tokens, ~200 buckets * 256 docs *
# 20 KB/doc ≈ 1 GB worst case. Larger _BATCH_FLUSH (e.g. 1024) blew the
# 8g worker on big nemotron shards.
_BATCH_FLUSH = 256

# Bound the per-batch pyarrow buffer when streaming tokenize. Default
# iter_batches batch_size is row-group sized (~64K), which on long-doc
# sources can pull a multi-GB chunk into memory at once.
_TOKENIZE_BATCH_SIZE = 8192


def _quality_bucket(score: float) -> int:
    """Map a fasttext ``score`` (float in [0, 1]) to a bucket index 0..N_QUALITY_BUCKETS-1."""
    return bisect.bisect_right(_QUALITY_THRESHOLDS, score)


def _join_filter_stream_shard(
    items: Iterator[list[dict[str, str]]],
    shard_info: ShardInfo,
    *,
    cluster_col: str,
    output_path: str,
) -> Iterator[dict[str, int]]:
    """One TASK (batch of N input shards) -> up to K_clusters x K_quality Levanter shard caches.

    Streams records to lazily-opened :class:`SerialCacheWriter` instances --
    one per ``(cluster_id, quality_bucket)`` that actually receives rows --
    SHARED across all input shards in the batch. Each writer produces ONE
    ``part-NNNNN-of-MMMMM`` file in its bucket, where NNNNN is the batch
    (task) index. Output-file count is therefore total_buckets_touched per
    task, not per input shard, giving the caller a knob (``shards_per_task``)
    to reduce the total file count by N x.

    Memory bounded by ``N_open_buckets * _BATCH_FLUSH * avg_doc_size``
    regardless of batch size -- pending buffers stay scoped to one bucket
    each, independent of how many input shards a task aggregates.

    Writes one per-task sidecar JSON to ``<output>/_done/shard-NNNNN.json``
    listing every cluster=K/quality=Q cache the batch produced; resume on
    retry skips the whole batch if the sidecar is already present.

    Yields one ``{shard_idx, n_buckets}`` confirmation per task.
    """
    batch_specs = next(iter(items))
    if not batch_specs:
        # Empty batch shouldn't happen but be defensive.
        yield {"shard_idx": shard_info.shard_idx, "n_buckets": 0}
        return

    sidecar = _sidecar_path(output_path, shard_info.shard_idx, shard_info.total_shards)
    cached = _load_shard_sidecar(sidecar)
    if cached is not None:
        counters.pipeline.update_counter("datakit_store/shards_resumed", 1)
        yield {"shard_idx": shard_info.shard_idx, "n_buckets": len(cached)}
        return

    written: dict[tuple[int, int], str] = {}
    base_path = output_path.rstrip("/")

    def _shard_dir(cluster_id: int, quality_id: int) -> str:
        return format_shard_path(
            f"{base_path}/cluster={cluster_id}/quality={quality_id}/part-{{shard:05d}}-of-{{total:05d}}",
            shard_info.shard_idx,
            shard_info.total_shards,
        )

    n_in_total = 0
    n_contaminated_total = 0
    n_dedup_dropped_total = 0
    n_out_total = 0

    with contextlib.ExitStack() as stack:
        # Writers + pending are shared across every spec in the batch so all
        # input shards routing to the same (cluster, quality) bucket merge
        # into one output part file.
        writers: dict[tuple[int, int], SerialCacheWriter] = {}
        pending: dict[tuple[int, int], list[dict[str, list[int]]]] = defaultdict(list)

        def get_writer(key: tuple[int, int], exemplar: dict[str, list[int]]) -> SerialCacheWriter:
            if key not in writers:
                cache_dir = _shard_dir(*key)
                written[key] = cache_dir
                tmp_path = stack.enter_context(atomic_rename(cache_dir))
                writers[key] = stack.enter_context(
                    SerialCacheWriter(
                        tmp_path,
                        exemplar,
                        shard_name=cache_dir,
                        metadata=CacheMetadata.empty(),
                    )
                )
            return writers[key]

        def flush(key: tuple[int, int]) -> None:
            buf = pending[key]
            if not buf:
                return
            get_writer(key, buf[0]).write_batch(buf)
            pending[key] = []

        for spec in batch_specs:
            source_name = spec["source_name"]
            tok_path = spec["tokenize"]
            decon_path = spec["decontam"]
            cluster_path = spec["cluster"]
            quality_path = spec["quality"]
            dedup_path = spec["dedup"]

            decon_ids, contaminated = _load_decon_table(decon_path)
            cluster_ids, cluster_vals = _load_cluster_table(cluster_path, cluster_col)
            quality_ids, scores = _load_quality_table(quality_path)
            n_decon, n_cluster, n_quality = len(decon_ids), len(cluster_ids), len(quality_ids)
            if not (n_decon == n_cluster == n_quality):
                raise RuntimeError(
                    f"{source_name}/{spec['basename']}: dense-table row count mismatch "
                    f"(decon={n_decon}, cluster={n_cluster}, quality={n_quality}) -- co-partitioning broken"
                )
            # One-shot vectorized id alignment across decon/cluster/quality. If it
            # passes we can drop the id arrays for the rest of the shard (the inner
            # row loop only needs the tokenize id for the dedup lookup).
            if not pc.all(pc.equal(decon_ids, cluster_ids)).as_py():
                raise RuntimeError(
                    f"{source_name}/{spec['basename']}: decon/cluster id mismatch -- co-partitioning broken"
                )
            if not pc.all(pc.equal(decon_ids, quality_ids)).as_py():
                raise RuntimeError(
                    f"{source_name}/{spec['basename']}: decon/quality id mismatch -- co-partitioning broken"
                )
            del decon_ids, cluster_ids, quality_ids
            dedup_canonical = _load_dedup_canonical(dedup_path)

            with StoragePath(tok_path).open("rb") as fh:
                pf = pq.ParquetFile(fh)
                row_idx = 0
                for batch in pf.iter_batches(batch_size=_TOKENIZE_BATCH_SIZE, columns=["id", "input_ids"]):
                    tok_ids = batch.column("id").to_pylist()
                    # Keep input_ids in pyarrow; per-survivor conversion to numpy
                    # int32 happens row-by-row below. ``to_pylist()`` here would
                    # materialize ~28 bytes per token as Python int boxes, blowing
                    # the worker on long-doc shards.
                    tok_input_ids_arr = batch.column("input_ids")
                    batch_len = len(tok_ids)
                    # Numpy slices -- O(0) views, not copies.
                    decon_slice = contaminated[row_idx : row_idx + batch_len]
                    cluster_slice = cluster_vals[row_idx : row_idx + batch_len]
                    quality_slice = scores[row_idx : row_idx + batch_len]
                    row_idx += batch_len

                    for i, doc_id in enumerate(tok_ids):
                        n_in_total += 1
                        if decon_slice[i]:
                            n_contaminated_total += 1
                            continue
                        if dedup_canonical.get(doc_id) is False:
                            n_dedup_dropped_total += 1
                            continue
                        # canonical True OR id missing from dedup (singleton) -> keep
                        key = (int(cluster_slice[i]), _quality_bucket(quality_slice[i]))
                        # ``.values.to_numpy()`` copies just this row's tokens
                        # into a fresh int32 buffer (~4 bytes/token vs ~28 for
                        # boxed Python ints), so the pyarrow batch can be GC'd
                        # after the loop.
                        pending[key].append({"input_ids": tok_input_ids_arr[i].values.to_numpy()})
                        n_out_total += 1
                        if len(pending[key]) >= _BATCH_FLUSH:
                            flush(key)

                if row_idx != n_decon:
                    raise RuntimeError(
                        f"{source_name}/{spec['basename']}: tokenize rows ({row_idx}) != "
                        f"decon rows ({n_decon}) -- co-partitioning broken"
                    )

        # Flush remaining buffers before ExitStack closes the writers.
        for key in list(pending):
            flush(key)

    # ExitStack done: SerialCacheWriter.__exit__ wrote each per-shard ledger;
    # atomic_rename.__exit__ renamed tmp_path -> cache_dir. Load each ledger
    # once so the driver can run _merge_sharded_ledgers without re-reading.
    counters.pipeline.update_counter("datakit_store/records_in", n_in_total)
    counters.pipeline.update_counter("datakit_store/contaminated_dropped", n_contaminated_total)
    counters.pipeline.update_counter("datakit_store/dedup_noncanonical_dropped", n_dedup_dropped_total)
    counters.pipeline.update_counter("datakit_store/records_out", n_out_total)

    metadata = CacheMetadata.empty()
    records: list[_WrittenShard] = []
    for (cluster_id, quality_id), cache_dir in written.items():
        ledger = CacheLedger.load(cache_dir, metadata)
        records.append(
            _WrittenShard(
                cluster_id=cluster_id,
                quality_bucket=quality_id,
                path=cache_dir,
                total_num_rows=ledger.total_num_rows,
                field_counts=dict(ledger.field_counts),
            )
        )
    # Sidecar is written LAST -- atomic-rename of the per-bucket caches
    # already happened in the ExitStack exit. So sidecar presence implies
    # every cache referenced inside is finalized.
    _write_shard_sidecar(sidecar, records)
    yield {"shard_idx": shard_info.shard_idx, "n_buckets": len(records)}


# ---------------------------------------------------------------------------
# Driver entry point.
# ---------------------------------------------------------------------------


def _validate_cluster_view(cluster_assign: dict[str, AssignmentAttrData], cluster_view: int) -> str:
    """Check ``cluster_view`` is materialized by every assignment artifact; return its column name."""
    for name, asg in cluster_assign.items():
        valid_views = {asg.k_train, *asg.k_views}
        if cluster_view not in valid_views:
            raise ValueError(
                f"cluster_view={cluster_view} not in {name}'s views (k_train={asg.k_train}, " f"k_views={asg.k_views})"
            )
    return f"cluster_{cluster_view}"


def _resolve_dedup_attr_dir(
    *,
    source_name: str,
    main_output_dir: str,
    dedup: FuzzyDupsAttrData,
) -> str:
    entry = dedup.sources.get(main_output_dir)
    if entry is None:
        raise KeyError(
            f"{source_name}: dedup.sources has no entry for source_main_dir={main_output_dir!r}. "
            "Drop the source from the config or rebuild dedup with it included."
        )
    return entry.attr_dir


def _merge_per_bucket_ledgers(
    *,
    map_shard_results: list[_WrittenShard],
    output_path: str,
) -> list[BucketCacheStats]:
    """Merge per-(input_shard, cluster, quality) ledgers into one ledger per ``(cluster, quality)`` bucket.

    Pure driver-side work: ``map_shard_results`` already carries each
    written shard's ``total_num_rows`` + ``field_counts`` (loaded by the
    worker right after ``write_levanter_cache`` and projected to the slim
    :class:`_WrittenShard`). We group by ``(cluster_id, quality_bucket)``
    in a Python dict (no Zephyr group_by, no shuffle), synthesize minimal
    ``CacheLedger`` stubs since ``_merge_sharded_ledgers`` only reads
    ``total_num_rows``, and call it -- which writes the small
    ``cluster=<C>/quality=<Q>/shard_ledger.json`` per bucket.
    """
    by_bucket: dict[tuple[int, int], list[_WrittenShard]] = defaultdict(list)
    for r in map_shard_results:
        by_bucket[(r.cluster_id, r.quality_bucket)].append(r)

    metadata = CacheMetadata.empty()
    buckets: list[BucketCacheStats] = []
    base_path = output_path.rstrip("/")
    for key in sorted(by_bucket):
        cluster_id, quality_id = key
        bucket_root = f"{base_path}/cluster={cluster_id}/quality={quality_id}"
        entries = sorted(by_bucket[key], key=lambda e: e.path)
        shard_paths = [e.path for e in entries]
        shard_ledgers = [
            CacheLedger(
                total_num_rows=e.total_num_rows,
                shard_rows={},
                finished_shards=[],
                field_counts={},
                metadata=metadata,
            )
            for e in entries
        ]
        per_shard_field_counts = [e.field_counts for e in entries]
        ledger = _merge_sharded_ledgers(bucket_root, shard_paths, shard_ledgers, per_shard_field_counts, metadata)
        total_tokens = ledger.field_counts.get("input_ids", 0)
        buckets.append(
            BucketCacheStats(
                cluster_id=cluster_id,
                quality_bucket=quality_id,
                path=bucket_root,
                total_elements=ledger.total_num_rows,
                total_tokens=total_tokens,
                n_shards=len(shard_paths),
            )
        )
        logger.info(
            "cluster=%d quality=%d: docs=%d tokens=%d shards=%d -> %s",
            cluster_id,
            quality_id,
            ledger.total_num_rows,
            total_tokens,
            len(shard_paths),
            bucket_root,
        )
    return buckets


def build_clustered_store(
    *,
    tokenize: dict[str, TokenizedAttrData],
    decontam: dict[str, DeconAttributes],
    cluster_assign: dict[str, AssignmentAttrData],
    quality: dict[str, LlmQualityOutput],
    dedup: FuzzyDupsAttrData,
    output_path: str,
    cluster_view: int = 40,
    split: str = "train",
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 4096,
    aggregate_only: bool = False,
    shards_per_task: int = 1,
) -> ClusteredStoreData:
    """Single map-side Zephyr pass: 5-way join + filter + per-(cluster, quality) Levanter caches.

    See module docstring for the per-shard logic.

    Every source name must appear in ``tokenize``, ``decontam``,
    ``cluster_assign``, and ``quality``; every included source's
    ``TokenizedAttrData.source_main_dirs[split]`` must appear as a key in
    ``dedup.sources`` (so the caller is responsible for dropping sources
    excluded from dedup, e.g. ``safety_pt/*`` / ``climblab-ja``).

    Returns a :class:`ClusteredStoreData` describing the per-bucket caches
    and saves the artifact at ``<output_path>/artifact.json``.
    """
    if not tokenize:
        raise ValueError("build_clustered_store: tokenize is empty")
    for label, d in (("decontam", decontam), ("cluster_assign", cluster_assign), ("quality", quality)):
        if set(d) != set(tokenize):
            missing = sorted(set(tokenize) - set(d))
            extra = sorted(set(d) - set(tokenize))
            raise ValueError(f"{label} source set must equal tokenize: missing={missing!r}, extra={extra!r}")

    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)
    counters: dict[str, int | float] = {}

    if aggregate_only:
        # Skip the zephyr pass entirely and pick up from the durable sidecars
        # already on GCS. Used to recover when a prior run finished the
        # map-side work but died before producing the per-bucket ledgers +
        # artifact.json (sequential sidecar load + driver-side merge).
        logger.info(
            "build_clustered_store: aggregate_only=True, skipping zephyr; sources=%d split=%s -> %s",
            len(tokenize),
            split,
            output_path,
        )
    else:
        if worker_resources is None:
            worker_resources = ResourceConfig(cpu=2, ram="16g", disk="10g")

        logger.info(
            "build_clustered_store: %d sources, cluster_view=%d (column=%s), quality_thresholds=%s, split=%s -> %s",
            len(tokenize),
            cluster_view,
            cluster_col,
            list(_QUALITY_THRESHOLDS),
            split,
            output_path,
        )

        shard_specs: list[dict[str, str]] = []
        for source_name in sorted(tokenize):
            tok = tokenize[source_name]
            decon = decontam[source_name]
            cluster_asg = cluster_assign[source_name]
            qual = quality[source_name]
            main_dir = tok.source_main_dirs.get(split)
            if main_dir is None:
                raise ValueError(f"{source_name}: tokenize has no source_main_dir for split={split!r}")
            if cluster_asg.source_main_dir != main_dir:
                raise ValueError(
                    f"{source_name}: cluster_assign.source_main_dir={cluster_asg.source_main_dir!r} "
                    f"!= tokenize.source_main_dirs[{split!r}]={main_dir!r}"
                )
            dedup_attr_dir = _resolve_dedup_attr_dir(source_name=source_name, main_output_dir=main_dir, dedup=dedup)
            shard_specs.extend(
                _per_source_shard_tuples(
                    source_name=source_name,
                    tokenize=tok,
                    decontam=decon,
                    cluster_assign=cluster_asg,
                    quality=qual,
                    dedup_attr_dir=dedup_attr_dir,
                    split=split,
                )
            )

        if not shard_specs:
            raise ValueError("No input shards resolved -- nothing to do")
        if shards_per_task < 1:
            raise ValueError(f"shards_per_task must be >= 1, got {shards_per_task}")
        # Group flat per-source-shard specs into per-task batches. Each task
        # sees one inner list and writes ONE output part file per (cluster,
        # quality) it touches -- so total output files scale with len(batched_specs),
        # not len(shard_specs).
        batched_specs: list[list[dict[str, str]]] = [
            shard_specs[i : i + shards_per_task] for i in range(0, len(shard_specs), shards_per_task)
        ]
        logger.info(
            "build_clustered_store: %d input shards across %d sources, batched into %d tasks (%d per task)",
            len(shard_specs),
            len(tokenize),
            len(batched_specs),
            shards_per_task,
        )

        # Zephyr coordinator needs more than the iris 1GB default: it tracks
        # the worker pool + retry state + per-shard confirmations. Workers yield
        # only tiny ``{shard_idx, n_buckets}`` records now (sidecar in GCS holds
        # the full _WrittenShard list), so 3g is enough headroom for ~2k workers.
        # preemptible=False because coord death = whole-job restart.
        ctx = ZephyrContext(
            resources=worker_resources,
            coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
            max_workers=min(max_workers, len(batched_specs)),
            name="datakit-clustered-store",
        )
        ds = Dataset.from_list(batched_specs).map_shard(
            lambda items, shard, cc=cluster_col, op=output_path: _join_filter_stream_shard(
                items, shard, cluster_col=cc, output_path=op
            )
        )
        outcome = ctx.execute(ds, verbose=True)
        logger.info(
            "build_clustered_store: zephyr pass produced %d shard confirmations (resumed=%d)",
            len(outcome.results),
            outcome.counters.get("datakit_store/shards_resumed", 0),
        )
        counters = dict(outcome.counters)

    # Aggregation: scan per-shard sidecars in GCS rather than carrying
    # the full _WrittenShard records through coord.outcome.results.
    sidecar_glob = f"{output_path.rstrip('/')}/_done/shard-*.json"
    sidecar_paths = sorted(str(m) for m in StoragePath(sidecar_glob).glob())
    logger.info("build_clustered_store: loading %d shard sidecars (parallel)", len(sidecar_paths))

    def _load_one(sp: str) -> list[_WrittenShard]:
        sp_url = sp if sp.startswith("gs://") else f"gs://{sp}"
        return _load_shard_sidecar(sp_url) or []

    # Sequential ``_load_shard_sidecar`` over O(100K) sidecars at ~50-100ms
    # per GCS GET runs into hours of wall-clock. The fetches are independent
    # JSON reads, so a bounded threadpool collapses it to minutes.
    all_written: list[_WrittenShard] = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        for recs in pool.map(_load_one, sidecar_paths):
            all_written.extend(recs)
    logger.info("build_clustered_store: %d per-bucket records across %d shards", len(all_written), len(sidecar_paths))

    buckets = _merge_per_bucket_ledgers(map_shard_results=all_written, output_path=output_path)

    tokenizer = next(iter(tokenize.values())).tokenizer
    artifact = ClusteredStoreData(
        cache_path=output_path,
        cluster_view=cluster_view,
        quality_thresholds=list(_QUALITY_THRESHOLDS),
        split=split,
        buckets=buckets,
        source_names=sorted(tokenize),
        tokenizer=tokenizer,
        counters=counters,
    )
    write_artifact(artifact, output_path)
    return artifact
