# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit -> per-(cluster, quality) Levanter store via a shuffle.

The store routes each surviving doc through a Zephyr ``group_by`` keyed by
``(cluster, quality, sub)``, so a single reducer streams all of one bucket-shard's
documents into one materialized Levanter cache. No per-input-shard leaves are
created. The only intermediate is the scatter spill (~84K files for the full
store), and the final store is bounded by the configured number of subshards per
bucket. A measured 1%-stride benchmark put the full shuffle at ~17 TB compressed
scatter / ~2 h wall at ~2k workers, all in-region (no egress), with the cost
dominated by token I/O.

Pipeline:

1. **map** (per input shard): a positional join over tokenization,
   decontamination, domain assignment, quality, exact-dedup, and fuzzy-dedup
   attributes, emitting ``{cluster, quality, sub, input_ids}`` per surviving
   doc. ``sub`` is a stable hash of the doc id mod that bucket's subshard count,
   so a hot bucket is split evenly across many reducers instead of one.
2. **group_by** ``(cluster, quality, sub)`` -> **reduce**: each reducer streams
   its group into one materialized cache at
   ``<output>/cluster=<C>/quality=<Q>/sub=<S>`` via ``SerialCacheWriter``.
3. **driver merge**: group reducer stats by ``(cluster, quality)`` and write the
   small per-bucket ``shard_ledger.json`` over the bucket's ``sub=*`` caches.

Without a prior store artifact, each bucket uses ``default_subshards`` (32 in
the production reference pipeline, one in smoke mode). Direct callers may pass
``bucket_token_hint`` from :func:`bucket_token_hint_from_artifact` to size each
bucket independently.

Zephyr retries individual map and reduce tasks, but the driver-side ledger merge
is not separately checkpointed. If the store driver dies after the shuffle
finishes and before the artifact is written, rerunning the store repeats the
shuffle.
"""

import dataclasses
import logging
import math
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping

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
from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.tokenize._core import CHUNK_INDEX_FIELD
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    VerifiedFuzzyDupsAttrData,
    VerifiedFuzzyDupsPerSource,
)
from marin.processing.tokenize.attributes import TokenizedAttrData
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.shard_keys import deterministic_hash
from zephyr.writers import atomic_rename

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.global_exact_dedup import ExactDupsPerSource, GlobalExactDedupData

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

    version: str = "v3"
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


# Records flushed to the SerialCacheWriter at a time on the reduce side. Bounds
# reducer memory at ~_WRITE_FLUSH * avg-doc-bytes regardless of group size, so
# even the hottest bucket-shard (~10B tokens with adequate subshards) streams
# in constant memory.
_WRITE_FLUSH = 1024

# Default skew-splitting target: aim for this many tokens per reduce cache.
# ~20B keeps the hottest reducer to a few hundred GB of token I/O.
DEFAULT_TARGET_TOKENS_PER_SUBSHARD = 20_000_000_000

# Without a prior store artifact to size buckets from, split every bucket enough
# to keep the known ~651B-token hot bucket from becoming one multi-hour reducer.
DEFAULT_SUBSHARDS = 32

# Rows read from a tokenized shard at once during the positional join.
_TOKENIZE_BATCH_SIZE = 8192


@dataclasses.dataclass(frozen=True)
class _SubshardStat:
    """One reducer's materialized ``(cluster, quality, sub)`` cache summary, returned to the driver."""

    cluster: int
    quality: int
    sub: int
    path: str
    rows: int
    tokens: int


# ---------------------------------------------------------------------------
# Map side: join + filter -> per-doc shuffle records.
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
                        yield doc_id, chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
                    doc_id, chunks = row_id, []
                elif row_id != doc_id or chunk_indices[i] != len(chunks):
                    raise RuntimeError(
                        f"{path}: row {i} is chunk {chunk_indices[i]} of {row_id}, but chunk "
                        f"{len(chunks)} of {doc_id} must come next"
                    )
                chunks.append(input_ids[i].values.to_numpy())
        if chunks:
            yield doc_id, chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


def _iter_surviving_docs(spec: dict[str, str], cluster_col: str) -> Iterator[tuple[int, int, str, np.ndarray]]:
    """Join one shard's datasets; yield ``(cluster, quality_bucket, doc_id, input_ids)`` per surviving doc.

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
    # before routing positionally, then drop their ID arrays; the loop only needs
    # tokenization IDs for the dedup lookup.
    where = f"{spec['source_name']}/{spec['basename']}"
    if not pc.all(pc.equal(decon_ids, cluster_ids)).as_py():
        raise RuntimeError(f"{where}: decon/cluster id mismatch -- co-partitioning broken")
    if not pc.all(pc.equal(decon_ids, quality_ids)).as_py():
        raise RuntimeError(f"{where}: decon/quality id mismatch -- co-partitioning broken")
    del decon_ids, cluster_ids, quality_ids
    exact_duplicates = _load_exact_duplicates(spec["exact_dedup"])
    verified_duplicates = _load_verified_duplicates(spec["dedup"])

    n_in = 0
    n_contaminated = 0
    n_exact_dedup_dropped = 0
    n_dedup_dropped = 0
    n_out = 0
    doc_idx = 0
    for doc_id, ids in _iter_tokenized_documents(spec["tokenize"]):
        if doc_idx >= n_decon:
            raise RuntimeError(
                f"{where}: tokenize holds more documents than decon rows ({n_decon}) -- co-partitioning broken"
            )
        n_in += 1
        position, doc_idx = doc_idx, doc_idx + 1
        if contaminated[position]:
            n_contaminated += 1
            continue
        # Verified attributes mark only the members a full-text comparison
        # confirmed, so membership alone decides. The unverified candidate
        # attributes needed a tri-state check because a cluster member that was
        # merely a candidate still had to reach the exact-dedup test.
        if doc_id in verified_duplicates:
            n_dedup_dropped += 1
            continue
        if doc_id in exact_duplicates:
            n_exact_dedup_dropped += 1
            continue
        n_out += 1
        yield int(cluster_vals[position]), int(quality_buckets[position]), doc_id, ids
    if doc_idx != n_decon:
        raise RuntimeError(
            f"{where}: tokenize documents ({doc_idx}) != decon rows ({n_decon}) -- co-partitioning broken"
        )
    counters.pipeline.update_counter("datakit_store/records_in", n_in)
    counters.pipeline.update_counter("datakit_store/contaminated_dropped", n_contaminated)
    counters.pipeline.update_counter("datakit_store/exact_duplicate_dropped", n_exact_dedup_dropped)
    counters.pipeline.update_counter("datakit_store/fuzzy_duplicate_dropped", n_dedup_dropped)
    counters.pipeline.update_counter("datakit_store/records_out", n_out)


def _emit_for_shuffle(
    items: Iterator[list[dict[str, str]]],
    _shard_info: ShardInfo,
    *,
    cluster_col: str,
    subshards_for_bucket: dict[tuple[int, int], int],
    default_subshards: int,
) -> Iterator[dict[str, object]]:
    """Map one task (a batch of source shards) to per-doc shuffle records.

    Yields ``{cluster, quality, sub, input_ids}``. ``sub`` is a stable hash of
    the doc id mod that bucket's subshard count (``subshards_for_bucket`` for
    hinted buckets, else ``default_subshards``), so a bucket's docs spread evenly
    across that many reducers regardless of how the docs are partitioned across
    map tasks (a per-task counter would pile every task's first doc onto ``sub=0``).
    """
    batch_specs = next(iter(items))
    n_tokens = 0
    for spec in batch_specs:
        for cluster, quality, doc_id, ids in _iter_surviving_docs(spec, cluster_col):
            k = subshards_for_bucket.get((cluster, quality), default_subshards)
            sub = deterministic_hash(doc_id) % k if k > 1 else 0
            n_tokens += len(ids)
            yield {"cluster": cluster, "quality": quality, "sub": sub, "input_ids": ids}
    counters.pipeline.update_counter("datakit_store/tokens_out", n_tokens)


# ---------------------------------------------------------------------------
# Reduce side: one group -> one materialized Levanter cache.
# ---------------------------------------------------------------------------


def _write_subshard_cache(
    key: tuple[int, int, int],
    group: Iterator[dict[str, object]],
    *,
    output_path: str,
) -> _SubshardStat:
    """Stream one ``(cluster, quality, sub)`` group into a materialized Levanter cache.

    Writes to ``<output>/cluster=<C>/quality=<Q>/sub=<S>`` via ``SerialCacheWriter``
    in ``_WRITE_FLUSH``-record batches (constant memory). Returns the slim stat
    the driver needs to build the per-bucket sharded ledger.
    """
    cluster, quality, sub = key
    cache_dir = f"{output_path.rstrip('/')}/cluster={cluster}/quality={quality}/sub={sub}"

    # group_by invokes reducers only for keys that received at least one record.
    it = iter(group)
    first = next(it)
    exemplar = {"input_ids": first["input_ids"]}

    with atomic_rename(cache_dir) as tmp_path:
        with SerialCacheWriter(tmp_path, exemplar, shard_name=cache_dir, metadata=CacheMetadata.empty()) as writer:
            buf: list[dict[str, object]] = [exemplar]
            for rec in it:
                buf.append({"input_ids": rec["input_ids"]})
                if len(buf) >= _WRITE_FLUSH:
                    writer.write_batch(buf)
                    buf = []
            if buf:
                writer.write_batch(buf)

    # SerialCacheWriter committed the ledger on clean exit; load it back so the
    # driver can merge without re-reading the tensorstore.
    ledger = CacheLedger.load(cache_dir, CacheMetadata.empty())
    counters.pipeline.update_counter("datakit_store/reduce_rows", ledger.total_num_rows)
    return _SubshardStat(
        cluster=cluster,
        quality=quality,
        sub=sub,
        path=cache_dir,
        rows=ledger.total_num_rows,
        tokens=ledger.field_counts.get("input_ids", 0),
    )


# ---------------------------------------------------------------------------
# Subshard planning + driver-side per-bucket ledger merge.
# ---------------------------------------------------------------------------


def bucket_token_hint_from_artifact(artifact_path: str) -> dict[tuple[int, int], int]:
    """Load a prior :class:`ClusteredStoreData` and return ``{(cluster, quality): total_tokens}``.

    Use as ``bucket_token_hint`` for :func:`build_clustered_store` so the
    next build splits hot buckets proportionally to last build's token mass.
    """
    prior = read_artifact(artifact_path, ClusteredStoreData)
    return {(b.cluster_id, b.quality_bucket): b.total_tokens for b in prior.buckets}


def _plan_subshards(
    *,
    bucket_token_hint: dict[tuple[int, int], int] | None,
    target_tokens_per_subshard: int,
    max_subshards: int,
    default_subshards: int,
) -> dict[tuple[int, int], int]:
    """Map each bucket to a subshard count from its hinted token mass.

    ``ceil(tokens / target)`` clamped to ``[1, max_subshards]``. Buckets without
    a hint use ``default_subshards``; without any hint, every bucket uses that
    uniform count.
    """
    if not bucket_token_hint:
        logger.warning(
            "build_clustered_store: no bucket_token_hint; every bucket uses default_subshards=%d. "
            "Pass bucket_token_hint_from_artifact(<prior store>) to split hot buckets.",
            default_subshards,
        )
        return {}
    plan = {}
    for key, tokens in bucket_token_hint.items():
        plan[key] = max(1, min(max_subshards, math.ceil(tokens / target_tokens_per_subshard)))
    logger.info(
        "build_clustered_store: subshard plan over %d buckets, max=%d, total reduce caches=%d",
        len(plan),
        max(plan.values(), default=0),
        sum(plan.values()),
    )
    return plan


def _merge_per_bucket_ledgers(
    *,
    subshard_stats: list[_SubshardStat],
    output_path: str,
) -> list[BucketCacheStats]:
    """Write one per-bucket ``shard_ledger.json`` over each bucket's ``sub=*`` caches.

    Pure driver-side work: each stat already carries its cache's row + token
    counts, so we synthesize the minimal ``CacheLedger`` stubs ``_merge_sharded_ledgers``
    needs (it only reads ``total_num_rows``) and call it per bucket.
    """
    by_bucket: dict[tuple[int, int], list[_SubshardStat]] = defaultdict(list)
    for s in subshard_stats:
        by_bucket[(s.cluster, s.quality)].append(s)

    metadata = CacheMetadata.empty()
    base_path = output_path.rstrip("/")
    buckets: list[BucketCacheStats] = []
    for key in sorted(by_bucket):
        cluster, quality = key
        bucket_root = f"{base_path}/cluster={cluster}/quality={quality}"
        subs = sorted(by_bucket[key], key=lambda s: s.sub)
        shard_paths = [s.path for s in subs]
        shard_ledgers = [
            CacheLedger(total_num_rows=s.rows, shard_rows={}, finished_shards=[], field_counts={}, metadata=metadata)
            for s in subs
        ]
        per_shard_field_counts = [{"input_ids": s.tokens} for s in subs]
        ledger = _merge_sharded_ledgers(bucket_root, shard_paths, shard_ledgers, per_shard_field_counts, metadata)
        total_tokens = ledger.field_counts.get("input_ids", 0)
        buckets.append(
            BucketCacheStats(
                cluster_id=cluster,
                quality_bucket=quality,
                path=bucket_root,
                total_elements=ledger.total_num_rows,
                total_tokens=total_tokens,
                n_shards=len(shard_paths),
            )
        )
        logger.info(
            "cluster=%d quality=%d: docs=%d tokens=%d subshards=%d -> %s",
            cluster,
            quality,
            ledger.total_num_rows,
            total_tokens,
            len(shard_paths),
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
    shards_per_task: int = 1,
    reduce_shards: int = 2048,
    bucket_token_hint: dict[tuple[int, int], int] | None = None,
    target_tokens_per_subshard: int = DEFAULT_TARGET_TOKENS_PER_SUBSHARD,
    max_subshards: int = 128,
    default_subshards: int = DEFAULT_SUBSHARDS,
    zephyr_context: ZephyrContext | None = None,
) -> ClusteredStoreData:
    """Shuffle the joined attributes into one materialized cache per ``(cluster, quality, sub)``.

    The store is born compact: reducers create the final materialized caches
    directly rather than producing per-input-shard leaf caches.

    Args:
        shards_per_task: Source shards per map task (batches reduce the map task
            count; does not affect the shuffle output).
        reduce_shards: ``num_output_shards`` for the ``group_by`` -- the number of
            reduce tasks the ~``sum(subshards)`` groups are spread across.
        bucket_token_hint: ``{(cluster, quality): tokens}`` used to size each
            bucket's subshard count (see :func:`bucket_token_hint_from_artifact`).
        target_tokens_per_subshard / max_subshards / default_subshards: subshard
            sizing knobs (see :func:`_plan_subshards`).
    """
    if not tokenize:
        raise ValueError("build_clustered_store: tokenize is empty")
    for label, d in (("decontam", decontam), ("cluster_assign", cluster_assign), ("quality", quality)):
        if set(d) != set(tokenize):
            missing = sorted(set(tokenize) - set(d))
            extra = sorted(set(d) - set(tokenize))
            raise ValueError(f"{label} source set must equal tokenize: missing={missing!r}, extra={extra!r}")
    if shards_per_task < 1:
        raise ValueError(f"shards_per_task must be >= 1, got {shards_per_task}")
    if reduce_shards < 1:
        raise ValueError(f"reduce_shards must be >= 1, got {reduce_shards}")
    if target_tokens_per_subshard < 1:
        raise ValueError(f"target_tokens_per_subshard must be >= 1, got {target_tokens_per_subshard}")
    if max_subshards < 1:
        raise ValueError(f"max_subshards must be >= 1, got {max_subshards}")
    if default_subshards < 1:
        raise ValueError(f"default_subshards must be >= 1, got {default_subshards}")

    # Every source must share one quality model so bucket IDs are comparable.
    models = {(q.model_dir, q.calib_file, tuple(q.bucket_edges)) for q in quality.values()}
    if len(models) != 1:
        raise ValueError(f"build_clustered_store: sources span multiple quality models: {sorted(models)}")
    bucket_edges = next(iter(quality.values())).bucket_edges

    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)
    subshards_for_bucket = _plan_subshards(
        bucket_token_hint=bucket_token_hint,
        target_tokens_per_subshard=target_tokens_per_subshard,
        max_subshards=max_subshards,
        default_subshards=default_subshards,
    )

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

    # Resolve the flat per-source-shard spec list.
    shard_specs: list[dict[str, str]] = []
    for source_name in sorted(tokenize):
        tok = tokenize[source_name]
        source_key = source_keys[source_name]
        cluster_asg = cluster_assign[source_name]
        if cluster_asg.source_key != source_key:
            raise ValueError(
                f"{source_name}: cluster_assign.source_key={cluster_asg.source_key!r} "
                f"!= tokenize.source_keys[{split!r}]={source_key!r}"
            )
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
        shard_specs.extend(
            _per_source_shard_tuples(
                source_name=source_name,
                tokenize=tok,
                decontam=decontam[source_name],
                cluster_assign=cluster_asg,
                quality=quality[source_name],
                exact_dedup_attr_dir=exact_dedup_attr_dir,
                dedup_attr_dir=dedup_attr_dir,
                split=split,
            )
        )
    if not shard_specs:
        raise ValueError("No input shards resolved -- nothing to do")

    batched_specs = [shard_specs[i : i + shards_per_task] for i in range(0, len(shard_specs), shards_per_task)]
    logger.info(
        "build_clustered_store: %d sources, %d input shards -> %d map tasks, reduce_shards=%d -> %s",
        len(tokenize),
        len(shard_specs),
        len(batched_specs),
        reduce_shards,
        output_path,
    )

    if worker_resources is None:
        # 16g: the reduce side streams big groups through a tensorstore write
        # buffer (~512 MB write-chunk) and the map holds numpy token payloads.
        worker_resources = ResourceConfig(cpu=2, ram="16g", disk="16g")

    ctx = zephyr_context or ZephyrContext(
        resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
        max_workers=min(max_workers, len(batched_specs)),
        name="datakit-clustered-store",
    )
    ds = (
        Dataset.from_list(batched_specs)
        .map_shard(
            lambda items, shard, cc=cluster_col, sfb=subshards_for_bucket, ds=default_subshards: _emit_for_shuffle(
                items, shard, cluster_col=cc, subshards_for_bucket=sfb, default_subshards=ds
            )
        )
        .group_by(
            key=lambda r: (r["cluster"], r["quality"], r["sub"]),
            reducer=lambda key, group, op=output_path: _write_subshard_cache(key, group, output_path=op),
            num_output_shards=reduce_shards,
        )
    )
    outcome = ctx.execute(
        ds,
        verbose=True,
        map_task_resources=worker_resources,
    )
    subshard_stats = [r for r in outcome.results if r is not None]
    logger.info(
        "build_clustered_store: wrote %d subshard caches (records_out=%d, tokens_out=%d)",
        len(subshard_stats),
        outcome.counters.get("datakit_store/records_out", 0),
        outcome.counters.get("datakit_store/tokens_out", 0),
    )

    buckets = _merge_per_bucket_ledgers(subshard_stats=subshard_stats, output_path=output_path)

    tokenizer = next(iter(tokenize.values())).tokenizer
    artifact = ClusteredStoreData(
        cache_path=output_path,
        cluster_view=cluster_view,
        bucket_edges=bucket_edges,
        split=split,
        buckets=buckets,
        source_names=sorted(tokenize),
        tokenizer=tokenizer,
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_path)
    return artifact
