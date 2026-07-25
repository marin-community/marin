# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffle-based datakit -> per-(cluster, quality) Levanter store.

Alternative to :mod:`experiments.datakit.store.datakit_store` that fixes the
object-count explosion of the map-side build (marin#6687). The map-side build
writes one tiny Levanter leaf cache per ``(input_shard, cluster, quality)`` and
never merges them -- the v0.1 store is ~14.2M leaf caches / ~70M GCS objects,
each holding a few KB, which also makes the dataloader open thousands of tiny
tensorstores per bucket.

This variant routes each surviving doc through a Zephyr ``group_by`` keyed by
``(cluster, quality, sub)``, so a single reducer streams ALL of one bucket-shard's
docs into ONE large materialized Levanter cache. No tiny leaves are ever
created -- the only intermediate is the scatter spill (~84K files for the full
store, vs 14.2M leaves), and the final store is ~200 buckets x ``sub`` large
caches. A measured 1%-stride benchmark put the full shuffle at ~17 TB compressed
scatter / ~2 h wall at ~2k workers, all in-region (no egress), with the cost
dominated by the same token I/O the map+compaction alternative would pay.

Pipeline:

1. **map** (per input shard): the same 5-way join + filter as the map-side
   build (:func:`_iter_surviving_docs` reuses ``datakit_store``'s loaders),
   emitting ``{cluster, quality, sub, input_ids}`` per surviving doc. ``sub``
   is a stable hash of the doc id mod that bucket's subshard count, so a hot
   bucket is split evenly across many reducers instead of one.
2. **group_by** ``(cluster, quality, sub)`` -> **reduce**: each reducer streams
   its group into one materialized cache at
   ``<output>/cluster=<C>/quality=<Q>/sub=<S>`` via ``SerialCacheWriter``.
3. **driver merge**: group reducer stats by ``(cluster, quality)`` and write the
   small per-bucket ``shard_ledger.json`` over the bucket's ``sub=*`` caches
   (sharded layout), exactly like the map-side build's final merge.

Skew control: pass ``bucket_token_hint`` (e.g. a prior build's per-bucket token
counts via :func:`bucket_token_hint_from_artifact`) so big buckets get many
subshards and small buckets stay at one -- avoiding both a single multi-hour
reducer on the 651B-token bucket AND re-proliferating tiny caches on small
buckets.
"""

import dataclasses
import logging
import math
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
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
from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem import url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.shard_keys import deterministic_hash
from zephyr.writers import atomic_rename

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.store.datakit_store import (
    _TOKENIZE_BATCH_SIZE,
    BucketCacheStats,
    ClusteredStoreData,
    _load_cluster_table,
    _load_decon_table,
    _load_dedup_canonical,
    _load_quality_table,
    _per_source_shard_tuples,
    _resolve_dedup_attr_dir,
    _validate_cluster_view,
)

logger = logging.getLogger(__name__)


# Records flushed to the SerialCacheWriter at a time on the reduce side. Bounds
# reducer memory at ~_WRITE_FLUSH * avg-doc-bytes regardless of group size, so
# even the hottest bucket-shard (~10B tokens with adequate subshards) streams
# in constant memory.
_WRITE_FLUSH = 1024

# Default skew-splitting target: aim for this many tokens per reduce cache.
# ~20B keeps the hottest reducer to a few hundred GB of token I/O.
_DEFAULT_TARGET_TOKENS_PER_SUBSHARD = 20_000_000_000


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


def _iter_surviving_docs(spec: dict[str, str], cluster_col: str) -> Iterator[tuple[int, int, str, np.ndarray]]:
    """Join one shard's five datasets; yield ``(cluster, quality_bucket, doc_id, input_ids)`` per surviving doc.

    Mirrors ``datakit_store._join_filter_stream_shard``'s per-shard join/filter:
    reads decon/cluster/quality densely, dedup sparsely, streams tokenize in
    positional lockstep, drops contaminated rows and dedup-cluster non-canonicals.
    Fails loud on missing/misaligned inputs -- a real build requires complete
    co-partitioned inputs.
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
    # Equal row counts don't imply equal id ORDER: verify the dense tables align
    # id-for-id before the positional-lockstep join routes docs (same guard the
    # map-side builder applies). Then drop the id arrays -- the loop only needs
    # the tokenize id (for the dedup lookup).
    where = f"{spec['source_name']}/{spec['basename']}"
    if not pc.all(pc.equal(decon_ids, cluster_ids)).as_py():
        raise RuntimeError(f"{where}: decon/cluster id mismatch -- co-partitioning broken")
    if not pc.all(pc.equal(decon_ids, quality_ids)).as_py():
        raise RuntimeError(f"{where}: decon/quality id mismatch -- co-partitioning broken")
    del decon_ids, cluster_ids, quality_ids
    dedup_canonical = _load_dedup_canonical(spec["dedup"])

    fs, resolved = url_to_fs(spec["tokenize"])
    with fs.open(resolved, "rb") as fh:
        pf = pq.ParquetFile(fh)
        row_idx = 0
        for batch in pf.iter_batches(batch_size=_TOKENIZE_BATCH_SIZE, columns=["id", "input_ids"]):
            tok_ids = batch.column("id").to_pylist()
            tok_input_ids = batch.column("input_ids")
            batch_len = len(tok_ids)
            contam_slice = contaminated[row_idx : row_idx + batch_len]
            cluster_slice = cluster_vals[row_idx : row_idx + batch_len]
            bucket_slice = quality_buckets[row_idx : row_idx + batch_len]
            row_idx += batch_len
            for i, doc_id in enumerate(tok_ids):
                if contam_slice[i]:
                    continue
                if dedup_canonical.get(doc_id) is False:
                    continue
                ids = tok_input_ids[i].values.to_numpy()
                yield int(cluster_slice[i]), int(bucket_slice[i]), doc_id, ids
        if row_idx != n_decon:
            raise RuntimeError(
                f"{spec['source_name']}/{spec['basename']}: tokenize rows ({row_idx}) != "
                f"decon rows ({n_decon}) -- co-partitioning broken"
            )


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
    n_out = 0
    n_tokens = 0
    for spec in batch_specs:
        for cluster, quality, doc_id, ids in _iter_surviving_docs(spec, cluster_col):
            k = subshards_for_bucket.get((cluster, quality), default_subshards)
            sub = deterministic_hash(doc_id) % k if k > 1 else 0
            n_out += 1
            n_tokens += len(ids)
            yield {"cluster": cluster, "quality": quality, "sub": sub, "input_ids": ids}
    counters.pipeline.update_counter("datakit_shuffle/records_out", n_out)
    counters.pipeline.update_counter("datakit_shuffle/tokens_out", n_tokens)


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
    counters.pipeline.update_counter("datakit_shuffle/reduce_rows", ledger.total_num_rows)
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

    Use as ``bucket_token_hint`` for :func:`build_clustered_store_shuffle` so the
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

    ``ceil(tokens / target)`` clamped to ``[1, max_subshards]``. Buckets with no
    hint fall back to ``default_subshards``. With no hint at all, every bucket
    uses ``default_subshards`` -- pass ``default_subshards`` deliberately (1 risks
    a single multi-hour reducer on the biggest bucket).
    """
    if not bucket_token_hint:
        logger.warning(
            "build_clustered_store_shuffle: no bucket_token_hint; every bucket uses default_subshards=%d. "
            "Pass bucket_token_hint_from_artifact(<prior store>) to split hot buckets.",
            default_subshards,
        )
        return {}
    plan = {}
    for key, tokens in bucket_token_hint.items():
        plan[key] = max(1, min(max_subshards, math.ceil(tokens / target_tokens_per_subshard)))
    logger.info(
        "build_clustered_store_shuffle: subshard plan over %d buckets, max=%d, total reduce caches=%d",
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


def build_clustered_store_shuffle(
    *,
    tokenize: dict[str, TokenizedAttrData],
    decontam: dict[str, DeconAttributes],
    cluster_assign: dict[str, AssignmentAttrData],
    quality: dict[str, QualityScores],
    dedup: FuzzyDupsAttrData,
    output_path: str,
    cluster_view: int = 40,
    split: str = "train",
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 4096,
    shards_per_task: int = 1,
    reduce_shards: int = 2048,
    bucket_token_hint: dict[tuple[int, int], int] | None = None,
    target_tokens_per_subshard: int = _DEFAULT_TARGET_TOKENS_PER_SUBSHARD,
    max_subshards: int = 128,
    default_subshards: int = 1,
) -> ClusteredStoreData:
    """Shuffle 5-way join + filter into one materialized cache per ``(cluster, quality, sub)``.

    Same inputs and output artifact as
    :func:`experiments.datakit.store.datakit_store.build_clustered_store`, but the
    store is born compact (no millions of tiny leaf caches). See the module
    docstring for the pipeline.

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
        raise ValueError("build_clustered_store_shuffle: tokenize is empty")
    for label, d in (("decontam", decontam), ("cluster_assign", cluster_assign), ("quality", quality)):
        if set(d) != set(tokenize):
            missing = sorted(set(tokenize) - set(d))
            extra = sorted(set(d) - set(tokenize))
            raise ValueError(f"{label} source set must equal tokenize: missing={missing!r}, extra={extra!r}")
    if shards_per_task < 1:
        raise ValueError(f"shards_per_task must be >= 1, got {shards_per_task}")

    # Every source must share one quality model, so its ``bucket_edges`` describe
    # the whole store (same invariant as the map-side build).
    models = {(q.model_dir, q.calib_file, tuple(q.bucket_edges)) for q in quality.values()}
    if len(models) != 1:
        raise ValueError(f"build_clustered_store_shuffle: sources span multiple quality models: {sorted(models)}")
    bucket_edges = next(iter(quality.values())).bucket_edges

    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)
    subshards_for_bucket = _plan_subshards(
        bucket_token_hint=bucket_token_hint,
        target_tokens_per_subshard=target_tokens_per_subshard,
        max_subshards=max_subshards,
        default_subshards=default_subshards,
    )

    # Resolve the flat per-source-shard spec list (same as the map-side build).
    shard_specs: list[dict[str, str]] = []
    for source_name in sorted(tokenize):
        tok = tokenize[source_name]
        main_dir = tok.source_main_dirs.get(split)
        if main_dir is None:
            raise ValueError(f"{source_name}: tokenize has no source_main_dir for split={split!r}")
        cluster_asg = cluster_assign[source_name]
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
                decontam=decontam[source_name],
                cluster_assign=cluster_asg,
                quality=quality[source_name],
                dedup_attr_dir=dedup_attr_dir,
                split=split,
            )
        )
    if not shard_specs:
        raise ValueError("No input shards resolved -- nothing to do")

    batched_specs = [shard_specs[i : i + shards_per_task] for i in range(0, len(shard_specs), shards_per_task)]
    logger.info(
        "build_clustered_store_shuffle: %d sources, %d input shards -> %d map tasks, reduce_shards=%d -> %s",
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

    ctx = ZephyrContext(
        resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
        max_workers=min(max_workers, len(batched_specs)),
        name="datakit-clustered-store-shuffle",
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
    outcome = ctx.execute(ds, verbose=True)
    subshard_stats = [r for r in outcome.results if r is not None]
    logger.info(
        "build_clustered_store_shuffle: wrote %d subshard caches (records_out=%d, tokens_out=%d)",
        len(subshard_stats),
        outcome.counters.get("datakit_shuffle/records_out", 0),
        outcome.counters.get("datakit_shuffle/tokens_out", 0),
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
