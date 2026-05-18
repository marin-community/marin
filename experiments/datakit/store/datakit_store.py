# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit -> per-cluster Levanter store: single map-side Zephyr pass.

Shape mirrors :mod:`marin.processing.classification.consolidate`, specialized
for the datakit attribute datasets that the global pipelines produce:

    tokenize        per-source ``{id, input_ids}``, dense, sorted by id
    decontam        per-source ``{id, attributes: {contaminated, ...}}``, dense
    cluster_assign  per-source ``{id, cluster_<K>, ...}``, dense
    dedup           per-source ``{id, attributes: {is_cluster_canonical, ...}}``,
                    SPARSE -- singletons omitted by ``compute_fuzzy_dups_attrs``

All four are co-partitioned with the source ``NormalizedData`` by basename, so
a single map-side pass joins them per shard with no shuffle. The pass:

1. Reads decon and cluster_assign fully (small columns; one shard's worth).
2. Reads dedup into ``{id -> is_canonical}`` if present.
3. Streams tokenize via ``ParquetFile.iter_batches`` in positional lockstep
   with decon and cluster_assign (sanity-asserts id alignment).
4. Drops contaminated rows; drops dedup-cluster non-canonicals (rows missing
   from dedup are singletons -> kept).
5. Routes each surviving row by ``cluster_<view>`` directly into one of K
   lazily-opened ``SerialCacheWriter`` instances (one per non-empty cluster),
   writing under ``<output>/cluster=<C>/part-NNNNN-of-MMMMM``. Memory peak
   stays at K * ``_BATCH_FLUSH`` * avg-doc-size (~tens of MB) -- independent
   of input-shard size, so even multi-GB tokenize shards stream cleanly.
6. After ``ExitStack`` closes the writers (committing per-shard ledgers),
   loads each just-written ledger (``SerialCacheWriter`` already populates
   ``field_counts``) and yields it back to the driver.

After the Zephyr execute, the driver groups results by ``cluster_id`` in a
plain Python dict (no Zephyr group_by, no shuffle) and calls
``_merge_sharded_ledgers`` per cluster -- each call only writes the small
``<output>/cluster=<C>/shard_ledger.json``. No second Zephyr context.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
from collections import defaultdict
from collections.abc import Iterator

import pyarrow.parquet as pq
from fray import ResourceConfig
from levanter.store.cache import (
    CacheLedger,
    CacheMetadata,
    SerialCacheWriter,
    _merge_sharded_ledgers,
)
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import Artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from marin.utils import fsspec_exists, fsspec_glob
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ZephyrContext, counters
from zephyr.dataset import ShardInfo, format_shard_path
from zephyr.writers import atomic_rename

from experiments.datakit.cluster.v0.assign import AssignmentAttrData

logger = logging.getLogger(__name__)


class ClusterCacheStats(BaseModel):
    """Per-cluster Levanter cache stats inside :class:`ClusteredStoreData`."""

    cluster_id: int
    path: str
    total_elements: int
    total_tokens: int
    n_shards: int


class ClusteredStoreData(BaseModel):
    """Outcome of :func:`build_clustered_store`: one Levanter cache per cluster.

    Persisted as ``<output_path>/artifact.json``. Load via
    ``Artifact.from_path(output_path, ClusteredStoreData)``.

    Attributes:
        cache_path: Root directory. Each cluster's cache lives at
            ``cache_path/cluster=<C>/``.
        cluster_view: Cluster-K used to partition the store
            (column name ``cluster_<view>`` read from the assignment shards).
        split: Tokenize split fed into the store (e.g. ``"train"``).
        clusters: Map from cluster id to per-cluster stats. Clusters that
            received zero records across every input shard are omitted.
        source_names: Registry names of the sources that contributed.
        tokenizer: Tokenizer name copied from the first tokenize source.
        counters: Aggregated zephyr counters across the join pass.
    """

    version: str = "v1"
    cache_path: str
    cluster_view: int
    split: str
    clusters: dict[int, ClusterCacheStats]
    source_names: list[str]
    tokenizer: str
    counters: dict[str, int]


# ---------------------------------------------------------------------------
# Shard input spec (per-source-shard 4-tuple of paths + global indexing).
# ---------------------------------------------------------------------------


def _per_source_shard_tuples(
    *,
    source_name: str,
    tokenize: TokenizedAttrData,
    decontam: DeconAttributes,
    cluster_assign: AssignmentAttrData,
    dedup_attr_dir: str,
    split: str,
) -> list[dict[str, str]]:
    """Align the four datasets' parquet shards for one source by basename.

    Returns a list of ``{tokenize, decontam, dedup, cluster, source_name, basename}``
    dicts -- one per source shard. ``dedup`` may be ``None`` for source shards
    whose dedup file is absent (sparse: a shard with zero non-singletons is
    legitimately missing).
    """
    tok_dir = tokenize.output_dirs.get(split)
    if tok_dir is None:
        raise FileNotFoundError(f"{source_name}: tokenize has no split={split!r}")
    tok_shards = sorted(fsspec_glob(f"{tok_dir.rstrip('/')}/*.parquet"))
    if not tok_shards:
        raise FileNotFoundError(f"{source_name}: no tokenize shards under {tok_dir}")

    # We trust the datakit co-partitioning invariant: tokenize / decon /
    # cluster_assign / dedup all mirror the source NormalizedData basenames.
    # Workers fail loud if a per-shard file is missing -- cheaper than
    # ~2 x N_shards serial GCS HEAD requests up front (would take tens of
    # minutes at full-fleet scale).
    decon_dir = decontam.output_dir.rstrip("/")
    cluster_dir = cluster_assign.output_dir.rstrip("/")
    dedup_dir = dedup_attr_dir.rstrip("/")
    return [
        {
            "tokenize": tok_path,
            "decontam": f"{decon_dir}/{os.path.basename(tok_path)}",
            "cluster": f"{cluster_dir}/{os.path.basename(tok_path)}",
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


def _load_decon_table(path: str) -> tuple[list[str], list[bool]]:
    """Return ``(ids, contaminated)`` arrays for one decon shard, in row order."""
    table = pq.read_table(path, columns=["id", "attributes"])
    ids = table.column("id").to_pylist()
    contaminated = table.column("attributes").combine_chunks().field("contaminated").to_pylist()
    return ids, contaminated


def _load_cluster_table(path: str, cluster_col: str) -> tuple[list[str], list[int]]:
    """Return ``(ids, cluster)`` arrays for one cluster-assign shard, in row order."""
    table = pq.read_table(path, columns=["id", cluster_col])
    ids = table.column("id").to_pylist()
    cluster = table.column(cluster_col).to_pylist()
    return ids, cluster


def _load_dedup_canonical(path: str) -> dict[str, bool]:
    """Return ``{id -> is_cluster_canonical}`` for one dedup shard, or ``{}`` if absent.

    Dedup is sparse: ids missing from this dict are singletons (kept). Ids
    present are non-singleton cluster members; only the canonical one survives.
    """
    if not fsspec_exists(path):
        return {}
    table = pq.read_table(path, columns=["id", "attributes"])
    ids = table.column("id").to_pylist()
    canonical = table.column("attributes").combine_chunks().field("is_cluster_canonical").to_pylist()
    return dict(zip(ids, canonical, strict=True))


@dataclasses.dataclass(frozen=True)
class _WrittenShard:
    """Slim summary of one per-(input_shard, cluster) Levanter shard cache.

    Only the fields the driver actually needs for ``_merge_sharded_ledgers``
    travel from worker to driver -- the full ``CacheLedger`` would carry
    ``shard_rows`` / ``finished_shards`` / ``field_counts_by_shard`` per
    record (10-20x larger), and at full-fleet scale (~17K input shards x
    ~20 non-empty clusters each = ~340K records) the resulting
    ``outcome.results`` payload starts to crowd the 1 GB iris coord.
    """

    cluster_id: int
    path: str
    total_num_rows: int
    field_counts: dict[str, int]


# Per-cluster pending-buffer flush threshold inside one map_shard task.
# Memory peak per worker ~ N_open_clusters * _BATCH_FLUSH * avg_doc_size --
# at K=40 and ~1 KB avg doc that's ~40 MB, independent of input-shard size.
# Anything smaller raises GCS request rate; anything larger reintroduces the
# whole-shard buffering we were trying to avoid.
_BATCH_FLUSH = 1024


def _join_filter_stream_shard(
    items: Iterator[dict[str, str]],
    shard_info: ShardInfo,
    *,
    cluster_col: str,
    output_path: str,
) -> Iterator[_WrittenShard]:
    """One input shard -> up to K Levanter shard caches under ``cluster=<C>/``.

    Streams records to K parallel :class:`SerialCacheWriter` instances opened
    lazily on first row for that cluster -- no whole-shard buffering. Memory
    bounded by ``K * _BATCH_FLUSH * avg_doc_size`` regardless of input-shard
    size. ExitStack guarantees writers (and their ``atomic_rename`` tmp
    paths) close cleanly on both success and failure.
    """
    spec = next(iter(items))
    source_name = spec["source_name"]
    tok_path = spec["tokenize"]
    decon_path = spec["decontam"]
    cluster_path = spec["cluster"]
    dedup_path = spec["dedup"]

    decon_ids, contaminated = _load_decon_table(decon_path)
    cluster_ids, cluster_vals = _load_cluster_table(cluster_path, cluster_col)
    if len(decon_ids) != len(cluster_ids):
        raise RuntimeError(
            f"{source_name}/{spec['basename']}: decon rows ({len(decon_ids)}) != "
            f"cluster rows ({len(cluster_ids)}) -- co-partitioning broken"
        )
    dedup_canonical = _load_dedup_canonical(dedup_path)

    n_in = 0
    n_contaminated = 0
    n_dedup_dropped = 0
    n_out = 0

    cluster_root = f"{output_path.rstrip('/')}"
    written: dict[int, str] = {}

    def _shard_dir(cluster_id: int) -> str:
        return format_shard_path(
            f"{cluster_root}/cluster={cluster_id}/part-{{shard:05d}}-of-{{total:05d}}",
            shard_info.shard_idx,
            shard_info.total_shards,
        )

    with contextlib.ExitStack() as stack:
        writers: dict[int, SerialCacheWriter] = {}
        pending: dict[int, list[dict[str, list[int]]]] = defaultdict(list)

        def get_writer(cluster_id: int, exemplar: dict[str, list[int]]) -> SerialCacheWriter:
            if cluster_id not in writers:
                cache_dir = _shard_dir(cluster_id)
                written[cluster_id] = cache_dir
                tmp_path = stack.enter_context(atomic_rename(cache_dir))
                writers[cluster_id] = stack.enter_context(
                    SerialCacheWriter(
                        tmp_path,
                        exemplar,
                        shard_name=cache_dir,
                        metadata=CacheMetadata.empty(),
                    )
                )
            return writers[cluster_id]

        def flush(cluster_id: int) -> None:
            buf = pending[cluster_id]
            if not buf:
                return
            get_writer(cluster_id, buf[0]).write_batch(buf)
            pending[cluster_id] = []

        fs, resolved = url_to_fs(tok_path)
        with fs.open(resolved, "rb") as fh:
            pf = pq.ParquetFile(fh)
            row_idx = 0
            for batch in pf.iter_batches(columns=["id", "input_ids"]):
                tok_ids = batch.column("id").to_pylist()
                tok_input_ids = batch.column("input_ids").to_pylist()
                batch_len = len(tok_ids)
                decon_slice_ids = decon_ids[row_idx : row_idx + batch_len]
                decon_slice_flags = contaminated[row_idx : row_idx + batch_len]
                cluster_slice_ids = cluster_ids[row_idx : row_idx + batch_len]
                cluster_slice_vals = cluster_vals[row_idx : row_idx + batch_len]
                row_idx += batch_len

                for i, doc_id in enumerate(tok_ids):
                    n_in += 1
                    if decon_slice_ids[i] != doc_id or cluster_slice_ids[i] != doc_id:
                        raise RuntimeError(
                            f"{source_name}/{spec['basename']}: id mismatch at row "
                            f"{row_idx - batch_len + i} (tokenize={doc_id!r}, "
                            f"decon={decon_slice_ids[i]!r}, cluster={cluster_slice_ids[i]!r}) "
                            "-- co-partitioning broken"
                        )
                    if decon_slice_flags[i]:
                        n_contaminated += 1
                        continue
                    if dedup_canonical.get(doc_id) is False:
                        n_dedup_dropped += 1
                        continue
                    # canonical True OR id missing from dedup (singleton) -> keep
                    cid = int(cluster_slice_vals[i])
                    pending[cid].append({"input_ids": tok_input_ids[i]})
                    n_out += 1
                    if len(pending[cid]) >= _BATCH_FLUSH:
                        flush(cid)

            if row_idx != len(decon_ids):
                raise RuntimeError(
                    f"{source_name}/{spec['basename']}: tokenize rows ({row_idx}) != "
                    f"decon rows ({len(decon_ids)}) -- co-partitioning broken"
                )

        # Flush remaining per-cluster buffers before ExitStack closes the writers.
        for cid in list(pending):
            flush(cid)

    # ExitStack done: SerialCacheWriter.__exit__ wrote each per-shard ledger;
    # atomic_rename.__exit__ renamed tmp_path -> cache_dir. Load each ledger
    # once so the driver can run _merge_sharded_ledgers without re-reading.
    counters.increment("datakit_store/records_in", n_in)
    counters.increment("datakit_store/contaminated_dropped", n_contaminated)
    counters.increment("datakit_store/dedup_noncanonical_dropped", n_dedup_dropped)
    counters.increment("datakit_store/records_out", n_out)

    metadata = CacheMetadata.empty()
    for cluster_id, cache_dir in written.items():
        ledger = CacheLedger.load(cache_dir, metadata)
        yield _WrittenShard(
            cluster_id=cluster_id,
            path=cache_dir,
            total_num_rows=ledger.total_num_rows,
            field_counts=dict(ledger.field_counts),
        )


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


def _merge_per_cluster_ledgers(
    *,
    map_shard_results: list[_WrittenShard],
    output_path: str,
) -> dict[int, ClusterCacheStats]:
    """Merge per-(input_shard, cluster) ledgers into one ledger per cluster.

    Pure driver-side work: ``map_shard_results`` already carries each
    written shard's ``total_num_rows`` + ``field_counts`` (loaded by the
    worker right after ``write_levanter_cache`` and projected to the slim
    :class:`_WrittenShard`). We group by ``cluster_id`` in a Python dict
    (no Zephyr group_by, no shuffle), synthesize minimal ``CacheLedger``
    stubs since ``_merge_sharded_ledgers`` only reads ``total_num_rows``,
    and call it -- which only writes the small
    ``cluster=<C>/shard_ledger.json`` per cluster.
    """
    by_cluster: dict[int, list[_WrittenShard]] = defaultdict(list)
    for r in map_shard_results:
        by_cluster[r.cluster_id].append(r)

    metadata = CacheMetadata.empty()
    clusters: dict[int, ClusterCacheStats] = {}
    for cluster_id in sorted(by_cluster):
        cluster_root = f"{output_path.rstrip('/')}/cluster={cluster_id}"
        entries = sorted(by_cluster[cluster_id], key=lambda e: e.path)
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
        ledger = _merge_sharded_ledgers(cluster_root, shard_paths, shard_ledgers, per_shard_field_counts, metadata)
        total_tokens = ledger.field_counts.get("input_ids", 0)
        clusters[cluster_id] = ClusterCacheStats(
            cluster_id=cluster_id,
            path=cluster_root,
            total_elements=ledger.total_num_rows,
            total_tokens=total_tokens,
            n_shards=len(shard_paths),
        )
        logger.info(
            "cluster=%d: docs=%d tokens=%d shards=%d -> %s",
            cluster_id,
            ledger.total_num_rows,
            total_tokens,
            len(shard_paths),
            cluster_root,
        )
    return clusters


def build_clustered_store(
    *,
    tokenize: dict[str, TokenizedAttrData],
    decontam: dict[str, DeconAttributes],
    cluster_assign: dict[str, AssignmentAttrData],
    dedup: FuzzyDupsAttrData,
    output_path: str,
    cluster_view: int = 40,
    split: str = "train",
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 4096,
) -> ClusteredStoreData:
    """Single map-side Zephyr pass: 4-way join + filter + per-cluster Levanter caches.

    See module docstring for the per-shard logic.

    Every source name must appear in ``tokenize``, ``decontam``, and
    ``cluster_assign``; every included source's
    ``TokenizedAttrData.source_main_dirs[split]`` must appear as a key in
    ``dedup.sources`` (so the caller is responsible for dropping sources
    excluded from dedup, e.g. ``safety_pt/*`` / ``climblab-ja``).

    Returns a :class:`ClusteredStoreData` describing the per-cluster caches
    and saves the artifact at ``<output_path>/artifact.json``.
    """
    if not tokenize:
        raise ValueError("build_clustered_store: tokenize is empty")
    for label, d in (("decontam", decontam), ("cluster_assign", cluster_assign)):
        if set(d) != set(tokenize):
            missing = sorted(set(tokenize) - set(d))
            extra = sorted(set(d) - set(tokenize))
            raise ValueError(f"{label} source set must equal tokenize: missing={missing!r}, extra={extra!r}")

    if worker_resources is None:
        worker_resources = ResourceConfig(cpu=2, ram="16g", disk="10g")

    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)
    logger.info(
        "build_clustered_store: %d sources, cluster_view=%d (column=%s), split=%s -> %s",
        len(tokenize),
        cluster_view,
        cluster_col,
        split,
        output_path,
    )

    shard_specs: list[dict[str, str]] = []
    for source_name in sorted(tokenize):
        tok = tokenize[source_name]
        decon = decontam[source_name]
        cluster_asg = cluster_assign[source_name]
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
                dedup_attr_dir=dedup_attr_dir,
                split=split,
            )
        )

    logger.info("build_clustered_store: %d input shards across %d sources", len(shard_specs), len(tokenize))
    if not shard_specs:
        raise ValueError("No input shards resolved -- nothing to do")

    ctx = ZephyrContext(
        resources=worker_resources,
        max_workers=min(max_workers, len(shard_specs)),
        name="datakit-clustered-store",
    )
    ds = Dataset.from_list(shard_specs).map_shard(
        lambda items, shard, cc=cluster_col, op=output_path: _join_filter_stream_shard(
            items, shard, cluster_col=cc, output_path=op
        )
    )
    outcome = ctx.execute(ds, verbose=True)

    n_non_empty_clusters = len({r.cluster_id for r in outcome.results})
    logger.info(
        "build_clustered_store: zephyr pass wrote %d shard caches across %d non-empty clusters",
        len(outcome.results),
        n_non_empty_clusters,
    )

    clusters = _merge_per_cluster_ledgers(map_shard_results=outcome.results, output_path=output_path)

    tokenizer = next(iter(tokenize.values())).tokenizer
    artifact = ClusteredStoreData(
        cache_path=output_path,
        cluster_view=cluster_view,
        split=split,
        clusters=clusters,
        source_names=sorted(tokenize),
        tokenizer=tokenizer,
        counters=dict(outcome.counters),
    )
    Artifact.save(artifact, output_path)
    return artifact
