# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a datakit-clustered Levanter smoke output against its inputs (on Iris).

Replays the join + filter logic of ``build_clustered_store`` shard-by-shard
in a Zephyr map_shard pass: each worker loads the three small attribute
tables for one input shard (``id+contaminated`` from decon, ``id+cluster_K``
from cluster_assign, ``id+is_cluster_canonical`` from dedup, NOT the
tokenize ``input_ids``), counts survivors per ``cluster_<view>``, and
yields per-source totals + per-cluster kept counts.

The driver then asserts:

* Aggregate walked totals (across all sources) match the store's
  ``artifact.counters`` for ``records_in``, ``contaminated_dropped``,
  ``dedup_noncanonical_dropped``, and ``records_out``.
* Per-cluster kept counts equal the artifact's ``ClusterCacheStats.total_elements``.
* Every cluster's ``CacheLedger`` loads cleanly and ``total_num_rows`` matches.

Note: per-source ``TokenizedAttrData.counters`` / ``DeconAttributes.counters``
are NOT used as ground truth -- those track records processed by *this*
run of tokenize/decon, but ``skip_existing=True`` preserves pre-existing
output parquets from earlier (resumed) runs, so the counter underestimates
the row count in the output directory. The walked totals (counting rows
in the *actual output parquets the pipeline reads*) are authoritative.

Submit on iris (eu-west4)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive --cpu 2 --memory 8GB --enable-extra-resources \\
        -- python scripts/verify_smoke_v0.py {v0,mixed}
"""

import argparse
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator

import pyarrow.compute as pc
from fray.types import ResourceConfig
from levanter.store.cache import CacheLedger, CacheMetadata
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import LlmQualityOutput
from experiments.datakit.store.all_sources_store import (
    CLUSTER_ASSIGN_ROOT,
    DECONTAM_ROOT,
    DEDUP_PATH,
    QUALITY_ROOT,
    TOKENIZE_ROOT,
    _resolve_artifact_dir,
)
from experiments.datakit.store.datakit_store import (
    ClusteredStoreData,
    _load_cluster_table,
    _load_decon_table,
    _load_dedup_canonical,
    _load_quality_table,
    _per_source_shard_tuples,
    _quality_bucket,
)

logger = logging.getLogger(__name__)


# Named smoke configurations. Each entry: (output_path, source_names, split).
_PRESETS: dict[str, tuple[str, tuple[str, ...], str]] = {}
try:
    from experiments.datakit.store.smoke_test import OUTPUT_PATH as _V0_OUT
    from experiments.datakit.store.smoke_test import SMOKE_SOURCES as _V0_SRC
    from experiments.datakit.store.smoke_test import SPLIT as _V0_SPLIT

    _PRESETS["v0"] = (_V0_OUT, _V0_SRC, _V0_SPLIT)
except ImportError:
    pass
try:
    from experiments.datakit.store.smoke_test_mixed import OUTPUT_PATH as _MIX_OUT
    from experiments.datakit.store.smoke_test_mixed import SMOKE_SOURCES as _MIX_SRC
    from experiments.datakit.store.smoke_test_mixed import SPLIT as _MIX_SPLIT

    _PRESETS["mixed"] = (_MIX_OUT, _MIX_SRC, _MIX_SPLIT)
except ImportError:
    pass


def _verify_shard(
    items: Iterator[dict[str, str]],
    shard_info: ShardInfo,
    *,
    cluster_col: str,
) -> Iterator[dict]:
    """Per-shard replay: load 4 attribute tables, count survivors per (cluster, quality) bucket.

    Doesn't touch tokenize ``input_ids`` -- a 10-100x I/O savings vs. the
    pipeline itself. Same join + filter + bucketing logic as
    ``_join_filter_stream_shard``, but only counts (no writes).
    """
    spec = next(iter(items))
    source_name = spec["source_name"]
    basename = spec["basename"]
    decon_ids, contaminated = _load_decon_table(spec["decontam"])
    cluster_ids, cluster_vals = _load_cluster_table(spec["cluster"], cluster_col)
    quality_ids, scores = _load_quality_table(spec["quality"])
    n_decon, n_cluster, n_quality = len(decon_ids), len(cluster_ids), len(quality_ids)
    if not (n_decon == n_cluster == n_quality):
        raise RuntimeError(
            f"{source_name}/{basename}: dense-table row count mismatch "
            f"(decon={n_decon}, cluster={n_cluster}, quality={n_quality}) -- co-partitioning broken"
        )
    if not pc.all(pc.equal(decon_ids, cluster_ids)).as_py():
        raise RuntimeError(f"{source_name}/{basename}: decon/cluster id mismatch -- co-partitioning broken")
    if not pc.all(pc.equal(decon_ids, quality_ids)).as_py():
        raise RuntimeError(f"{source_name}/{basename}: decon/quality id mismatch -- co-partitioning broken")
    # ids only needed to confirm alignment with decon's id list against the
    # dedup hash. Materialize once as a Python list (much smaller than
    # carrying 3 separate id list copies through the loop) and drop the rest.
    decon_id_list = decon_ids.to_pylist()
    del decon_ids, cluster_ids, quality_ids
    dedup_canonical = _load_dedup_canonical(spec["dedup"])

    # per_bucket key is "c=<cluster_id>,q=<quality_bucket>" -- str so the dict
    # is JSON/zephyr-friendly when traveling back to the driver.
    per_bucket: dict[str, int] = defaultdict(int)
    n_contam = 0
    n_dedup_drop = 0
    for did, contam, cval, qscore in zip(decon_id_list, contaminated, cluster_vals, scores, strict=True):
        if contam:
            n_contam += 1
            continue
        if dedup_canonical.get(did) is False:
            n_dedup_drop += 1
            continue
        per_bucket[f"c={int(cval)},q={_quality_bucket(qscore)}"] += 1
    yield {
        "source_name": source_name,
        "n_in": n_decon,
        "contaminated": n_contam,
        "dedup_noncanonical": n_dedup_drop,
        "per_bucket": dict(per_bucket),
    }


def _build_shard_specs(sources: tuple[str, ...], split: str, dedup: FuzzyDupsAttrData) -> list[dict[str, str]]:
    """Resolve per-source artifacts and project to per-shard spec dicts."""
    specs: list[dict[str, str]] = []
    for src in sources:
        tok = read_artifact(_resolve_artifact_dir(TOKENIZE_ROOT, src), TokenizedAttrData)
        decon = read_artifact(_resolve_artifact_dir(DECONTAM_ROOT, src), DeconAttributes)
        cluster_a = read_artifact(_resolve_artifact_dir(CLUSTER_ASSIGN_ROOT, src), AssignmentAttrData)
        qual = read_artifact(_resolve_artifact_dir(QUALITY_ROOT, src), LlmQualityOutput)
        main_dir = tok.source_main_dirs[split]
        dedup_attr_dir = dedup.sources[main_dir].attr_dir
        specs.extend(
            _per_source_shard_tuples(
                source_name=src,
                tokenize=tok,
                decontam=decon,
                cluster_assign=cluster_a,
                quality=qual,
                dedup_attr_dir=dedup_attr_dir,
                split=split,
            )
        )
    return specs


def _verify_cache_loads(artifact: ClusteredStoreData) -> None:
    """Every ``cluster=K/quality=Q/`` ledger opens cleanly and matches the artifact's ``total_elements``."""
    metadata = CacheMetadata.empty()
    for stats in sorted(artifact.buckets, key=lambda b: (b.cluster_id, b.quality_bucket)):
        ledger = CacheLedger.load(stats.path, metadata)
        if ledger.total_num_rows != stats.total_elements:
            raise AssertionError(
                f"cluster={stats.cluster_id} quality={stats.quality_bucket}: "
                f"ledger.total_num_rows={ledger.total_num_rows} "
                f"!= artifact.total_elements={stats.total_elements}"
            )
    logger.info("cache-load: every bucket ledger loaded and row counts match artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke", choices=sorted(_PRESETS), help="Which smoke output to verify.")
    args = parser.parse_args()
    output_path, sources, split = _PRESETS[args.smoke]

    configure_logging(logging.INFO)
    artifact = read_artifact(output_path, ClusteredStoreData)
    logger.info("loaded artifact at %s: %d buckets", output_path, len(artifact.buckets))

    dedup = read_artifact(DEDUP_PATH, FuzzyDupsAttrData)
    cluster_col = f"cluster_{artifact.cluster_view}"

    shard_specs = _build_shard_specs(sources, split, dedup)
    logger.info("verify: %d input shards across %d sources", len(shard_specs), len(sources))

    ctx = ZephyrContext(
        resources=ResourceConfig(cpu=1, ram="2g"),
        max_workers=min(1024, len(shard_specs)),
        name="verify-smoke",
    )
    ds = Dataset.from_list(shard_specs).map_shard(
        lambda items, shard, cc=cluster_col: _verify_shard(items, shard, cluster_col=cc)
    )
    outcome = ctx.execute(ds, verbose=True)

    # Aggregate per-source and per-(cluster, quality) bucket.
    per_source_in: Counter = Counter()
    per_source_contam: Counter = Counter()
    per_source_dedup: Counter = Counter()
    per_bucket_kept: Counter = Counter()
    for r in outcome.results:
        per_source_in[r["source_name"]] += r["n_in"]
        per_source_contam[r["source_name"]] += r["contaminated"]
        per_source_dedup[r["source_name"]] += r["dedup_noncanonical"]
        for bucket_key, n in r["per_bucket"].items():
            per_bucket_kept[bucket_key] += n

    total_in = sum(per_source_in.values())
    total_contam = sum(per_source_contam.values())
    total_dedup = sum(per_source_dedup.values())
    total_kept = sum(per_bucket_kept.values())
    logger.info(
        "totals: in=%d contaminated=%d dedup_noncanonical=%d kept=%d",
        total_in,
        total_contam,
        total_dedup,
        total_kept,
    )

    # 1. Per-source totals -- logged for visibility. We don't compare against
    # each source's own TokenizedAttrData / DeconAttributes counters because
    # those track records processed *by this run of tokenize/decon*; with
    # skip_existing=True, output parquets from earlier (resumed) runs persist
    # and contribute rows to what we read here. The walked count is the row
    # count in the actual output dirs the pipeline reads -- which is what we
    # want to verify against the store's aggregate counters.
    for src in sources:
        logger.info(
            "[%s] walked: in=%d contaminated=%d dedup_noncanonical=%d",
            src,
            per_source_in[src],
            per_source_contam[src],
            per_source_dedup[src],
        )

    # 2. Aggregate counters match the store's own artifact.counters.
    c = artifact.counters
    expected = {
        "datakit_store/records_in": total_in,
        "datakit_store/contaminated_dropped": total_contam,
        "datakit_store/dedup_noncanonical_dropped": total_dedup,
        "datakit_store/records_out": total_kept,
    }
    for key, walked in expected.items():
        if c.get(key, -1) != walked:
            raise AssertionError(f"counter {key}: artifact={c.get(key)!r} walked={walked!r}")
    logger.info("store counter reconciliation matches walked totals across all 4 buckets")

    # 3. Per-(cluster, quality) kept matches artifact's per-bucket total_elements.
    artifact_counts: Counter = Counter(
        {f"c={b.cluster_id},q={b.quality_bucket}": b.total_elements for b in artifact.buckets}
    )
    if artifact_counts != per_bucket_kept:
        only_artifact = dict(artifact_counts - per_bucket_kept)
        only_walked = dict(per_bucket_kept - artifact_counts)
        raise AssertionError(
            f"per-bucket counts disagree. extra-in-artifact={only_artifact!r} extra-in-walked={only_walked!r}"
        )
    logger.info("per-(cluster, quality) kept counts match across all %d buckets", len(artifact.buckets))

    # 4. Levanter cache loadability.
    _verify_cache_loads(artifact)

    logger.info("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
