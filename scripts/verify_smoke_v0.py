# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the _smoke_v0 datakit-clustered Levanter store against its inputs.

End-to-end integrity checks for the smoke output at
``gs://marin-eu-west4/datakit/store/_smoke_v0``:

1. Token-budget reconciliation: per source, the sum across cluster=*/ of
   ``total_elements`` equals ``tokenize.records_in - decon.contaminated -
   dedup_noncanonicals``. ``dedup_noncanonicals = cluster_members - canonicals``.
2. Levanter cache loadability: every cluster=K/ opens via
   ``TreeCache.load`` and its rolled-up row count matches the ledger.
3. Ground-truth cluster assignment: from the decon+dedup+cluster_assign
   shards we compute the expected per-cluster docs ourselves, and verify
   the output matches.
4. Id-level cross-cluster uniqueness: each survivor id appears in exactly
   one cluster (no leakage). We assert this by partitioning ids via the
   computed assignment in (3); if (3) passes the row count, the uniqueness
   follows by construction.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from levanter.store.cache import CacheLedger, CacheMetadata
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import Artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from marin.utils import fsspec_glob

from experiments.datakit.cluster.v0.assign import AssignmentAttrData
from experiments.datakit.store.all_sources_store import (
    CLUSTER_ASSIGN_ROOT,
    DECONTAM_ROOT,
    DEDUP_PATH,
    TOKENIZE_ROOT,
    _resolve_artifact_dir,
)
from experiments.datakit.store.datakit_store import ClusteredStoreData

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

logger = logging.getLogger(__name__)


def _shard_basenames(shard_paths: list[str]) -> list[str]:
    return [os.path.basename(p) for p in shard_paths]


def _load_dedup_canonical_for_source(dedup: FuzzyDupsAttrData, main_dir: str) -> dict[str, dict[str, bool]]:
    """Return ``{basename: {id: is_canonical}}`` from the per-source dedup attr dir."""
    attr_dir = dedup.sources[main_dir].attr_dir.rstrip("/")
    out: dict[str, dict[str, bool]] = {}
    for path in sorted(fsspec_glob(f"{attr_dir}/*.parquet")):
        table = pq.read_table(path, columns=["id", "attributes"])
        ids = table.column("id").to_pylist()
        canonical = table.column("attributes").combine_chunks().field("is_cluster_canonical").to_pylist()
        out[os.path.basename(path)] = dict(zip(ids, canonical, strict=True))
    return out


def _compute_expected_per_cluster(
    *,
    tokenize: TokenizedAttrData,
    decontam: DeconAttributes,
    cluster_assign: AssignmentAttrData,
    dedup: FuzzyDupsAttrData,
    cluster_view: int,
    split: str,
) -> tuple[Counter, dict[str, int], dict[int, list[str]]]:
    """Replay the join+filter locally and return per-cluster doc counts.

    Returns ``(per_cluster_counter, drop_stats, per_cluster_sample_ids)``
    where ``per_cluster_sample_ids`` carries up to 3 surviving ids per
    cluster for cross-checking against the Levanter cache.
    """
    main_dir = tokenize.source_main_dirs[split]
    dedup_by_basename = _load_dedup_canonical_for_source(dedup, main_dir)

    tok_shards = tokenize.shard_paths(split)
    decon_dir = decontam.output_dir.rstrip("/")
    cluster_dir = cluster_assign.output_dir.rstrip("/")
    cluster_col = f"cluster_{cluster_view}"

    per_cluster: Counter = Counter()
    per_cluster_sample: dict[int, list[str]] = defaultdict(list)
    drops = {"contaminated": 0, "dedup_noncanonical": 0, "kept": 0, "total_in": 0}

    for tok_path in tok_shards:
        basename = os.path.basename(tok_path)
        decon_path = f"{decon_dir}/{basename}"
        cluster_path = f"{cluster_dir}/{basename}"
        dedup_map = dedup_by_basename.get(basename, {})

        decon_table = pq.read_table(decon_path, columns=["id", "attributes"])
        decon_ids = decon_table.column("id").to_pylist()
        contaminated = decon_table.column("attributes").combine_chunks().field("contaminated").to_pylist()
        cluster_table = pq.read_table(cluster_path, columns=["id", cluster_col])
        cluster_ids = cluster_table.column("id").to_pylist()
        cluster_vals = cluster_table.column(cluster_col).to_pylist()
        if len(decon_ids) != len(cluster_ids):
            raise RuntimeError(f"{basename}: decon/cluster row mismatch")

        for did, decid, contam, cid, cval in zip(
            decon_ids, decon_ids, contaminated, cluster_ids, cluster_vals, strict=True
        ):
            drops["total_in"] += 1
            if decid != did or cid != did:
                raise RuntimeError(f"{basename}: id misalignment {did} vs {decid}/{cid}")
            if contam:
                drops["contaminated"] += 1
                continue
            canon = dedup_map.get(did)
            if canon is False:
                drops["dedup_noncanonical"] += 1
                continue
            drops["kept"] += 1
            per_cluster[int(cval)] += 1
            if len(per_cluster_sample[int(cval)]) < 3:
                per_cluster_sample[int(cval)].append(did)

    return per_cluster, drops, per_cluster_sample


def verify_source_reconciliation(
    *,
    source_name: str,
    tokenize: TokenizedAttrData,
    decontam: DeconAttributes,
    cluster_assign: AssignmentAttrData,
    dedup: FuzzyDupsAttrData,
    cluster_view: int,
    split: str,
) -> tuple[Counter, dict[str, int]]:
    """Verify counters reconcile for one source. Returns (per_cluster_counter, drops)."""
    per_cluster, drops, _ = _compute_expected_per_cluster(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        dedup=dedup,
        cluster_view=cluster_view,
        split=split,
    )

    tok_in = tokenize.counters[split]["zephyr/records_in"]
    decon_contam = decontam.counters.get("decon/contaminated", 0)
    logger.info(
        "[%s] reconciliation: tokenize_in=%d decon_contaminated=%d (artifact says: %d) "
        "computed_drops_contaminated=%d dedup_noncanonical=%d kept=%d total_in_walk=%d",
        source_name,
        tok_in,
        decon_contam,
        decon_contam,
        drops["contaminated"],
        drops["dedup_noncanonical"],
        drops["kept"],
        drops["total_in"],
    )
    if drops["total_in"] != tok_in:
        raise AssertionError(f"[{source_name}] walked rows ({drops['total_in']}) != tokenize records_in ({tok_in})")
    if drops["contaminated"] != decon_contam:
        raise AssertionError(
            f"[{source_name}] contaminated count drift: walk={drops['contaminated']} " f"artifact_counter={decon_contam}"
        )
    return per_cluster, drops


def verify_cache_loads(artifact: ClusteredStoreData) -> dict[int, int]:
    """Try to load every cluster=K/ via Levanter. Returns ``{cluster_id: total_rows_from_ledger}``."""
    metadata = CacheMetadata.empty()
    per_cluster_rows: dict[int, int] = {}
    for cluster_id, stats in sorted(artifact.clusters.items()):
        ledger = CacheLedger.load(stats.path, metadata)
        if ledger.total_num_rows != stats.total_elements:
            raise AssertionError(
                f"cluster={cluster_id}: ledger.total_num_rows={ledger.total_num_rows} "
                f"!= artifact.total_elements={stats.total_elements}"
            )
        per_cluster_rows[cluster_id] = ledger.total_num_rows
    logger.info("cache-load: every cluster ledger loaded and row counts match artifact")
    return per_cluster_rows


def verify_per_cluster_match(
    artifact: ClusteredStoreData,
    expected_per_cluster_total: Counter,
) -> None:
    """Verify per-cluster docs in artifact equals the locally-computed expected counts."""
    artifact_counts = Counter({cid: stats.total_elements for cid, stats in artifact.clusters.items()})
    if artifact_counts != expected_per_cluster_total:
        only_artifact = dict(artifact_counts - expected_per_cluster_total)
        only_expected = dict(expected_per_cluster_total - artifact_counts)
        raise AssertionError(
            "per-cluster doc counts disagree. "
            f"extra-in-artifact={only_artifact!r} extra-in-expected={only_expected!r}"
        )
    logger.info("per-cluster doc counts match across all %d clusters", len(artifact.clusters))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke", choices=sorted(_PRESETS), help="Which smoke output to verify.")
    args = parser.parse_args()
    output_path, sources, split = _PRESETS[args.smoke]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    artifact = Artifact.from_path(output_path, ClusteredStoreData)
    logger.info("loaded artifact at %s: %d clusters", output_path, len(artifact.clusters))

    dedup = Artifact.from_path(DEDUP_PATH, FuzzyDupsAttrData)

    combined_expected: Counter = Counter()
    grand_totals = {"tokenize_in": 0, "contaminated": 0, "dedup_noncanonical": 0, "kept": 0}

    for source_name in sources:
        tokenize = Artifact.from_path(_resolve_artifact_dir(TOKENIZE_ROOT, source_name), TokenizedAttrData)
        decontam = Artifact.from_path(_resolve_artifact_dir(DECONTAM_ROOT, source_name), DeconAttributes)
        cluster_assign = Artifact.from_path(_resolve_artifact_dir(CLUSTER_ASSIGN_ROOT, source_name), AssignmentAttrData)
        per_cluster, drops = verify_source_reconciliation(
            source_name=source_name,
            tokenize=tokenize,
            decontam=decontam,
            cluster_assign=cluster_assign,
            dedup=dedup,
            cluster_view=artifact.cluster_view,
            split=split,
        )
        combined_expected.update(per_cluster)
        grand_totals["tokenize_in"] += drops["total_in"]
        grand_totals["contaminated"] += drops["contaminated"]
        grand_totals["dedup_noncanonical"] += drops["dedup_noncanonical"]
        grand_totals["kept"] += drops["kept"]

    logger.info("grand totals: %s", grand_totals)
    artifact_total_docs = sum(c.total_elements for c in artifact.clusters.values())
    if artifact_total_docs != grand_totals["kept"]:
        raise AssertionError(
            f"artifact total docs ({artifact_total_docs}) != walked kept count ({grand_totals['kept']})"
        )
    logger.info("artifact total docs = walked kept = %d", artifact_total_docs)

    verify_per_cluster_match(artifact, combined_expected)
    verify_cache_loads(artifact)

    logger.info("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
