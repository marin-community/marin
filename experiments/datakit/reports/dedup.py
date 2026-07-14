# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for the cross-source fuzzy dedup step.

Headline numbers come from the artifact's aggregated counters. The per-source
table and the cluster-size histogram come from a bounded sample of each
source's dup-marker parquet, which is sparse: only non-singleton cluster
members get a row, and sources with zero non-singletons have empty shards.
"""

from collections import Counter

from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData

from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

SAMPLE_LIMIT = 1000
COUNTER_PREFIX = "dedup/fuzzy/document"


def _source_label(source_main_dir: str) -> str:
    return "/".join(source_main_dir.rstrip("/").split("/")[-3:])


def dedup_report(output_path: str, dedup: FuzzyDupsAttrData) -> StageReport:
    """Render the fuzzy-dedup stage report and return its path plus headline stats."""
    cluster_members = int(dedup.counters.get(f"{COUNTER_PREFIX}/cluster_members", 0))
    clusters = int(dedup.counters.get(f"{COUNTER_PREFIX}/canonicals", 0))
    singletons_skipped = int(dedup.counters.get(f"{COUNTER_PREFIX}/singletons_skipped", 0))
    duplicates_to_drop = cluster_members - clusters
    total_docs = cluster_members + singletons_skipped

    # dup_cluster_id is global across sources, so pooling the per-source
    # samples yields cross-source cluster sizes (within the sample).
    sampled_cluster_sizes: Counter[str] = Counter()
    per_source = []
    for source_main_dir, entry in dedup.sources.items():
        rows = sample_rows(entry.attr_dir, ["id", "attributes"], SAMPLE_LIMIT)
        source_clusters = {r["attributes"]["dup_cluster_id"] for r in rows}
        sampled_cluster_sizes.update(r["attributes"]["dup_cluster_id"] for r in rows)
        per_source.append(
            {
                "label": _source_label(source_main_dir),
                "source_main_dir": source_main_dir,
                "sampled_members": len(rows),
                "sampled_clusters": len(source_clusters),
            }
        )

    stats = {
        "cluster_members": cluster_members,
        "clusters": clusters,
        "duplicates_to_drop": duplicates_to_drop,
        "singletons_skipped": singletons_skipped,
        "dup_rate": duplicates_to_drop / total_docs if total_docs else 0.0,
        "n_sources": len(dedup.sources),
    }
    data = {
        "params": dedup.params.model_dump(),
        "stats": stats,
        "sources": per_source,
        "cluster_size_hist": [
            {"size": size, "clusters": count} for size, count in sorted(Counter(sampled_cluster_sizes.values()).items())
        ],
        "sample_limit": SAMPLE_LIMIT,
        "sampling": (
            f"headline numbers from dedup counters (exact); per-source table + cluster-size histogram "
            f"from the first {SAMPLE_LIMIT} non-singleton rows per source (file order)"
        ),
    }
    page = render_template("dedup.html", title="Datakit dedup", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
