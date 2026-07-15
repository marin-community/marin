# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for the final per-(cluster, quality) store.

Artifact-only: :class:`ClusteredStoreData` already carries every bucket's doc
and token counts, so the page needs no parquet reads. Shows the corpus-level
funnel (records in vs contaminated / dedup-dropped from the join counters),
the cluster x quality token heatmap, and per-quality / per-cluster totals.
"""

from experiments.datakit.reports.common import StageReport, render_template, write_report
from experiments.datakit.store.datakit_store import ClusteredStoreData


def store_report(output_path: str, store: ClusteredStoreData) -> StageReport:
    n_quality = len(store.bucket_edges) + 1
    clusters = sorted({b.cluster_id for b in store.buckets})
    heat = {(b.cluster_id, b.quality_bucket): b for b in store.buckets}
    grid = [
        {
            "cluster": c,
            "tokens": [heat[(c, q)].total_tokens if (c, q) in heat else 0 for q in range(n_quality)],
            "docs": [heat[(c, q)].total_elements if (c, q) in heat else 0 for q in range(n_quality)],
        }
        for c in clusters
    ]

    stats = {
        "total_tokens": sum(b.total_tokens for b in store.buckets),
        "total_docs": sum(b.total_elements for b in store.buckets),
        "n_buckets": len(store.buckets),
        "n_clusters": len(clusters),
        "n_quality_buckets": n_quality,
        "n_sources": len(store.source_names),
        "records_in": store.counters.get("datakit_store/records_in", 0),
        "contaminated_dropped": store.counters.get("datakit_store/contaminated_dropped", 0),
        "dedup_noncanonical_dropped": store.counters.get("datakit_store/dedup_noncanonical_dropped", 0),
        "records_out": store.counters.get("datakit_store/records_out", 0),
    }
    data = {
        "meta": {
            "cache_path": store.cache_path,
            "cluster_view": store.cluster_view,
            "bucket_edges": store.bucket_edges,
            "split": store.split,
            "tokenizer": store.tokenizer,
            "sources": store.source_names,
            "sampling": "exact — aggregated from the per-bucket cache ledgers; no sampling",
        },
        "stats": stats,
        "grid": grid,
    }
    page = render_template("store.html", title="Datakit store", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
