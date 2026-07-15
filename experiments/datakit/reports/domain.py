# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for domain cluster assignment.

Doc counts come from the ``assign/docs_in`` counters; cluster occupancy,
assignment-distance percentiles, and cluster x source concentration come from
a bounded per-source sample (two small columns, ``SAMPLE_LIMIT`` rows) of the
assignment parquet.
"""

import os
from collections import Counter

import numpy as np

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

SAMPLE_LIMIT = 2000
TOP_CLUSTERS = 12
TOP_CLUSTERS_PER_SOURCE = 3


def assign_report(output_path: str, sources: dict[str, AssignmentAttrData], cluster_view: int) -> StageReport:
    """Aggregate per-source assignment artifacts into one HTML stage report.

    Args:
        output_path: Directory the rendered ``report.html`` is written to.
        sources: Source name -> that source's :class:`AssignmentAttrData`.
        cluster_view: Which ``cluster_<k>`` view to histogram occupancy over.
    """
    first = next(iter(sources.values()))
    k_train, k_views = first.k_train, first.k_views

    occupancy: Counter[int] = Counter()
    cluster_sources: dict[int, Counter[str]] = {}
    source_rows = []
    for name, art in sorted(sources.items()):
        rows = sample_rows(art.output_dir, [f"cluster_{cluster_view}", f"dist_{art.k_train}"], SAMPLE_LIMIT)
        clusters = Counter(r[f"cluster_{cluster_view}"] for r in rows)
        occupancy.update(clusters)
        for cid, n in clusters.items():
            cluster_sources.setdefault(cid, Counter())[name] = n
        dists = [r[f"dist_{art.k_train}"] for r in rows]
        source_rows.append(
            {
                "name": name,
                "docs": art.counters["assign/docs_in"],
                "sampled": len(rows),
                "top_clusters": clusters.most_common(TOP_CLUSTERS_PER_SOURCE),
                "dist_p50": float(np.percentile(dists, 50)),
                "dist_p90": float(np.percentile(dists, 90)),
            }
        )

    concentration = [
        {"cluster": cid, "sampled": n, "sources": cluster_sources[cid].most_common()}
        for cid, n in occupancy.most_common(TOP_CLUSTERS)
    ]

    stats = {
        "docs_assigned": sum(a.counters["assign/docs_in"] for a in sources.values()),
        "n_sources": len(sources),
        "k_train": k_train,
        "n_views": len(k_views),
        "cluster_view": cluster_view,
        "clusters_seen": len(occupancy),
        "sampled_rows": sum(s["sampled"] for s in source_rows),
    }
    data_root = os.path.commonprefix([a.source_main_dir for a in sources.values()]).rsplit("/", 1)[0]
    data = {
        "meta": {
            "k_train": k_train,
            "k_views": k_views,
            "cluster_view": cluster_view,
            "sample_limit": SAMPLE_LIMIT,
            "data_root": data_root,
            "sampling": (
                f"cluster occupancy, concentration and distance percentiles from the first {SAMPLE_LIMIT} "
                f"rows per source (file order); docs-assigned from step counters (exact)"
            ),
        },
        "stats": stats,
        "occupancy": sorted(occupancy.items()),
        "sources": source_rows,
        "concentration": concentration,
    }
    page = render_template("domain.html", title="Datakit domain clusters", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
