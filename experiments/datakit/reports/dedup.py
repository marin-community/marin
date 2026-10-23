# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for fuzzy-duplicate cluster verification.

Headline numbers and the per-source table come from the artifact's aggregated counters.
"""

from marin.processing.classification.deduplication.cluster_verify import (
    COUNTER_PREFIX as CLUSTER_VERIFY_COUNTER_PREFIX,
)
from marin.processing.classification.deduplication.cluster_verify import (
    ClusterVerifiedFuzzyDupsAttrData,
)
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData

from experiments.datakit.reports.common import StageReport, render_template, write_report


def cluster_dedup_report(
    output_path: str,
    candidates: FuzzyDupsAttrData,
    verified: ClusterVerifiedFuzzyDupsAttrData,
) -> StageReport:
    """Report measured cluster-verification counts and its recorded rule."""
    prefix = CLUSTER_VERIFY_COUNTER_PREFIX
    members = int(verified.counters[f"{prefix}/documents"])
    removed = int(verified.counters[f"{prefix}/markers"])
    stats = {
        "cluster_members": members,
        "verified_duplicates": removed,
        "retained_members": members - removed,
        "n_sources": len(verified.sources),
        "mid_cluster_flushes": int(verified.counters.get(f"{prefix}/mid_cluster_flushes", 0)),
        "truncated_documents": int(verified.counters.get(f"{prefix}/truncated_documents", 0)),
    }
    data = {
        "params": candidates.params.model_dump(mode="json"),
        "rule": verified.rule.model_dump(mode="json"),
        "limits": verified.limits.model_dump(mode="json"),
        "stats": stats,
        "sources": [
            {"source": key, "removed": int(verified.counters.get(f"{prefix}/source/{entry.source_tag}/markers", 0))}
            for key, entry in sorted(verified.sources.items())
        ],
    }
    page = render_template("cluster_dedup.html", title="Datakit cluster verification", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
