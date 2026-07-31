# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for fuzzy candidate search and full-text verification.

Headline numbers come from the artifact's aggregated counters. The per-source
table and the cluster-size histogram come from bounded attribute samples.
"""

from collections import Counter

from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.verify_fuzzy_dups import VerifiedFuzzyDupsAttrData

from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

SAMPLE_LIMIT = 1000
COUNTER_PREFIX = "dedup/fuzzy/document"
VERIFICATION_COUNTER_PREFIX = "dedup/fuzzy/verification"


def _source_label(source_key: str) -> str:
    return "/".join(source_key.rstrip("/").split("/")[-3:])


def _score_histogram(counters: dict[str, int | float]) -> list[dict[str, int | str]]:
    prefix = f"{VERIFICATION_COUNTER_PREFIX}/histogram/"
    rows = []
    for key, value in counters.items():
        if not key.startswith(prefix):
            continue
        metric, score_bin = key.removeprefix(prefix).split("/", maxsplit=1)
        if metric not in {"member_containment", "jaccard", "char_jaccard"}:
            continue
        rows.append({"metric": metric, "score_percent": int(score_bin), "count": int(value)})
    return sorted(rows, key=lambda row: (str(row["metric"]), int(row["score_percent"])))


def dedup_report(
    output_path: str,
    candidates: FuzzyDupsAttrData,
    verified: VerifiedFuzzyDupsAttrData,
) -> StageReport:
    """Render the fuzzy-dedup stage report and return its path plus headline stats."""
    cluster_members = int(candidates.counters.get(f"{COUNTER_PREFIX}/cluster_members", 0))
    clusters = int(candidates.counters.get(f"{COUNTER_PREFIX}/canonicals", 0))
    singletons_skipped = int(candidates.counters.get(f"{COUNTER_PREFIX}/singletons_skipped", 0))
    candidate_duplicates = cluster_members - clusters
    verified_duplicates = int(verified.counters.get(f"{VERIFICATION_COUNTER_PREFIX}/verified_duplicates", 0))
    rejected_candidates = candidate_duplicates - verified_duplicates
    total_docs = cluster_members + singletons_skipped

    # dup_cluster_id is global across sources, so pooling the per-source
    # samples yields cross-source cluster sizes (within the sample).
    sampled_cluster_sizes: Counter[str] = Counter()
    per_source = []
    for source_key, entry in candidates.sources.items():
        rows = sample_rows(entry.attr_dir, ["id", "dup_cluster_id"], SAMPLE_LIMIT)
        verified_rows = sample_rows(verified.sources[source_key].attr_dir, ["id"], SAMPLE_LIMIT)
        source_clusters = {r["dup_cluster_id"] for r in rows}
        sampled_cluster_sizes.update(r["dup_cluster_id"] for r in rows)
        per_source.append(
            {
                "label": _source_label(source_key),
                "source_key": source_key,
                "sampled_members": len(rows),
                "sampled_clusters": len(source_clusters),
                "sampled_verified": len(verified_rows),
            }
        )

    decisions = {
        key.removeprefix(f"{VERIFICATION_COUNTER_PREFIX}/decision/"): int(value)
        for key, value in verified.counters.items()
        if key.startswith(f"{VERIFICATION_COUNTER_PREFIX}/decision/")
    }
    stats = {
        "cluster_members": cluster_members,
        "clusters": clusters,
        "candidate_duplicates": candidate_duplicates,
        "verified_duplicates": verified_duplicates,
        "rejected_candidates": rejected_candidates,
        "singletons_skipped": singletons_skipped,
        "candidate_acceptance_rate": verified_duplicates / candidate_duplicates if candidate_duplicates else 0.0,
        "dup_rate": verified_duplicates / total_docs if total_docs else 0.0,
        "n_sources": len(candidates.sources),
    }
    data = {
        "params": candidates.params.model_dump(),
        "verification_params": verified.verification.model_dump(),
        "decisions": [{"name": name, "count": count} for name, count in sorted(decisions.items())],
        "score_histogram": _score_histogram(verified.counters),
        "stats": stats,
        "sources": per_source,
        "cluster_size_hist": [
            {"size": size, "clusters": count} for size, count in sorted(Counter(sampled_cluster_sizes.values()).items())
        ],
        "sample_limit": SAMPLE_LIMIT,
        "sampling": (
            "headline numbers from candidate and verification counters (exact); "
            "per-source table + cluster-size histogram "
            f"from the first {SAMPLE_LIMIT} non-singleton rows per source (file order)"
        ),
    }
    page = render_template("dedup.html", title="Datakit dedup", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
