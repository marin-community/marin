# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage report for fuzzy candidate search and full-text verification.

Headline numbers and histograms come from the artifact's aggregated counters.
The per-source table comes from bounded attribute samples.
"""

import os

from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    PERCENT_HISTOGRAM_METRICS,
    UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN,
    VerifiedFuzzyDupsAttrData,
)

from experiments.datakit.reports.common import StageReport, render_template, sample_rows, write_report

SAMPLE_LIMIT = 1000
COUNTER_PREFIX = "dedup/fuzzy/document"
VERIFICATION_COUNTER_PREFIX = "dedup/fuzzy/verification"


def _source_label(source_key: str) -> str:
    return "/".join(source_key.rstrip("/").split("/")[-3:])


def _verification_histogram(counters: dict[str, int | float]) -> list[dict[str, int | str]]:
    prefix = f"{VERIFICATION_COUNTER_PREFIX}/histogram/"
    rows: list[tuple[str, int, int]] = []
    for key, value in counters.items():
        if not key.startswith(prefix):
            continue
        metric, value_bin = key.removeprefix(prefix).split("/", maxsplit=1)
        if metric not in {*PERCENT_HISTOGRAM_METRICS, "member_unique"}:
            continue
        rows.append((metric, int(value_bin), int(value)))

    output = []
    for metric, value_bin, count in sorted(rows):
        if metric in PERCENT_HISTOGRAM_METRICS:
            label = f"{value_bin}%"
        elif value_bin == UNIQUE_NGRAM_HISTOGRAM_OVERFLOW_BIN:
            label = f"{value_bin}+"
        else:
            label = str(value_bin)
        output.append({"metric": metric, "bin": label, "count": count})
    return output


def _cluster_size_histogram(counters: dict[str, int | float]) -> list[dict[str, int | str]]:
    prefix = f"{VERIFICATION_COUNTER_PREFIX}/cluster_size/"
    rows = [
        {
            "size": key.removeprefix(prefix),
            "clusters": int(value),
        }
        for key, value in counters.items()
        if key.startswith(prefix)
    ]
    return sorted(rows, key=lambda row: int(str(row["size"]).split("-", maxsplit=1)[0]))


def _source_decisions(counters: dict[str, int | float], source_tag: str) -> dict[str, int]:
    prefix = f"{VERIFICATION_COUNTER_PREFIX}/source/{source_tag}/decision/"
    return {key.removeprefix(prefix): int(value) for key, value in counters.items() if key.startswith(prefix)}


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
    total_docs = cluster_members + singletons_skipped

    per_source = []
    for source_key, entry in candidates.sources.items():
        rows = sample_rows(entry.attr_dir, ["id", "dup_cluster_id"], SAMPLE_LIMIT)
        source_clusters = {r["dup_cluster_id"] for r in rows}
        verified_source = verified.sources[source_key]
        source_tag = os.path.basename(verified_source.attr_dir.rstrip("/"))
        source_decisions = _source_decisions(verified.counters, source_tag)
        source_accepted = source_decisions.get("accepted", 0)
        source_delegated = source_decisions.get("delegated_global_exact", 0)
        source_rejected = sum(source_decisions.values()) - source_accepted - source_delegated
        per_source.append(
            {
                "label": _source_label(source_key),
                "source_key": source_key,
                "sampled_members": len(rows),
                "sampled_clusters": len(source_clusters),
                "verified": source_accepted,
                "rejected": source_rejected,
                "delegated_global_exact": source_delegated,
            }
        )

    decisions = {
        key.removeprefix(f"{VERIFICATION_COUNTER_PREFIX}/decision/"): int(value)
        for key, value in verified.counters.items()
        if key.startswith(f"{VERIFICATION_COUNTER_PREFIX}/decision/")
    }
    delegated_global_exact = decisions.get("delegated_global_exact", 0)
    direct_comparisons = sum(decisions.values()) - delegated_global_exact
    rejected_candidates = direct_comparisons - verified_duplicates
    stats = {
        "cluster_members": cluster_members,
        "clusters": clusters,
        "candidate_duplicates": candidate_duplicates,
        "direct_comparisons": direct_comparisons,
        "delegated_global_exact": delegated_global_exact,
        "verified_duplicates": verified_duplicates,
        "rejected_candidates": rejected_candidates,
        "singletons_skipped": singletons_skipped,
        "candidate_acceptance_rate": verified_duplicates / direct_comparisons if direct_comparisons else 0.0,
        "dup_rate": verified_duplicates / total_docs if total_docs else 0.0,
        "n_sources": len(candidates.sources),
    }
    data = {
        "params": candidates.params.model_dump(),
        "verification_params": verified.verification.model_dump(),
        "decisions": [{"name": name, "count": count} for name, count in sorted(decisions.items())],
        "verification_histogram": _verification_histogram(verified.counters),
        "stats": stats,
        "sources": per_source,
        "cluster_size_hist": _cluster_size_histogram(verified.counters),
        "sample_limit": SAMPLE_LIMIT,
        "sampling": (
            "headline numbers and histograms from counters (exact); per-source table "
            f"from the first {SAMPLE_LIMIT} non-singleton rows per source (file order)"
        ),
    }
    page = render_template("dedup.html", title="Datakit dedup", data=data)
    return StageReport(html_path=write_report(output_path, page), stats=stats)
