# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print a compact summary of the saved Arctic Embed Large report."""

import json
from statistics import median
from typing import Any

import fsspec
from evaluate_ladder import EVALUATION_ROOT
from evaluate_teacher_large import OUTPUT_NAME

REPORT_URL = f"{EVALUATION_ROOT}/{OUTPUT_NAME}/report.json"
LOG_CHUNK_CHARACTERS = 2_000


def probe_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    probe = metrics["probe"]
    return {
        "macro_f1": probe["macro_f1"],
        "worst_source_recall": probe["worst_source_recall"],
        "category_macro_f1": probe["category_macro_f1"],
    }


def comparison_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_f1_delta": comparison["macro_f1_delta"],
        "worst_source_recall_delta": comparison["worst_source_recall_delta"],
        "category_macro_f1_delta": comparison["category_macro_f1_delta"],
        "probe_uncertainty": comparison["probe_uncertainty"],
        "cluster_distribution_delta": comparison["cluster_distribution_delta"],
    }


def code_cluster_summary(metrics: dict[str, Any]) -> dict[str, float]:
    by_seed = metrics["collapse"]["cluster_distribution"]["by_seed"]
    fields = ("largest_cluster_share", "effective_cluster_count", "source_cluster_nmi")
    return {field: median(seed["categories"]["code"][field] for seed in by_seed) for field in fields}


def main() -> None:
    filesystem, path = fsspec.core.url_to_fs(REPORT_URL)
    with filesystem.open(path) as file:
        report = json.load(file)
    comparison = report["large_vs_luxical"]
    regular_sources = {
        source: metrics
        for source, metrics in comparison["collapse"]["source_results"].items()
        if not metrics["ood_exception"]
    }
    large_collapse = report["arctic_large"]["collapse"]
    summary = {
        "report_url": REPORT_URL,
        "html_url": f"{EVALUATION_ROOT}/{OUTPUT_NAME}/report.html",
        "evaluation_rows": report["evaluation_rows"],
        "teacher": report["teacher"],
        "embedding_run": report["embedding_run"],
        "all_required_gates_passed": comparison["all_required_gates_passed"],
        "gates": comparison["gates"],
        "probes": {name: probe_summary(report[name]) for name in ("luxical_one", "arctic_medium", "arctic_large")},
        "large_vs_luxical": comparison_summary(comparison),
        "large_vs_medium": comparison_summary(report["large_vs_medium"]),
        "collapse": {
            "finite_fraction": large_collapse["finite_fraction"],
            "exact_unique_fraction": large_collapse["exact_unique_fraction"],
            "unique_fraction_4dp": large_collapse["unique_fraction_4dp"],
            "failure_summary": report["failure_summary"],
            "regular_failure_count": len(comparison["collapse"]["regular_failures"]),
            "ood_failures": comparison["collapse"]["ood_failures"],
            "minimum_effective_rank_ratio": min(metrics["effective_rank_ratio"] for metrics in regular_sources.values()),
            "minimum_variance_ratio": min(metrics["variance_ratio"] for metrics in regular_sources.values()),
            "maximum_cluster_source": max(
                regular_sources,
                key=lambda source: regular_sources[source]["largest_cluster_share"],
            ),
            "maximum_cluster_share": max(metrics["largest_cluster_share"] for metrics in regular_sources.values()),
            "code_cluster_distribution": {
                name: code_cluster_summary(report[name]) for name in ("luxical_one", "arctic_medium", "arctic_large")
            },
        },
    }
    serialized = json.dumps(summary, sort_keys=True)
    for index, start in enumerate(range(0, len(serialized), LOG_CHUNK_CHARACTERS)):
        chunk = serialized[start : start + LOG_CHUNK_CHARACTERS]
        print(f"LUXICAL_ARCTIC_LARGE_REPORT_CHUNK={index:04d}:{chunk}")


if __name__ == "__main__":
    main()
