# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate stored Arctic vectors directly against Luxical-One."""

import html
import json
import logging
import os
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
from evaluate_ladder import (
    BASELINE_FILE,
    BASELINE_REPO,
    BASELINE_REVISION,
    BOOTSTRAP_SAMPLES,
    CLUSTER_COUNT,
    CLUSTER_MAX_SOURCE_SHARE,
    CLUSTER_SEEDS,
    CPU_THREADS,
    EVALUATION_ROOT,
    MANIFEST_URL,
    MIN_EFFECTIVE_RANK_RATIO,
    MIN_UNIQUE_FRACTION,
    MIN_VARIANCE_RATIO,
    QUALITY_DELTA,
    TEACHER_EMBEDDING_DIMENSION,
    TEACHER_QUANTIZATION_LIMIT,
    fixed_evaluation_data,
    model_metrics,
    pair_indices,
    read_json,
    teacher_comparison_report,
    vector_metrics,
)
from huggingface_hub import hf_hub_download
from ladder_config import PREDECLARED_OOD_SOURCES, SourceCategory
from luxical.embedder import Embedder
from rigging.filesystem import atomic_rename
from threadpoolctl import threadpool_limits

OUTPUT_NAME = "teacher-arctic-v1"
RESULT_FILE = Path("/tmp/luxical-arctic-teacher-evaluation")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def source_categories(labels: np.ndarray, categories: np.ndarray) -> dict[str, str]:
    """Return one category for each source."""
    result = {}
    for source, category in zip(labels, categories, strict=True):
        source_name = str(source)
        category_name = str(category)
        previous = result.setdefault(source_name, category_name)
        if previous != category_name:
            raise ValueError(f"Source {source_name} has categories {previous} and {category_name}")
    return result


def failure_reasons(metrics: dict[str, Any]) -> list[str]:
    """Return the failed composite-collapse checks for one source."""
    reasons = []
    if metrics["largest_cluster_share"] > CLUSTER_MAX_SOURCE_SHARE:
        reasons.append("cluster")
    if metrics["unique_fraction_4dp"] < MIN_UNIQUE_FRACTION:
        reasons.append("unique")
    if metrics["effective_rank_ratio"] < MIN_EFFECTIVE_RANK_RATIO:
        reasons.append("rank")
    if metrics["variance_ratio"] < MIN_VARIANCE_RATIO:
        reasons.append("variance")
    return reasons


def add_source_details(comparison: dict[str, Any], categories_by_source: dict[str, str]) -> None:
    """Add source categories and explicit collapse failure reasons."""
    for source, metrics in comparison["collapse"]["source_results"].items():
        metrics["source_category"] = categories_by_source[source]
        metrics["failure_reasons"] = failure_reasons(metrics)


def failure_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    """Summarize regular-source collapse failures by category and reason."""
    category_results = {
        category.value: {"source_count": 0, "failure_count": 0, "failed_sources": []}
        for category in SourceCategory
        if category != SourceCategory.OOD
    }
    reason_counts = {"cluster": 0, "unique": 0, "rank": 0, "variance": 0}
    for source, metrics in comparison["collapse"]["source_results"].items():
        if metrics["ood_exception"]:
            continue
        category_result = category_results[metrics["source_category"]]
        category_result["source_count"] += 1
        if metrics["passed"]:
            continue
        category_result["failure_count"] += 1
        category_result["failed_sources"].append(source)
        for reason in metrics["failure_reasons"]:
            reason_counts[reason] += 1
    return {"by_category": category_results, "by_reason": reason_counts}


def html_report(report: dict[str, Any]) -> str:
    """Render a standalone direct-teacher report."""
    comparison = report["comparison"]
    gate_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in comparison["gates"].items()
    )
    category_rows = "".join(
        f"<tr><td>{html.escape(category)}</td><td>{delta:.4f}</td></tr>"
        for category, delta in comparison["category_macro_f1_delta"].items()
    )
    collapse_rows = "".join(
        f"<tr><td>{html.escape(category)}</td><td>{values['failure_count']}</td>"
        f"<td>{values['source_count']}</td></tr>"
        for category, values in report["failure_summary"]["by_category"].items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Direct Arctic evaluation</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 90rem; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #bbb; padding: .35rem .6rem; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f5f5; padding: 1rem; }}
</style></head><body>
<h1>Direct Arctic evaluation</h1>
<p>All required gates: {'PASS' if comparison['all_required_gates_passed'] else 'FAIL'}</p>
<table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table>
<p>Macro-F1 delta: {comparison['macro_f1_delta']:.4f}</p>
<p>Worst-source recall delta: {comparison['worst_source_recall_delta']:.4f}</p>
<h2>Category macro-F1 deltas</h2>
<table><thead><tr><th>Category</th><th>Arctic minus Luxical-One</th></tr></thead>
<tbody>{category_rows}</tbody></table>
<h2>Regular-source collapse failures</h2>
<table><thead><tr><th>Category</th><th>Failures</th><th>Sources</th></tr></thead>
<tbody>{collapse_rows}</tbody></table>
<details><summary>Complete JSON</summary><pre>{html.escape(json.dumps(report, indent=2, sort_keys=True))}</pre></details>
</body></html>"""


def write_report(report: dict[str, Any]) -> tuple[str, str]:
    """Write JSON and HTML reports atomically."""
    json_url = f"{EVALUATION_ROOT}/{OUTPUT_NAME}/report.json"
    html_url = f"{EVALUATION_ROOT}/{OUTPUT_NAME}/report.html"
    for url, payload in (
        (json_url, json.dumps(report, indent=2, sort_keys=True)),
        (html_url, html_report(report)),
    ):
        filesystem, path = fsspec.core.url_to_fs(url)
        with atomic_rename(path, fs=filesystem) as temporary_path:
            with filesystem.open(temporary_path, "w") as file:
                file.write(payload)
    return json_url, html_url


@threadpool_limits.wrap(limits=CPU_THREADS)
def main() -> None:
    """Evaluate stored Arctic vectors against Luxical-One."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    manifest = read_json(MANIFEST_URL)
    texts, labels, probe_roles, categories, teacher_vectors = fixed_evaluation_data(manifest)
    left, right = pair_indices(labels)
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    baseline = Embedder.load(baseline_path)
    baseline_metrics = model_metrics(
        baseline,
        texts,
        labels,
        probe_roles,
        categories,
        teacher_vectors,
        left,
        right,
    )
    teacher_metrics = vector_metrics(teacher_vectors, labels, probe_roles, categories)
    comparison = teacher_comparison_report(teacher_metrics, baseline_metrics)
    add_source_details(comparison, source_categories(labels, categories))
    failures = failure_summary(comparison)
    report = {
        "evaluation": OUTPUT_NAME,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "evaluation_rows": len(labels),
        "predeclared_ood_sources": sorted(PREDECLARED_OOD_SOURCES),
        "present_ood_sources": sorted(set(manifest["sources"]) & PREDECLARED_OOD_SOURCES),
        "teacher": {
            "root": f"{MANIFEST_URL.rsplit('/', 1)[0]}/teacher-arctic-v1",
            "embedding_dimension": TEACHER_EMBEDDING_DIMENSION,
            "quantization_limit": TEACHER_QUANTIZATION_LIMIT,
        },
        "thresholds": {
            "minimum_unique_fraction": MIN_UNIQUE_FRACTION,
            "maximum_source_cluster_share": CLUSTER_MAX_SOURCE_SHARE,
            "cluster_count": CLUSTER_COUNT,
            "cluster_seeds": list(CLUSTER_SEEDS),
            "minimum_effective_rank_ratio": MIN_EFFECTIVE_RANK_RATIO,
            "minimum_variance_ratio": MIN_VARIANCE_RATIO,
            "minimum_quality_delta": QUALITY_DELTA,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "baseline": baseline_metrics,
        "arctic": teacher_metrics,
        "comparison": comparison,
        "failure_summary": failures,
    }
    json_url, html_url = write_report(report)
    regular_source_results = {
        source: metrics
        for source, metrics in comparison["collapse"]["source_results"].items()
        if not metrics["ood_exception"]
    }
    maximum_cluster_source, maximum_cluster_metrics = max(
        regular_source_results.items(),
        key=lambda item: item[1]["largest_cluster_share"],
    )
    summary = {
        "json_url": json_url,
        "html_url": html_url,
        "evaluation_rows": len(labels),
        "all_required_gates_passed": comparison["all_required_gates_passed"],
        "failed_gates": [name for name, passed in comparison["gates"].items() if not passed],
        "regular_collapse_failures": comparison["collapse"]["regular_failures"],
        "ood_collapse_failures": comparison["collapse"]["ood_failures"],
        "failure_summary": failures,
        "maximum_regular_source_cluster_share": {
            "source": maximum_cluster_source,
            "share": maximum_cluster_metrics["largest_cluster_share"],
        },
        "minimum_regular_source_unique_fraction_4dp": min(
            metrics["unique_fraction_4dp"] for metrics in regular_source_results.values()
        ),
        "minimum_regular_source_effective_rank_ratio": min(
            metrics["effective_rank_ratio"] for metrics in regular_source_results.values()
        ),
        "minimum_regular_source_variance_ratio": min(
            metrics["variance_ratio"] for metrics in regular_source_results.values()
        ),
        "macro_f1_delta": comparison["macro_f1_delta"],
        "worst_source_recall_delta": comparison["worst_source_recall_delta"],
        "category_macro_f1_delta": comparison["category_macro_f1_delta"],
        "probe_uncertainty": comparison["probe_uncertainty"],
        "cluster_distribution_delta": comparison["cluster_distribution_delta"],
        "baseline": {
            "macro_f1": baseline_metrics["probe"]["macro_f1"],
            "worst_source_recall": baseline_metrics["probe"]["worst_source_recall"],
            "category_macro_f1": baseline_metrics["probe"]["category_macro_f1"],
            "cluster_distribution": baseline_metrics["collapse"]["cluster_distribution"],
        },
        "arctic": {
            "macro_f1": teacher_metrics["probe"]["macro_f1"],
            "worst_source_recall": teacher_metrics["probe"]["worst_source_recall"],
            "category_macro_f1": teacher_metrics["probe"]["category_macro_f1"],
            "finite_fraction": teacher_metrics["collapse"]["finite_fraction"],
            "exact_unique_fraction": teacher_metrics["collapse"]["exact_unique_fraction"],
            "unique_fraction_4dp": teacher_metrics["collapse"]["unique_fraction_4dp"],
            "cluster_distribution": teacher_metrics["collapse"]["cluster_distribution"],
        },
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_TEACHER_EVALUATION=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
