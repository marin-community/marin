# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the 750K cross-dimension Qwen student on fixed gates."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from evaluate_fast_student import SPEED_REPORT_URL, load_student, speed_metrics
from evaluate_ladder import (
    BASELINE_FILE,
    BASELINE_REPO,
    BASELINE_REVISION,
    CLUSTER_MAX_SOURCE_SHARE,
    CPU_THREADS,
    MANIFEST_URL,
    MIN_UNIQUE_FRACTION,
    MIN_VARIANCE_RATIO,
    PREDECLARED_OOD_SOURCES,
    QUALITY_DELTA,
    comparison_report,
    cosine_fidelity,
    embed_on_cpu,
    fixed_evaluation_data,
    pair_indices,
    read_json,
    vector_metrics,
)
from evaluate_teacher_candidate import (
    CANDIDATES,
    candidate_output_url,
    expected_metadata,
    normalized_vectors,
    quantized_vectors,
)
from huggingface_hub import hf_hub_download
from ladder_config import MANIFEST_ROOT, SourceCategory
from luxical.embedder import Embedder
from rigging.filesystem import atomic_rename
from threadpoolctl import threadpool_limits

TRAINING_NAME = "full-qwen3-06b-1024-crossdim"
RUNG = "750k"
QWEN_CANDIDATE_NAME = "qwen3-embedding-0.6b"
QWEN_REPORT_URL = f"{MANIFEST_ROOT}/evaluation/teacher-{QWEN_CANDIDATE_NAME}/report.json"
OUTPUT_ROOT = f"{MANIFEST_ROOT}/evaluation/fast-student/{TRAINING_NAME}/{RUNG}"
JSON_URL = f"{OUTPUT_ROOT}/report.json"
HTML_URL = f"{OUTPUT_ROOT}/report.html"
RESULT_FILE = Path("/tmp/luxical-qwen-fast-student-evaluation")
STUDENT_ONLY_FAILURE_LIMIT = 5
CATEGORY_MEDIAN_RATIO_LIMIT = 0.50
CPU_SPEED_RATIO_LIMIT = 0.85

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def qwen_evaluation_vectors(manifest: dict[str, Any]) -> np.ndarray:
    """Load Qwen vectors in the fixed evaluation-row order."""
    candidate = CANDIDATES[QWEN_CANDIDATE_NAME]
    batches = []
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Loading Qwen evaluation source %d/%d: %s", index, len(manifest["sources"]), source)
        manifest_filesystem, manifest_path = fsspec.core.url_to_fs(result["output_url"])
        source_table = pq.read_table(
            manifest_path,
            filesystem=manifest_filesystem,
            columns=["raw_sha256", "split", "eval_rank"],
        )
        source_table = source_table.filter(pc.equal(source_table["split"], "eval"))
        output_url = candidate_output_url(candidate, result["output_url"])
        teacher_filesystem, teacher_path = fsspec.core.url_to_fs(output_url)
        teacher_table = pq.read_table(
            teacher_path,
            filesystem=teacher_filesystem,
            columns=["raw_sha256", "embedding"],
        )
        metadata = teacher_table.schema.metadata or {}
        expected = expected_metadata(candidate, manifest["sha256"])
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Qwen evaluation metadata differs for {source}")
        if source_table["raw_sha256"].to_pylist() != teacher_table["raw_sha256"].to_pylist():
            raise ValueError(f"Qwen evaluation rows are not aligned for {source}")
        batches.append(normalized_vectors(quantized_vectors(teacher_table, candidate.embedding_dimension)))
    output = np.concatenate(batches)
    if not np.isfinite(output).all():
        raise ValueError("Qwen evaluation vectors contain non-finite values")
    return output


def source_categories(labels: np.ndarray, categories: np.ndarray) -> dict[str, str]:
    """Return the fixed category for each source."""
    output = {}
    for label, category in zip(labels, categories, strict=True):
        previous = output.setdefault(str(label), str(category))
        if previous != category:
            raise ValueError(f"Source {label} has multiple categories")
    return output


def category_geometry(
    student_metrics: dict[str, Any],
    qwen_report: dict[str, Any],
    categories: dict[str, str],
) -> dict[str, Any]:
    """Measure student-to-Qwen variance ratios and diagnostic rank ratios."""
    qwen_sources = qwen_report["candidate"]["collapse"]["per_source"]
    student_sources = student_metrics["collapse"]["per_source"]
    if set(qwen_sources) != set(student_sources):
        raise ValueError("Qwen and student source sets differ")
    output = {}
    for category in (SourceCategory.CODE.value, SourceCategory.MULTILINGUAL.value, SourceCategory.STANDARD.value):
        sources = [source for source, value in categories.items() if value == category]
        rank_ratios = [
            student_sources[source]["effective_rank"] / max(qwen_sources[source]["effective_rank"], 1e-30)
            for source in sources
        ]
        variance_ratios = [
            student_sources[source]["total_variance"] / max(qwen_sources[source]["total_variance"], 1e-30)
            for source in sources
        ]
        rank_median = float(np.median(rank_ratios))
        variance_median = float(np.median(variance_ratios))
        output[category] = {
            "source_count": len(sources),
            "effective_rank_ratio_median_diagnostic": rank_median,
            "variance_ratio_median": variance_median,
            "variance_gate_passed": variance_median >= CATEGORY_MEDIAN_RATIO_LIMIT,
        }
    return output


def failure_attribution(comparison: dict[str, Any], qwen_report: dict[str, Any]) -> dict[str, Any]:
    """Compare regular student and Qwen failure sets."""
    student_failures = set(comparison["collapse"]["regular_failures"])
    qwen_failures = set(qwen_report["candidate_vs_luxical"]["collapse"]["regular_failures"])
    overlap = student_failures & qwen_failures
    union = student_failures | qwen_failures
    return {
        "teacher_failure_count": len(qwen_failures),
        "student_failure_count": len(student_failures),
        "overlap_count": len(overlap),
        "student_only_count": len(student_failures - qwen_failures),
        "teacher_only_count": len(qwen_failures - student_failures),
        "jaccard": len(overlap) / len(union) if union else 1.0,
        "student_only": sorted(student_failures - qwen_failures),
        "teacher_only": sorted(qwen_failures - student_failures),
        "overlap": sorted(overlap),
    }


def compact_html(report: dict[str, Any]) -> str:
    """Return a small HTML view of the canonical JSON report."""
    summary = report["summary"]
    gates = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(name, "pass" if passed else "fail")
        for name, passed in summary["gates"].items()
    )
    style = (
        "<style>body{font-family:sans-serif;margin:2rem;max-width:80rem}"
        "td,th{border:1px solid #bbb;padding:.35rem}"
        "table{border-collapse:collapse}</style>"
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Qwen fast-student evaluation</title>
{style}
</head><body><h1>Qwen fast-student evaluation</h1>
<p>All POC gates passed: {summary['all_poc_gates_passed']}</p>
<p>Macro-F1: {summary['student_macro_f1']:.5f}</p>
<p>Regular failures: {summary['regular_failure_count']}</p>
<p>Student-only failures: {summary['student_only_failure_count']}</p>
<p>CPU speed ratio: {summary['cpu_speed_ratio']:.4f}</p>
<table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gates}</tbody></table>
<p>The JSON report is canonical.</p></body></html>"""


def write_report(report: dict[str, Any]) -> None:
    """Write the JSON and HTML reports atomically."""
    for url, payload in (
        (JSON_URL, json.dumps(report, indent=2, sort_keys=True)),
        (HTML_URL, compact_html(report)),
    ):
        filesystem, path = fsspec.core.url_to_fs(url)
        with atomic_rename(path, fs=filesystem) as temporary_path:
            with filesystem.open(temporary_path, "w") as file:
                file.write(payload)


@threadpool_limits.wrap(limits=CPU_THREADS)
def main() -> None:
    """Run the fixed quality, geometry, failure, fidelity, and speed gates."""
    manifest = read_json(MANIFEST_URL)
    texts, labels, probe_roles, categories, arctic_vectors = fixed_evaluation_data(manifest)
    qwen_vectors = qwen_evaluation_vectors(manifest)
    if len(qwen_vectors) != len(texts):
        raise ValueError(f"Qwen vectors have {len(qwen_vectors)} rows; expected {len(texts)}")
    left, right = pair_indices(labels)
    baseline_path = hf_hub_download(
        repo_id=BASELINE_REPO,
        filename=BASELINE_FILE,
        revision=BASELINE_REVISION,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        student, training_report = load_student("full", TRAINING_NAME, RUNG, Path(temporary_directory))
        baseline = Embedder.load(baseline_path)
        baseline_vectors = embed_on_cpu(baseline, texts)
        student_vectors = embed_on_cpu(student, texts)

    baseline_metrics = vector_metrics(baseline_vectors, labels, probe_roles, categories)
    student_metrics = vector_metrics(student_vectors, labels, probe_roles, categories)
    baseline_metrics["arctic_fidelity"] = cosine_fidelity(baseline_vectors, arctic_vectors, left, right)
    student_metrics["arctic_fidelity"] = cosine_fidelity(student_vectors, arctic_vectors, left, right)
    baseline_metrics["qwen_fidelity"] = cosine_fidelity(baseline_vectors, qwen_vectors, left, right)
    student_metrics["qwen_fidelity"] = cosine_fidelity(student_vectors, qwen_vectors, left, right)

    speed_report = read_json(SPEED_REPORT_URL)
    baseline_metrics["speed"] = speed_metrics(speed_report, "baseline")
    student_metrics["speed"] = speed_metrics(speed_report, "student")
    if training_report["parameters"] > speed_report["parameters"]:
        raise ValueError("The trained model is larger than the paired speed model")
    comparison = comparison_report(student_metrics, baseline_metrics)
    qwen_report = read_json(QWEN_REPORT_URL)
    attribution = failure_attribution(comparison, qwen_report)
    geometry = category_geometry(student_metrics, qwen_report, source_categories(labels, categories))
    qwen_fidelity_delta = (
        student_metrics["qwen_fidelity"]["within_source_spearman"]
        - baseline_metrics["qwen_fidelity"]["within_source_spearman"]
    )
    selected_comparison_gates = {
        name: value
        for name, value in comparison["gates"].items()
        if name not in ("regular_source_collapse", "arctic_fidelity", "cpu_speed_minimum")
    }
    gates = selected_comparison_gates | {
        "student_only_failures": attribution["student_only_count"] <= STUDENT_ONLY_FAILURE_LIMIT,
        "category_median_variance": all(result["variance_gate_passed"] for result in geometry.values()),
        "qwen_fidelity": qwen_fidelity_delta >= 0.0,
        "cpu_speed": comparison["speed_ratio"] >= CPU_SPEED_RATIO_LIMIT,
    }
    summary = {
        "all_poc_gates_passed": all(gates.values()),
        "gates": gates,
        "student_macro_f1": student_metrics["probe"]["macro_f1"],
        "category_macro_f1": student_metrics["probe"]["category_macro_f1"],
        "regular_failure_count": len(comparison["collapse"]["regular_failures"]),
        "student_only_failure_count": attribution["student_only_count"],
        "cpu_speed_ratio": comparison["speed_ratio"],
        "qwen_within_source_fidelity": student_metrics["qwen_fidelity"]["within_source_spearman"],
        "qwen_within_source_fidelity_delta": qwen_fidelity_delta,
        "finite_fraction": student_metrics["collapse"]["finite_fraction"],
        "unique_fraction_4dp": student_metrics["collapse"]["unique_fraction_4dp"],
    }
    comparison_for_report = {
        name: value for name, value in comparison.items() if name not in ("all_required_gates_passed", "gates")
    }
    report = {
        "evaluation": f"{TRAINING_NAME}-{RUNG}",
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "training_report": training_report,
        "qwen_report_url": QWEN_REPORT_URL,
        "speed_report_url": SPEED_REPORT_URL,
        "predeclared_ood_sources": sorted(PREDECLARED_OOD_SOURCES),
        "thresholds": {
            "minimum_quality_delta": QUALITY_DELTA,
            "minimum_unique_fraction": MIN_UNIQUE_FRACTION,
            "maximum_source_cluster_share": CLUSTER_MAX_SOURCE_SHARE,
            "minimum_variance_ratio": MIN_VARIANCE_RATIO,
            "maximum_student_only_failures": STUDENT_ONLY_FAILURE_LIMIT,
            "minimum_category_median_student_to_teacher_variance_ratio": CATEGORY_MEDIAN_RATIO_LIMIT,
            "minimum_cpu_speed_ratio": CPU_SPEED_RATIO_LIMIT,
        },
        "baseline": baseline_metrics,
        "student": student_metrics,
        "comparison": comparison_for_report,
        "qwen_failure_attribution": attribution,
        "student_to_qwen_geometry": geometry,
        "summary": summary,
    }
    write_report(report)
    result = {"json_url": JSON_URL, "html_url": HTML_URL, **summary}
    RESULT_FILE.write_text(json.dumps(result, sort_keys=True))
    logger.info("QWEN_FAST_STUDENT_EVALUATION=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
