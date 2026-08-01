# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare one 750K source-geometry treatment with the fixed baseline."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
from audit_fast_student_attribution import (
    category_geometry,
    failure_sets,
    per_source_comparison,
    set_comparison,
    source_categories,
)
from ladder_config import MANIFEST_ROOT, PREDECLARED_OOD_SOURCES, SourceCategory
from rigging.filesystem import atomic_rename

TEACHER_REPORT_URL = f"{MANIFEST_ROOT}/evaluation/teacher-arctic-v1/report.json"
BASELINE_REPORT_URL = f"{MANIFEST_ROOT}/evaluation/fast-student/full/750k/report.json"
RESULT_FILE = Path("/tmp/luxical-source-geometry-comparison")
TREATMENT_NAMES = ("full-source-geometry-w0.25", "full-source-geometry-w0.5", "full-source-geometry-w1")
QUALITY_LOSS_LIMIT = -0.02
RANK_MEDIAN_IMPROVEMENT = 0.02
VARIANCE_MEDIAN_LOSS_LIMIT = -0.02


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def failure_reason_counts(report: dict[str, Any]) -> dict[str, int]:
    """Count overlapping composite failure reasons for regular sources."""
    thresholds = report["thresholds"]
    counts = Counter()
    for source, metrics in report["comparison"]["collapse"]["source_results"].items():
        if source in PREDECLARED_OOD_SOURCES or metrics["passed"]:
            continue
        if metrics["largest_cluster_share"] > thresholds["maximum_source_cluster_share"]:
            counts["concentration"] += 1
        if metrics["unique_fraction_4dp"] < thresholds["minimum_unique_fraction"]:
            counts["uniqueness"] += 1
        if metrics["effective_rank_ratio"] < thresholds["minimum_effective_rank_ratio"]:
            counts["rank"] += 1
        if metrics["variance_ratio"] < thresholds["minimum_variance_ratio"]:
            counts["variance"] += 1
    return {reason: counts[reason] for reason in ("concentration", "uniqueness", "rank", "variance")}


def evaluation_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return the fixed decision metrics from one evaluation report."""
    student = report["student"]
    comparison = report["comparison"]
    regular = {
        source: metrics
        for source, metrics in comparison["collapse"]["source_results"].items()
        if source not in PREDECLARED_OOD_SOURCES
    }
    return {
        "macro_f1": student["probe"]["macro_f1"],
        "category_macro_f1": student["probe"]["category_macro_f1"],
        "within_source_arctic_fidelity": student["arctic_fidelity"]["within_source_spearman"],
        "regular_collapse_failures": len(comparison["collapse"]["regular_failures"]),
        "failure_reason_counts": failure_reason_counts(report),
        "minimum_effective_rank_ratio_to_stock": min(metrics["effective_rank_ratio"] for metrics in regular.values()),
        "minimum_variance_ratio_to_stock": min(metrics["variance_ratio"] for metrics in regular.values()),
        "speed_ratio": comparison["speed_ratio"],
    }


def quality_deltas(treatment: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    """Return treatment-minus-baseline quality deltas."""
    treatment_student = treatment["student"]
    baseline_student = baseline["student"]
    deltas = {
        "macro_f1": treatment_student["probe"]["macro_f1"] - baseline_student["probe"]["macro_f1"],
        "within_source_arctic_fidelity": (
            treatment_student["arctic_fidelity"]["within_source_spearman"]
            - baseline_student["arctic_fidelity"]["within_source_spearman"]
        ),
    }
    for category in (SourceCategory.CODE.value, SourceCategory.MULTILINGUAL.value, SourceCategory.STANDARD.value):
        deltas[f"{category}_macro_f1"] = (
            treatment_student["probe"]["category_macro_f1"][category]
            - baseline_student["probe"]["category_macro_f1"][category]
        )
    return deltas


def geometry_deltas(treatment: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Return treatment-minus-baseline category geometry deltas."""
    output = {}
    for category in (SourceCategory.CODE.value, SourceCategory.MULTILINGUAL.value, SourceCategory.STANDARD.value):
        output[category] = {
            "effective_rank_median_delta": (
                treatment[category]["effective_rank_ratio"]["median"]
                - baseline[category]["effective_rank_ratio"]["median"]
            ),
            "variance_median_delta": (
                treatment[category]["variance_ratio"]["median"] - baseline[category]["variance_ratio"]["median"]
            ),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment-name", choices=TREATMENT_NAMES, required=True)
    return parser.parse_args()


def main() -> None:
    """Write the fixed 750K source-geometry comparison."""
    treatment_name = parse_args().treatment_name
    treatment_report_url = f"{MANIFEST_ROOT}/evaluation/fast-student/{treatment_name}/750k/report.json"
    output_url = f"{MANIFEST_ROOT}/evaluation/fast-student/{treatment_name}/750k/comparison.json"
    teacher = read_json(TEACHER_REPORT_URL)
    baseline = read_json(BASELINE_REPORT_URL)
    treatment = read_json(treatment_report_url)
    digests = {report["manifest_sha256"] for report in (teacher, baseline, treatment)}
    if len(digests) != 1:
        raise ValueError("The teacher, baseline, and treatment reports have different manifest digests")

    categories = source_categories(teacher)
    teacher_failures, _ = failure_sets(teacher)
    baseline_failures, _ = failure_sets(baseline)
    treatment_failures, _ = failure_sets(treatment)
    baseline_attribution = set_comparison(teacher_failures, baseline_failures)
    treatment_attribution = set_comparison(teacher_failures, treatment_failures)
    baseline_geometry = category_geometry(per_source_comparison(categories, teacher, baseline))
    treatment_geometry = category_geometry(per_source_comparison(categories, teacher, treatment))
    quality = quality_deltas(treatment, baseline)
    geometry = geometry_deltas(treatment_geometry, baseline_geometry)
    training_report = treatment["training_report"]
    inference_identity = {
        "model_config_equal": treatment["training_report"]["config"] == baseline["training_report"]["config"],
        "token_remap_equal": (
            treatment["training_report"]["raw_to_compact_sha256"] == baseline["training_report"]["raw_to_compact_sha256"]
        ),
        "speed_report_equal": treatment["speed_report_url"] == baseline["speed_report_url"],
    }
    gates = {
        "quality_loss_limit": all(delta >= QUALITY_LOSS_LIMIT for delta in quality.values()),
        "regular_collapse_failures_do_not_increase": len(treatment_failures) <= len(baseline_failures),
        "student_only_failures_do_not_increase": (
            treatment_attribution["student_only_count"] <= baseline_attribution["student_only_count"]
        ),
        "rank_median_improves_in_each_category": all(
            values["effective_rank_median_delta"] >= RANK_MEDIAN_IMPROVEMENT for values in geometry.values()
        ),
        "variance_median_does_not_regress": all(
            values["variance_median_delta"] >= VARIANCE_MEDIAN_LOSS_LIMIT for values in geometry.values()
        ),
        "inference_path_is_unchanged": all(inference_identity.values()),
    }
    report = {
        "manifest_sha256": digests.pop(),
        "treatment_name": treatment_name,
        "teacher_report_url": TEACHER_REPORT_URL,
        "baseline_report_url": BASELINE_REPORT_URL,
        "treatment_report_url": treatment_report_url,
        "decision_thresholds": {
            "minimum_quality_delta": QUALITY_LOSS_LIMIT,
            "minimum_rank_median_improvement": RANK_MEDIAN_IMPROVEMENT,
            "minimum_variance_median_delta": VARIANCE_MEDIAN_LOSS_LIMIT,
        },
        "baseline": evaluation_summary(baseline),
        "treatment": evaluation_summary(treatment),
        "quality_deltas": quality,
        "baseline_teacher_attribution": baseline_attribution,
        "treatment_teacher_attribution": treatment_attribution,
        "baseline_category_geometry": baseline_geometry,
        "treatment_category_geometry": treatment_geometry,
        "category_geometry_deltas": geometry,
        "training": {
            "source_geometry_weight": training_report["source_geometry_weight"],
            "final_loss": training_report["history"][-1]["final_loss"],
            "final_distillation_loss": training_report["history"][-1]["final_distillation_loss"],
            "final_source_geometry_loss": training_report["history"][-1]["final_source_geometry_loss"],
            "elapsed_seconds": training_report["history"][-1]["elapsed_seconds"],
        },
        "inference_identity": inference_identity,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "next_action": (
            "run_3m_source_geometry_confirmation" if all(gates.values()) else "stop_source_geometry_treatment"
        ),
    }
    write_json(output_url, report)
    summary = {
        "output_url": output_url,
        "treatment_name": treatment_name,
        "baseline_regular_failures": len(baseline_failures),
        "treatment_regular_failures": len(treatment_failures),
        "baseline_student_only_failures": baseline_attribution["student_only_count"],
        "treatment_student_only_failures": treatment_attribution["student_only_count"],
        "quality_deltas": quality,
        "category_geometry_deltas": geometry,
        "gates": gates,
        "all_gates_passed": report["all_gates_passed"],
        "next_action": report["next_action"],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
