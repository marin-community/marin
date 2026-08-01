# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attribute fast-student collapse failures to the Arctic teacher."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from evaluate_ladder import MANIFEST_URL, PREDECLARED_OOD_SOURCES, read_json
from ladder_config import MANIFEST_ROOT, TEACHER_ID, TEACHER_REVISION, SourceCategory, teacher_windows_from_view
from rigging.filesystem import atomic_rename
from scipy.stats import spearmanr
from transformers import AutoTokenizer, PreTrainedTokenizerFast

TEACHER_REPORT_URL = f"{MANIFEST_ROOT}/evaluation/teacher-arctic-v1/report.json"
STUDENT_REPORT_URL = f"{MANIFEST_ROOT}/evaluation/fast-student/full/3m/report.json"
OUTPUT_URL = f"{MANIFEST_ROOT}/evaluation/fast-student/full/3m/attribution.json"
RESULT_FILE = Path("/tmp/luxical-fast-student-attribution")
MAX_TEACHER_TOKENS = 512
TOKENIZER_BATCH_SIZE = 1_024
STUDENT_ONLY_FAILURE_LIMIT = 5
CATEGORY_MEDIAN_RATIO_LIMIT = 0.50
CONCENTRATION_THRESHOLDS = (0.85, 0.90, 0.95)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def quantiles(values: list[float]) -> dict[str, float]:
    """Return compact distribution values."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def source_categories(teacher_report: dict[str, Any]) -> dict[str, str]:
    """Return the fixed category for each source."""
    results = teacher_report["comparison"]["collapse"]["source_results"]
    return {source: metrics["source_category"] for source, metrics in results.items()}


def failure_sets(report: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return regular failure and pass sets."""
    results = report["comparison"]["collapse"]["source_results"]
    regular = {source for source in results if source not in PREDECLARED_OOD_SOURCES}
    failures = {source for source in regular if not results[source]["passed"]}
    return failures, regular - failures


def set_comparison(teacher_failures: set[str], student_failures: set[str]) -> dict[str, Any]:
    """Compare teacher and student failure sets."""
    overlap = teacher_failures & student_failures
    union = teacher_failures | student_failures
    return {
        "teacher_failure_count": len(teacher_failures),
        "student_failure_count": len(student_failures),
        "overlap_count": len(overlap),
        "student_only_count": len(student_failures - teacher_failures),
        "teacher_only_count": len(teacher_failures - student_failures),
        "jaccard": len(overlap) / len(union),
        "overlap": sorted(overlap),
        "student_only": sorted(student_failures - teacher_failures),
        "teacher_only": sorted(teacher_failures - student_failures),
    }


def category_set_comparison(
    categories: dict[str, str],
    teacher_failures: set[str],
    student_failures: set[str],
) -> dict[str, Any]:
    """Compare failure sets in each regular category."""
    output = {}
    for category in (SourceCategory.CODE.value, SourceCategory.MULTILINGUAL.value, SourceCategory.STANDARD.value):
        sources = {source for source, value in categories.items() if value == category}
        output[category] = {
            "source_count": len(sources),
            "teacher_failures": len(sources & teacher_failures),
            "student_failures": len(sources & student_failures),
            "overlap": len(sources & teacher_failures & student_failures),
            "student_only": len((sources & student_failures) - teacher_failures),
            "teacher_only": len((sources & teacher_failures) - student_failures),
        }
    return output


def per_source_comparison(
    categories: dict[str, str],
    teacher_report: dict[str, Any],
    student_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare raw source metrics without cluster-ID alignment."""
    teacher_results = teacher_report["arctic"]["collapse"]["per_source"]
    student_results = student_report["student"]["collapse"]["per_source"]
    if set(teacher_results) != set(student_results):
        raise ValueError("Teacher and student source sets differ")
    output = {}
    for source in sorted(teacher_results):
        teacher = teacher_results[source]
        student = student_results[source]
        output[source] = {
            "source_category": categories[source],
            "teacher_largest_cluster_share": teacher["largest_cluster_share"],
            "student_largest_cluster_share": student["largest_cluster_share"],
            "largest_cluster_share_delta": student["largest_cluster_share"] - teacher["largest_cluster_share"],
            "teacher_unique_fraction_4dp": teacher["unique_fraction_4dp"],
            "student_unique_fraction_4dp": student["unique_fraction_4dp"],
            "student_to_teacher_effective_rank_ratio": student["effective_rank"] / max(teacher["effective_rank"], 1e-30),
            "student_to_teacher_variance_ratio": student["total_variance"] / max(teacher["total_variance"], 1e-30),
        }
    return output


def category_geometry(source_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize student-to-teacher geometry ratios by category."""
    output = {}
    for category in (SourceCategory.CODE.value, SourceCategory.MULTILINGUAL.value, SourceCategory.STANDARD.value):
        metrics = [values for values in source_results.values() if values["source_category"] == category]
        rank = quantiles([values["student_to_teacher_effective_rank_ratio"] for values in metrics])
        variance = quantiles([values["student_to_teacher_variance_ratio"] for values in metrics])
        output[category] = {
            "source_count": len(metrics),
            "effective_rank_ratio": rank,
            "variance_ratio": variance,
            "median_gate_passed": (
                rank["median"] >= CATEGORY_MEDIAN_RATIO_LIMIT and variance["median"] >= CATEGORY_MEDIAN_RATIO_LIMIT
            ),
        }
    return output


def concentration_sets(
    report: dict[str, Any],
    threshold: float,
) -> set[str]:
    """Return regular sources above one concentration threshold."""
    results = report["comparison"]["collapse"]["source_results"]
    return {
        source
        for source, metrics in results.items()
        if source not in PREDECLARED_OOD_SOURCES and metrics["largest_cluster_share"] > threshold
    }


def concentration_sensitivity(
    categories: dict[str, str],
    teacher_report: dict[str, Any],
    student_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare concentration sets at three fixed thresholds."""
    output = {}
    for threshold in CONCENTRATION_THRESHOLDS:
        teacher = concentration_sets(teacher_report, threshold)
        student = concentration_sets(student_report, threshold)
        output[str(threshold)] = {
            "all_regular": set_comparison(teacher, student),
            "by_category": category_set_comparison(categories, teacher, student),
        }
    return output


def seed_failure_count(metrics: dict[str, Any], threshold: float) -> int:
    """Count cluster seeds above one source threshold."""
    return sum(share > threshold for share in metrics["largest_cluster_share_by_seed"].values())


def seed_stability(report: dict[str, Any]) -> dict[str, Any]:
    """Count regular sources by the number of failed cluster seeds."""
    results = report["comparison"]["collapse"]["source_results"]
    counts = Counter(
        seed_failure_count(metrics, 0.90) for source, metrics in results.items() if source not in PREDECLARED_OOD_SOURCES
    )
    return {"failed_seed_count": {str(count): counts[count] for count in range(4)}}


def tokenizer_lengths(tokenizer: PreTrainedTokenizerFast, texts: list[str]) -> list[int]:
    """Return exact untruncated token counts in bounded batches."""
    lengths = []
    for start in range(0, len(texts), TOKENIZER_BATCH_SIZE):
        inputs = tokenizer(
            texts[start : start + TOKENIZER_BATCH_SIZE],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths.extend(map(int, inputs["length"]))
    return lengths


def truncation_metrics(lengths: list[int]) -> dict[str, Any]:
    """Summarize window and document truncation."""
    array = np.asarray(lengths, dtype=np.int64).reshape(-1, 3)
    truncated = array > MAX_TEACHER_TOKENS
    document_tokens = array.max(axis=1)
    return {
        "documents": len(array),
        "windows": int(array.size),
        "truncated_documents": int(truncated.any(axis=1).sum()),
        "truncated_document_fraction": float(truncated.any(axis=1).mean()),
        "truncated_windows": int(truncated.sum()),
        "truncated_window_fraction": float(truncated.mean()),
        "maximum_window_token_count": int(array.max()),
        "maximum_tokens_per_document": quantiles(list(map(float, document_tokens))),
    }


def truncation_audit(manifest: dict[str, Any], categories: dict[str, str]) -> dict[str, Any]:
    """Measure exact Arctic truncation on the fixed evaluation texts."""
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_ID,
        revision=TEACHER_REVISION,
        trust_remote_code=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(f"Expected a fast tokenizer, got {type(tokenizer).__name__}")
    per_source = {}
    all_lengths = []
    category_lengths: dict[str, list[int]] = {category.value: [] for category in SourceCategory}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Token audit source %d/%d: %s", index, len(manifest["sources"]), source)
        filesystem, path = fsspec.core.url_to_fs(result["output_url"])
        table = pq.read_table(path, filesystem=filesystem, columns=["split", "text"])
        table = table.filter(pc.equal(table["split"], "eval"))
        texts = table["text"].to_pylist()
        windows = [window for text in texts for window in teacher_windows_from_view(text)]
        lengths = tokenizer_lengths(tokenizer, windows)
        metrics = truncation_metrics(lengths)
        metrics["source_category"] = categories[source]
        per_source[source] = metrics
        all_lengths.extend(lengths)
        category_lengths[categories[source]].extend(lengths)
    return {
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "maximum_teacher_tokens": MAX_TEACHER_TOKENS,
        "definition": "a window is truncated when its token count with special tokens is greater than 512",
        "all": truncation_metrics(all_lengths),
        "by_category": {
            category: truncation_metrics(lengths) for category, lengths in category_lengths.items() if lengths
        },
        "per_source": per_source,
    }


def finite_spearman(left: list[float], right: list[float]) -> float:
    """Return a finite Spearman value."""
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError("Spearman correlation is not finite")
    return value


def truncation_association(
    source_results: dict[str, dict[str, Any]],
    truncation: dict[str, Any],
    teacher_failures: set[str],
    student_failures: set[str],
) -> dict[str, Any]:
    """Compare truncation with failure and concentration values."""
    regular = sorted(source for source in source_results if source not in PREDECLARED_OOD_SOURCES)
    fractions = [truncation["per_source"][source]["truncated_window_fraction"] for source in regular]
    teacher_shares = [source_results[source]["teacher_largest_cluster_share"] for source in regular]
    student_shares = [source_results[source]["student_largest_cluster_share"] for source in regular]

    def group_mean(sources: set[str]) -> float | None:
        if not sources:
            return None
        return float(np.mean([truncation["per_source"][source]["truncated_window_fraction"] for source in sources]))

    all_sources = set(regular)
    return {
        "spearman_with_teacher_largest_cluster_share": finite_spearman(fractions, teacher_shares),
        "spearman_with_student_largest_cluster_share": finite_spearman(fractions, student_shares),
        "mean_truncated_window_fraction": {
            "teacher_failures": group_mean(teacher_failures),
            "teacher_passes": group_mean(all_sources - teacher_failures),
            "student_failures": group_mean(student_failures),
            "student_passes": group_mean(all_sources - student_failures),
            "student_only_failures": group_mean(student_failures - teacher_failures),
        },
    }


def main() -> None:
    """Run the saved-report and teacher-truncation attribution audit."""
    manifest = read_json(MANIFEST_URL)
    teacher_report = read_json(TEACHER_REPORT_URL)
    student_report = read_json(STUDENT_REPORT_URL)
    expected_digest = manifest["sha256"]
    for name, report in (("teacher", teacher_report), ("student", student_report)):
        if report["manifest_sha256"] != expected_digest:
            raise ValueError(f"The {name} report has a different manifest digest")

    categories = source_categories(teacher_report)
    teacher_failures, teacher_passes = failure_sets(teacher_report)
    student_failures, student_passes = failure_sets(student_report)
    source_results = per_source_comparison(categories, teacher_report, student_report)
    geometry = category_geometry(source_results)
    truncation = truncation_audit(manifest, categories)
    comparison = set_comparison(teacher_failures, student_failures)
    decision_gates = {
        "student_only_failures": comparison["student_only_count"] <= STUDENT_ONLY_FAILURE_LIMIT,
        "no_category_wide_rank_or_variance_loss": all(category["median_gate_passed"] for category in geometry.values()),
    }
    report = {
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": expected_digest,
        "teacher_report_url": TEACHER_REPORT_URL,
        "student_report_url": STUDENT_REPORT_URL,
        "method_limits": {
            "cluster_ids_are_not_aligned": True,
            "reason": "each saved model report fit its own cluster model",
        },
        "decision_thresholds": {
            "maximum_student_only_failures": STUDENT_ONLY_FAILURE_LIMIT,
            "minimum_category_median_student_to_teacher_rank_ratio": CATEGORY_MEDIAN_RATIO_LIMIT,
            "minimum_category_median_student_to_teacher_variance_ratio": CATEGORY_MEDIAN_RATIO_LIMIT,
        },
        "failure_set_comparison": comparison,
        "failure_sets_by_category": category_set_comparison(categories, teacher_failures, student_failures),
        "concentration_threshold_sensitivity": concentration_sensitivity(
            categories,
            teacher_report,
            student_report,
        ),
        "concentration_seed_stability": {
            "teacher": seed_stability(teacher_report),
            "student": seed_stability(student_report),
        },
        "category_geometry": geometry,
        "truncation": truncation,
        "truncation_association": truncation_association(
            source_results,
            truncation,
            teacher_failures,
            student_failures,
        ),
        "source_results": source_results,
        "decision_gates": decision_gates,
        "all_decision_gates_passed": all(decision_gates.values()),
        "next_action": (
            "run_10m_canary" if all(decision_gates.values()) else "run_750k_source_conditioned_regularizer_ablation"
        ),
        "set_accounting": {
            "teacher": len(teacher_failures) + len(teacher_passes),
            "student": len(student_failures) + len(student_passes),
        },
    }
    write_json(OUTPUT_URL, report)
    summary = {
        "output_url": OUTPUT_URL,
        "student_only_failures": comparison["student_only_count"],
        "overlap_failures": comparison["overlap_count"],
        "decision_gates": decision_gates,
        "all_decision_gates_passed": report["all_decision_gates_passed"],
        "next_action": report["next_action"],
        "truncated_window_fraction": truncation["all"]["truncated_window_fraction"],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("FAST_STUDENT_ATTRIBUTION=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
