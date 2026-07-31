# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect compact corrected-run values for the final report."""

import json
import logging
from typing import Any

import fsspec
from ladder_config import MANIFEST_ROOT, PREDECLARED_OOD_SOURCES

TEACHER_AUDIT_URL = f"{MANIFEST_ROOT}/teacher-arctic-v1/audit.json"
STUDENT_ROOT = f"{MANIFEST_ROOT}/students"
EVALUATION_ROOT = f"{MANIFEST_ROOT}/evaluation"
RUNGS = ("750k", "3m")
TRAINING_FIELDS = (
    "training_rows",
    "model_url",
    "model_sha256",
    "steps",
    "first_loss",
    "final_loss",
)
AUDIT_FIELDS = (
    "manifest_sha256",
    "teacher_id",
    "teacher_revision",
    "source_count",
    "row_count",
    "minimum_source_unique_fraction",
    "minimum_source_varying_dimensions",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    with fsspec.open(url) as file:
        return json.load(file)


def collapse_failure_counts(report: dict[str, Any]) -> dict[str, int]:
    """Count each regular-source collapse failure type."""
    source_results = report["comparison"]["collapse"]["source_results"]
    regular_results = [metrics for metrics in source_results.values() if not metrics["ood_exception"]]
    baseline_results = [
        metrics
        for source, metrics in report["baseline"]["collapse"]["per_source"].items()
        if source not in PREDECLARED_OOD_SOURCES
    ]
    return {
        "regular_total": len(report["comparison"]["collapse"]["regular_failures"]),
        "cluster": sum(metrics["largest_cluster_share"] > 0.9 for metrics in regular_results),
        "unique": sum(metrics["unique_fraction_4dp"] < 0.99 for metrics in regular_results),
        "effective_rank": sum(metrics["effective_rank_ratio"] < 0.5 for metrics in regular_results),
        "variance": sum(metrics["variance_ratio"] < 0.5 for metrics in regular_results),
        "baseline_cluster": sum(metrics["largest_cluster_share"] > 0.9 for metrics in baseline_results),
    }


def main() -> None:
    """Print the compact corrected-run result."""
    audit = read_json(TEACHER_AUDIT_URL)
    rungs = {}
    for rung in RUNGS:
        training_url = f"{STUDENT_ROOT}/{rung}/training.json"
        evaluation_url = f"{EVALUATION_ROOT}/{rung}/report.json"
        training = read_json(training_url)
        evaluation = read_json(evaluation_url)
        rungs[rung] = {
            "training_url": training_url,
            "evaluation_url": evaluation_url,
            "training": {field: training[field] for field in TRAINING_FIELDS},
            "probe_uncertainty": evaluation["comparison"]["probe_uncertainty"],
            "collapse_failure_counts": collapse_failure_counts(evaluation),
        }
    result = {
        "teacher_audit_url": TEACHER_AUDIT_URL,
        "teacher_audit": {field: audit[field] for field in AUDIT_FIELDS},
        "rungs": rungs,
    }
    logger.info("LUXICAL_ARCTIC_RESULTS=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
