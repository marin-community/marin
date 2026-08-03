# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and upload one GLM hierarchy adjudication report."""

import argparse
import json
from typing import Any

from export_glm_hierarchy_adjudication import DEFAULT_TAIL_FRACTION, adjudication_package
from glm_hierarchical_labels import OUTPUT_ROOT, HierarchicalAssignment, parse_hierarchy
from glm_semantic_labels import SampleDocument, read_json, read_jsonl
from publish_fast_embedding_bundle import BLIND_REVIEW_MODEL, write_once
from rigging.filesystem import StoragePath
from verify_glm_hierarchy_with_claude import comparison, review_package_sha256, validate_claude_rows


def validate_adjudication_report(package: dict[str, Any], report: dict[str, Any]) -> None:
    """Reject a review report that does not match its exact private package."""
    rows = report.get("claude_assignments")
    metrics = report.get("adjudication")
    if not isinstance(rows, list) or not isinstance(metrics, dict):
        raise ValueError("The adjudication report has no complete assignment result")
    expected = {
        "claude_model": BLIND_REVIEW_MODEL,
        "package_sha256": review_package_sha256(package),
    }
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError("The adjudication report has a different package or model")
    validate_claude_rows(package, rows)
    expected_metrics = comparison(package, rows)["adjudication"]
    if metrics != expected_metrics:
        raise ValueError("The adjudication metrics do not match the assignments")


def load_adjudication_package(
    pilot_run_id: str,
    variant: str,
    evaluation_run_id: str,
    tail_fraction: float,
) -> dict[str, Any]:
    """Load and rebuild the exact source-blind adjudication package."""
    variant_root = OUTPUT_ROOT / pilot_run_id / variant
    evaluation_root = variant_root / evaluation_run_id
    summary = read_json(str(evaluation_root / "summary.json"))
    if summary.get("complete") is not True:
        raise ValueError("The held-out GLM run is not complete")
    documents = [SampleDocument(**row) for row in read_jsonl(evaluation_root / "sample-private.jsonl.gz")]
    taxonomy = read_json(str(variant_root / "taxonomy.json"))
    parse_hierarchy(taxonomy)
    assignment_paths = sorted((evaluation_root / "assignments" / "*.jsonl.gz").glob(), key=str)
    assignments = [HierarchicalAssignment(**row) for path in assignment_paths for row in read_jsonl(path)]
    if len(assignments) != len(documents):
        raise ValueError("The held-out hierarchy assignments are not complete")
    return adjudication_package(documents, assignments, taxonomy, tail_fraction)


def main() -> None:
    """Load, validate, and upload one immutable adjudication report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-url", required=True)
    parser.add_argument("--pilot-run-id", required=True)
    parser.add_argument("--variant", choices=("compact", "balanced"), required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    parser.add_argument("--output-url", required=True)
    args = parser.parse_args()

    report_payload = StoragePath(args.report_url).read_bytes()
    report = json.loads(report_payload)
    package = load_adjudication_package(
        args.pilot_run_id,
        args.variant,
        args.evaluation_run_id,
        args.tail_fraction,
    )
    validate_adjudication_report(package, report)
    write_once(StoragePath(args.output_url), report_payload)
    print(
        json.dumps(
            {
                "claude_model": report["claude_model"],
                "documents": report["adjudication"]["documents"],
                "output_url": args.output_url,
                "package_sha256": report["package_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
