# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and upload one blind-neighborhood review report."""

import argparse
import json
from pathlib import Path
from typing import Any

from publish_fast_embedding_bundle import BLIND_REVIEW_MODEL, write_once
from rigging.filesystem import StoragePath
from verify_blind_neighborhood_with_claude import review_package_sha256, validate_decisions

RESULT_FILE = Path("/tmp/luxical-blind-review-upload")


def validate_review_report(package: dict[str, Any], report: dict[str, Any]) -> None:
    """Reject a review report that does not match its private package."""
    items = package["items"]
    decisions = report["decisions"]
    validate_decisions(package, decisions)
    expected = {
        "package_sha256": review_package_sha256(package),
        "claude_model": BLIND_REVIEW_MODEL,
        "student_model": package["student_model"],
    }
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError("The blind review does not identify its exact package and model")
    if report["overall"]["documents"] != len(items) or len(items) != 200:
        raise ValueError("The blind review does not contain exactly 200 documents")
    subgroup_documents = sum(report[name]["documents"] for name in ("code", "non_english", "other"))
    if subgroup_documents != len(items):
        raise ValueError("The blind review subgroup counts do not cover its documents")


def main() -> None:
    """Read, validate, and upload one immutable review report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--package-url", required=True)
    parser.add_argument("--output-url", required=True)
    args = parser.parse_args()

    report_payload = args.report_path.read_bytes()
    report = json.loads(report_payload)
    package = json.loads(StoragePath(args.package_url).read_text(compression="gzip"))
    validate_review_report(package, report)
    write_once(StoragePath(args.output_url), report_payload)
    result = {
        "output_url": args.output_url,
        "package_sha256": report["package_sha256"],
        "student_model": report["student_model"],
        "claude_model": report["claude_model"],
        "documents": report["overall"]["documents"],
    }
    RESULT_FILE.write_text(json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
