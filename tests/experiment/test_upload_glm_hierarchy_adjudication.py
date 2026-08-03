# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from upload_glm_hierarchy_adjudication import validate_adjudication_report  # noqa: E402
from verify_glm_hierarchy_with_claude import comparison, review_package_sha256  # noqa: E402


def adjudication_inputs() -> tuple[dict, dict]:
    package = {
        "taxonomy": {
            "parents": [{"bucket_id": "SCIENCE"}],
            "leaves": [{"bucket_id": "BIOLOGY", "parent_id": "SCIENCE"}],
            "forms": [{"bucket_id": "RESEARCH"}],
        },
        "documents": [{"sample_index": 7, "text": "Biology"}],
        "glm_assignments": [
            {
                "sample_index": 7,
                "primary_parent_id": "SCIENCE",
                "secondary_parent_ids": [],
                "primary_leaf_id": "BIOLOGY",
                "secondary_leaf_ids": [],
                "form_id": "RESEARCH",
                "confidence": 0.4,
                "rationale": "Biology document.",
            }
        ],
        "samples": {"adjudication": [7]},
    }
    rows = [package["glm_assignments"][0] | {"confidence": 0.9, "rationale": "Biology research."}]
    report = {
        "adjudication": comparison(package, rows)["adjudication"],
        "claude_assignments": rows,
        "claude_model": "claude-opus-5",
        "package_sha256": review_package_sha256(package),
    }
    return package, report


def test_validate_adjudication_report_accepts_exact_report() -> None:
    package, report = adjudication_inputs()

    validate_adjudication_report(package, report)


@pytest.mark.parametrize("fault", ["package", "model", "index", "metrics"])
def test_validate_adjudication_report_rejects_wrong_identity_or_result(fault: str) -> None:
    package, report = adjudication_inputs()
    if fault == "package":
        report["package_sha256"] = "wrong"
    elif fault == "model":
        report["claude_model"] = "claude-other"
    elif fault == "index":
        report["claude_assignments"][0]["sample_index"] = 8
    else:
        report["adjudication"]["form_exact_agreement"] = 0.0

    with pytest.raises(ValueError):
        validate_adjudication_report(package, report)
