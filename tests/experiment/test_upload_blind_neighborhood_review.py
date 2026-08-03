# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from upload_blind_neighborhood_review import validate_review_report  # noqa: E402
from verify_blind_neighborhood_with_claude import review_package_sha256  # noqa: E402


def review_inputs() -> tuple[dict, dict]:
    items = [
        {
            "sample_index": index,
            "student_side": "A",
        }
        for index in range(200)
    ]
    package = {"student_model": "student", "items": items}
    decisions = [
        {
            "sample_index": index,
            "choice": "A",
            "query_language": "en",
            "code_central": index < 40,
        }
        for index in range(200)
    ]
    report = {
        "package_sha256": review_package_sha256(package),
        "claude_model": "claude-opus-5",
        "student_model": "student",
        "decisions": decisions,
        "overall": {"documents": 200},
        "code": {"documents": 40},
        "non_english": {"documents": 30},
        "other": {"documents": 130},
    }
    return package, report


def test_validate_review_report_accepts_one_exact_complete_review() -> None:
    package, report = review_inputs()

    validate_review_report(package, report)


@pytest.mark.parametrize("fault", ["package", "model", "missing", "subgroups"])
def test_validate_review_report_rejects_wrong_identity_or_population(fault: str) -> None:
    package, report = review_inputs()
    if fault == "package":
        report["package_sha256"] = "wrong"
    elif fault == "model":
        report["claude_model"] = "claude-other"
    elif fault == "missing":
        report["decisions"].pop()
    else:
        report["other"]["documents"] -= 1

    with pytest.raises(ValueError):
        validate_review_report(package, report)
