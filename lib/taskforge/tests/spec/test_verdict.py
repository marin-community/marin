# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.spec.verdict import GraderCheck, grader_checks, verdict_json

CHECKS = (GraderCheck("heading", True, 1.0), GraderCheck("total", False, 2.0))


def test_grader_checks_round_trip_through_the_verdict_file():
    verdict = json.loads(verdict_json(1 / 3, CHECKS))

    grade = GradeResult(Outcome.GRADED, verdict["reward"], detail=verdict["detail"])

    assert grader_checks(grade) == CHECKS


@pytest.mark.parametrize("detail", [None, {"reason": "collection failed"}])
def test_a_grade_without_checks_has_none(detail):
    assert grader_checks(GradeResult(Outcome.GRADED, 1.0, detail=detail)) is None


@pytest.mark.parametrize(
    "checks",
    [
        [{"name": "total", "weight": 1.0}],
        [{"name": "total", "passed": "yes", "weight": 1.0}],
        [{"name": "total", "passed": True, "weight": True}],
        {"name": "total", "passed": True, "weight": 1.0},
    ],
)
def test_malformed_checks_raise(checks):
    with pytest.raises(ValueError):
        grader_checks(GradeResult(Outcome.GRADED, 1.0, detail={"checks": checks}))
