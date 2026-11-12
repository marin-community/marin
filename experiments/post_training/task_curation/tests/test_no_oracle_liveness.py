# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskcompendium.grading_result import GradeResult, Outcome
from experiments.post_training.task_curation.no_oracle_liveness import classify_result


def test_crashed_or_unexecuted_verifier_cannot_qualify_a_zero_reward():
    crashed = GradeResult(Outcome.GRADED, 0.0, detail={"total": 2}, diagnostics={"exit_code": 1, "stderr": "ImportError"})
    result = classify_result("task", "base", "snapshot", crashed)
    assert not result.passed and result.exit_code == 1 and "ImportError" in result.detail
    no_tests = GradeResult(Outcome.GRADED, 0.0, diagnostics={"exit_code": 0})
    assert not classify_result("task", "base", "snapshot", no_tests).passed
    rejected = GradeResult(Outcome.GRADED, 0.0, detail={"total": 2}, diagnostics={"exit_code": 0})
    assert classify_result("task", "base", "snapshot", rejected).passed
