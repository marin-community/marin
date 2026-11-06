# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import StdoutReward

from forge.effects.issues import Restart
from forge.lowering import requirements, script_grader
from forge.model.review import Cause
from forge.resources.grade_health import check_timeout
from forge.resources.sandbox.images import GRADER_BASE_IMAGE
from forge.stages.classify import grade_cause


@pytest.mark.parametrize("exit_code", [124, 137])
@pytest.mark.parametrize("failure", [None, GradingFailure.TIMEOUT])
def test_early_command_cut_is_restart_not_authored_timeout(exit_code: int, failure: GradingFailure | None) -> None:
    task = arithmetic_task(
        grader=script_grader(
            ("run",), StdoutReward(), environment=requirements(image=GRADER_BASE_IMAGE), timeout=600, answer_path=None
        ).grader
    )
    result = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        failure=failure,
        diagnostics={
            "exit_code": exit_code,
            "grader_elapsed_seconds": 63,
            "grader_timeout_seconds": 600,
        },
    )
    with pytest.raises(Restart, match="elapsed=63s limit=600"):
        check_timeout(result, task)


@pytest.mark.parametrize("elapsed", [None, 63, 600])
def test_only_measured_full_timeout_is_task_defect(elapsed: float | None) -> None:
    result = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        failure=GradingFailure.TIMEOUT,
        diagnostics={"grader_elapsed_seconds": elapsed, "grader_timeout_seconds": 600},
    )
    if elapsed == 600:
        assert grade_cause(result) is Cause.GRADER_TIMEOUT
    else:
        with pytest.raises(Restart):
            grade_cause(result)
