# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject infrastructure cuts before a grade can enter the replay cache."""

import math

from taskcompendium.grading_result import GradeResult, GradingFailure
from taskcompendium.models import ScriptGrader, TaskSpec

from forge.effects.issues import Restart


def check_timeout(grade: GradeResult, task: TaskSpec | None = None) -> None:
    elapsed = grade.diagnostics.get("grader_elapsed_seconds")
    limit = (
        task.grader.timeout
        if task is not None and isinstance(task.grader, ScriptGrader)
        else grade.diagnostics.get("grader_timeout_seconds")
    )
    timed_out = grade.failure is GradingFailure.TIMEOUT
    cut = grade.diagnostics.get("exit_code") in (124, 137)
    if not (timed_out or cut):
        return
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or not isinstance(limit, (int, float))
        or isinstance(limit, bool)
        or not math.isfinite(limit)
        or limit <= 0
    ):
        raise Restart(f"Grader timeout/cut lacks reliable timing: elapsed={elapsed} limit={limit}; {grade.error or ''}")
    if elapsed < limit:
        raise Restart(
            f"Grader interrupted before declared limit: elapsed={elapsed}s limit={limit}s "
            f"exit_code={grade.diagnostics.get('exit_code')}; {grade.error or ''}"
        )
