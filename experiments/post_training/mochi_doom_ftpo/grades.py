# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declare the campaign's full-credit correctness contract on native graders."""

from taskcompendium.models import ScriptGrader, StdoutReward, TaskSpec, VerifyitGrader


def full_credit_task(task: TaskSpec) -> TaskSpec:
    """Require a native score of at least one from the campaign's stdout checkers."""
    grader = task.grader
    if isinstance(grader, VerifyitGrader):
        return task
    if not isinstance(grader, ScriptGrader) or not isinstance(grader.reward, StdoutReward):
        raise ValueError(f"Unexpected campaign grader for {task.id}: {grader.kind}")
    return task.model_copy(update={"grader": grader.model_copy(update={"reward": StdoutReward(pass_at_least=1.0)})})
