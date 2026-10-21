# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt source tasks into terminal ejection without changing their visible problem."""

from verifyit.spec import UnsolvableSpec

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.grader import verifyit_package
from taskcompendium.models import FinalAction, TaskSpec, VerifyitGrader
from taskcompendium.submission import EJECT_CALL_NAME, eject_button, require_submission_compatibility


def enable_ejection(task: TaskSpec) -> TaskSpec:
    """Advertise the reserved terminal action independently of solvability."""
    tool = eject_button()
    existing = tuple(function for function in task.final_tools if function.name == EJECT_CALL_NAME)
    if any(function.name == EJECT_CALL_NAME for function in task.interaction_tools) or (
        existing and existing != (tool,)
    ):
        raise ValueError("Source function collides with the reserved eject_button action")
    if existing:
        return task
    return TaskSpec.model_validate({**dict(task), "final_tools": (*task.final_tools, tool)})


def mark_unsolvable(task: TaskSpec) -> TaskSpec:
    """Privately label a confirmed unsolvable task, retaining its normal presentation.

    Callers must establish unsolvability. Unsupported runtimes and missing evidence do not
    establish it; this helper supports only tasks that can already be graded in direct chat.
    """
    unsupported = unsupported_direct_chat_features(task)
    if unsupported:
        raise NotImplementedError(f"Direct chat cannot satisfy requirements: {', '.join(unsupported)}")
    if not isinstance(task.grader, VerifyitGrader):
        raise NotImplementedError("Unsolvable classification requires an in-process verifyit grader")
    require_submission_compatibility(task)
    enabled = enable_ejection(task)
    answer_format = task.answer_format
    if not isinstance(answer_format, FinalAction):
        answer_format = FinalAction(normal_submission=answer_format)
    return TaskSpec.model_validate(
        {**dict(enabled), "answer_format": answer_format, "grader": verifyit_package(UnsolvableSpec()).grader}
    )
