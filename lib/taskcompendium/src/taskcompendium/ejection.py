# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt source tasks into terminal ejection without changing their visible problem."""

from tasktrove_verify.candidate import supports_candidate_mode
from tasktrove_verify.spec import Mode

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.grading import unsolvable
from taskcompendium.models import TaskSpec
from taskcompendium.submission import (
    EJECT_CALL_NAME,
    FinalAction,
    SubmissionConvention,
    eject_button,
    submission_compatibility,
)


def enable_ejection(specification: TaskSpec) -> TaskSpec:
    """Advertise the reserved terminal action independently of solvability."""
    tool = eject_button()
    existing = tuple(function for function in specification.final_tools if function.name == EJECT_CALL_NAME)
    if existing and existing != (tool,):
        raise ValueError("Source function collides with the reserved eject_button action")
    if existing:
        return specification
    return TaskSpec.model_validate({**specification.model_dump(), "final_tools": (*specification.final_tools, tool)})


def mark_unsolvable(
    specification: TaskSpec, normal_submission: SubmissionConvention
) -> tuple[TaskSpec, SubmissionConvention]:
    """Label a confirmed broken source task privately, retaining its presentation.

    The caller must establish that the problem cannot be solved. Unsupported
    runtime requirements and data-read failures do not establish unsolvability.
    """
    unsupported = unsupported_direct_chat_features(specification)
    if unsupported:
        raise NotImplementedError(f"Direct chat cannot satisfy requirements: {', '.join(unsupported)}")
    if not supports_candidate_mode(specification.verifier.kind) or specification.verifier.kind == Mode.STRUCTURED_EXACT:
        raise NotImplementedError(f"Direct chat cannot submit to verifier: {specification.verifier.kind!r}")
    compatibility = submission_compatibility(specification, normal_submission)
    if not compatibility.compatible:
        raise ValueError(f"Normal submission is incompatible: {'; '.join(compatibility.reasons)}")
    enabled = enable_ejection(specification)
    convention = (
        normal_submission
        if isinstance(normal_submission, FinalAction)
        else FinalAction(id=normal_submission.id, normal_submission=normal_submission)
    )
    broken = TaskSpec.model_validate({**enabled.model_dump(), "verifier": unsolvable()})
    return broken, convention
