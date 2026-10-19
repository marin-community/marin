# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build checks that a built task's verifier is the kind its proposal declared.

``built_verification`` reads the kind off the TaskSpec's grader: an in-process verifyit mode is
answer-in-box, a verifyit ``JudgeSpec`` is an LLM judge, the TaskSpec form of a ``DecisionGrader``
(``decision.task_package``) is a decision model, and a script grader or any other verifyit mode is
code. ``run_build`` refuses a draft ``verification_problem`` rejects, so the author fixes it in the
same session.

``unbuildable_problem`` refuses a declared kind no grader can implement yet before the program runs:
a composite, because a TaskSpec holds one grader and there is no composite grader type.
"""

from taskcompendium.models import ScriptGrader, TaskSpec, VerifyitGrader, verifyit_spec
from verifyit.candidate import IN_PROCESS_MODES
from verifyit.spec import JudgeSpec, mode_of

from taskforge.spec.decision import task_decision_grader
from taskforge.stage10_proposal.model import Verification

UNBUILDABLE: dict[Verification, str] = {
    Verification.COMPOSITE: "a TaskSpec holds one grader and no composite grader type exists yet",
}
"""Declared kinds no build can implement, with the reason."""


def built_verification(task: TaskSpec) -> Verification | None:
    """The verification kind ``task``'s grader implements, or ``None`` for a grader no kind names."""
    grader = task.grader
    if task_decision_grader(task) is not None:
        return Verification.DECISION_MODEL
    if isinstance(grader, ScriptGrader):
        return Verification.CODE
    if not isinstance(grader, VerifyitGrader):
        return None
    spec = verifyit_spec(grader)
    if isinstance(spec, JudgeSpec):
        return Verification.LLM_AS_JUDGE
    if mode_of(spec) in IN_PROCESS_MODES:
        return Verification.ANSWER_IN_BOX
    return Verification.CODE


def unbuildable_problem(declared: Verification) -> str | None:
    """Why no build can implement the ``declared`` kind, or ``None`` when one can."""
    reason = UNBUILDABLE.get(declared)
    return None if reason is None else f"the proposal declares {declared} verification, which cannot be built: {reason}"


def verification_problem(declared: Verification, task: TaskSpec) -> str | None:
    """Why the built verifier's kind contradicts the declared one, or ``None`` when it implements it."""
    built = built_verification(task)
    if built is declared:
        return None
    return (
        f"the proposal declares {declared} verification but the built grader is "
        f"{task.grader.kind if built is None else built}: build the grader of the declared kind"
    )
