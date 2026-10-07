# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply positive and negative controls to task graders."""

from verifyit.grade import negative_candidate
from verifyit.modes.extract import extract_boxed
from verifyit.spec import ExactSpec, McqSpec, NumericSpec, PredictedActionSpec

from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import CheckResult, CheckStatus, GraderReadiness
from taskcompendium.runtime.task_grading import grade_task, resolve_verifier
from taskcompendium.submission import FinalAction, PlainText

PLAIN = PlainText(id="pipeline-plain")


def grader_readiness(checks: list[CheckResult]) -> GraderReadiness:
    """Summarize control coverage independently of the static quality decision."""
    if any(check.status == CheckStatus.FAIL for check in checks):
        return GraderReadiness.FAILED
    if not checks or any(check.status != CheckStatus.PASS for check in checks):
        return GraderReadiness.UNVERIFIED
    return GraderReadiness.READY


def verify_task(task: TaskSpec) -> list[CheckResult]:
    """Check the answer grader and record unsupported runtime requirements."""
    if task.verifier.environment_requirements != EnvironmentRequirements():
        return [
            CheckResult(
                check="runtime", status=CheckStatus.UNSUPPORTED, detail="A private grading environment is required"
            )
        ]
    try:
        verifier = resolve_verifier(task.verifier)
    except ValueError as error:
        return [CheckResult(check="verifier_contract", status=CheckStatus.FAIL, detail=str(error))]

    if task.environment_requirements != EnvironmentRequirements():
        return [CheckResult(check="runtime", status=CheckStatus.UNSUPPORTED, detail="An isolated runtime is required")]

    if isinstance(verifier, PredictedActionSpec):
        return _action_checks(task, verifier)
    if isinstance(verifier, NumericSpec):
        perturbed = negative_candidate(verifier)
        assert perturbed is not None
        negative = extract_boxed(perturbed) or perturbed
        positive = verifier.expected
    elif isinstance(verifier, McqSpec):
        positive = verifier.expected
        negative = "B" if positive != "B" else "A"
    elif isinstance(verifier, ExactSpec):
        positive = verifier.expected[0]
        negative = f"{positive}\n__incorrect_answer__"
    else:
        return [CheckResult(check="grader_controls", status=CheckStatus.UNSUPPORTED, detail=task.verifier.kind)]

    return answer_checks(task, (("empty", "", 0.0), ("reference", positive, 1.0), ("perturbed", negative, 0.0)))


def _grade_control(task: TaskSpec, answer: str) -> GradeResult:
    events = (*task.context.events, TextMessage(role="assistant", content=answer))
    return grade_task(task, PLAIN, ConversationTrace(events=events))


def answer_checks(task: TaskSpec, controls: tuple[tuple[str, str, float], ...]) -> list[CheckResult]:
    """Grade plain-response controls while retaining unavailable graders as unsupported."""
    results = []
    for name, answer, expected in controls:
        result = _grade_control(task, answer)
        passed = (
            result.reward == expected
            if result.status == Outcome.GRADED
            else (expected == 0.0 and result.status == Outcome.SUBMISSION_FAILURE)
        )
        status = CheckStatus.PASS if passed else CheckStatus.FAIL
        if result.status == Outcome.INFRA_ERROR:
            status = CheckStatus.UNSUPPORTED
        results.append(CheckResult(check=name, status=status, detail=f"{result.status}: reward={result.reward}"))
    return results


def verify_witness(task: TaskSpec, witness: str, negative: str) -> list[CheckResult]:
    """Check a separately supplied feasible answer and two failing submissions."""
    return answer_checks(task, (("empty", "", 0.0), ("witness", witness, 1.0), ("negative", negative, 0.0)))


def _action_checks(task: TaskSpec, verifier: PredictedActionSpec) -> list[CheckResult]:
    convention = FinalAction(id="pipeline-action", require_call=True, max_calls=len(verifier.expected_calls))
    calls = tuple(
        ConversationToolCall(call_id=f"control-{index}", name=call.name, arguments=call.arguments)
        for index, call in enumerate(verifier.expected_calls)
    )
    reference = AssistantToolCalls(calls=calls)
    wrong = AssistantToolCalls(calls=(calls[0].model_copy(update={"name": "__wrong_tool__"}), *calls[1:]))
    results = []
    for name, response, expected in (
        ("empty", TextMessage(role="assistant", content=""), 0.0),
        ("reference", reference, 1.0),
        ("perturbed", wrong, 0.0),
    ):
        grade = grade_task(task, convention, ConversationTrace(events=(*task.context.events, response)))
        passed = grade.reward == expected or (expected == 0.0 and grade.status == Outcome.SUBMISSION_FAILURE)
        results.append(
            CheckResult(
                check=name,
                status=CheckStatus.PASS if passed else CheckStatus.FAIL,
                detail=f"{grade.status}: reward={grade.reward}",
            )
        )
    return results
