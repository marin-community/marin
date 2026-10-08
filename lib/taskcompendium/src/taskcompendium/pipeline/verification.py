# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply positive and negative controls to task graders."""

import json

from verifyit.grade import negative_candidate, positive_candidate
from verifyit.modes.extract import extract_boxed
from verifyit.spec import ExactSpec, JsonSchemaSpec, McqSpec, NumericSpec, PredictedActionSpec

from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    ANSWER_FIELD,
    AnswerCall,
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    JsonAnswer,
    NoGrader,
    SessionGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    verifyit_spec,
)
from taskcompendium.pipeline.models import CheckResult, CheckStatus, GraderReadiness
from taskcompendium.runtime.task_grading import grade_task


def control_result(grade: GradeResult, name: str, expected_reward: float) -> CheckResult:
    """Require the expected reward, counting a rejected submission as zero, while retaining runtime errors."""
    if grade.status == Outcome.UNAVAILABLE:
        status = CheckStatus.UNSUPPORTED
    elif grade.status == Outcome.INFRA_ERROR:
        status = CheckStatus.INFRA_ERROR
    elif (grade.status == Outcome.GRADED and grade.reward == expected_reward) or (
        grade.status == Outcome.SUBMISSION_FAILURE and expected_reward == 0.0
    ):
        status = CheckStatus.PASS
    else:
        status = CheckStatus.FAIL
    detail = f"{grade.status}: reward={grade.reward}; expected={expected_reward}"
    if grade.error is not None:
        detail += f"; error={grade.error}"
    if grade.detail and status != CheckStatus.PASS:
        detail += f"; detail={grade.detail}"
    return CheckResult(check=name, status=status, detail=detail)


def grader_readiness(checks: list[CheckResult]) -> GraderReadiness:
    """Summarize control coverage independently of the static quality decision."""
    if any(check.status == CheckStatus.FAIL for check in checks):
        return GraderReadiness.FAILED
    if not checks or any(check.status != CheckStatus.PASS for check in checks):
        return GraderReadiness.UNVERIFIED
    return GraderReadiness.READY


def verify_task(task: TaskSpec) -> list[CheckResult]:
    """Check the answer grader and record unsupported runtime requirements."""
    grader = task.grader
    if isinstance(grader, NoGrader):
        return [CheckResult(check="source_evaluator", status=CheckStatus.UNSUPPORTED, detail=grader.reason)]
    if isinstance(grader, SessionGrader):
        return [
            CheckResult(
                check="runtime", status=CheckStatus.UNSUPPORTED, detail="A session grader runs in its rollout session"
            )
        ]
    if not isinstance(grader, VerifyitGrader) or grader.environment is not None:
        return [CheckResult(check="runtime", status=CheckStatus.UNSUPPORTED, detail="A grading environment is required")]
    try:
        verifier = verifyit_spec(grader)
    except ValueError as error:
        return [CheckResult(check="verifier_contract", status=CheckStatus.FAIL, detail=str(error))]

    if task.environment_requirements != EnvironmentRequirements():
        return [CheckResult(check="runtime", status=CheckStatus.UNSUPPORTED, detail="An isolated runtime is required")]

    if isinstance(verifier, PredictedActionSpec):
        return _action_checks(task, verifier)
    if isinstance(verifier, JsonSchemaSpec):
        return [
            *answer_checks(task, (("empty", "", 0.0), ("malformed", "[}", 0.0))),
            CheckResult(check="reference", status=CheckStatus.SKIPPED, detail="No schema-valid reference supplied"),
        ]
    if isinstance(verifier, NumericSpec):
        perturbed = negative_candidate(verifier)
        assert perturbed is not None
        negative = extract_boxed(perturbed) or perturbed
        positive = verifier.expected
    elif isinstance(verifier, McqSpec):
        positive = verifier.expected
        negative = "B" if positive != "B" else "A"
    elif isinstance(verifier, ExactSpec):
        positive = positive_candidate(verifier)
        assert positive is not None
        negative = f"{positive}\n__incorrect_answer__"
    else:
        return [CheckResult(check="grader_controls", status=CheckStatus.UNSUPPORTED, detail=grader.mode)]

    return answer_checks(task, (("empty", "", 0.0), ("reference", positive, 1.0), ("perturbed", negative, 0.0)))


def answer_event(task: TaskSpec, answer: str) -> TextMessage | AssistantToolCalls:
    """Present a control answer in the task's answer format."""
    if isinstance(task.answer_format, AnswerCall):
        call = ConversationToolCall(call_id="control", name=ANSWER_CALL_NAME, arguments={ANSWER_FIELD: answer})
        return AssistantToolCalls(calls=(call,))
    if isinstance(task.answer_format, JsonAnswer):
        return TextMessage(role="assistant", content=json.dumps({ANSWER_FIELD: answer}))
    return TextMessage(role="assistant", content=answer)


def _grade_control(task: TaskSpec, answer: str) -> GradeResult:
    events = (*task.context.events, answer_event(task, answer))
    return grade_task(task, GradingAttempt(ConversationTrace(events=events)))


def answer_checks(task: TaskSpec, controls: tuple[tuple[str, str, float], ...]) -> list[CheckResult]:
    """Grade control answers while retaining unavailable graders as unsupported."""
    results = []
    for name, answer, expected in controls:
        result = _grade_control(task, answer)
        passed = (
            result.reward == expected
            if result.status == Outcome.GRADED
            else (expected == 0.0 and result.status == Outcome.SUBMISSION_FAILURE)
        )
        status = CheckStatus.PASS if passed else CheckStatus.FAIL
        if result.status in (Outcome.UNAVAILABLE, Outcome.INFRA_ERROR):
            status = CheckStatus.UNSUPPORTED
        results.append(CheckResult(check=name, status=status, detail=f"{result.status}: reward={result.reward}"))
    return results


def verify_witness(task: TaskSpec, witness: str, negative: str) -> list[CheckResult]:
    """Check a separately supplied feasible answer and two failing submissions."""
    return answer_checks(task, (("empty", "", 0.0), ("witness", witness, 1.0), ("negative", negative, 0.0)))


def _action_checks(task: TaskSpec, verifier: PredictedActionSpec) -> list[CheckResult]:
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
        grade = grade_task(task, GradingAttempt(ConversationTrace(events=(*task.context.events, response))))
        passed = grade.status == Outcome.GRADED and grade.reward == expected
        passed = passed or (expected == 0.0 and grade.status == Outcome.SUBMISSION_FAILURE)
        results.append(
            CheckResult(
                check=name,
                status=(
                    CheckStatus.UNSUPPORTED
                    if grade.status == Outcome.UNAVAILABLE
                    else CheckStatus.PASS if passed else CheckStatus.FAIL
                ),
                detail=f"{grade.status}: reward={grade.reward}",
            )
        )
    return results
