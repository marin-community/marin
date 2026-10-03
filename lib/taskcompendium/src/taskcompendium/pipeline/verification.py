# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply positive and negative controls to task graders."""

from verifyit.candidate import CandidateSpec, supports_candidate_mode
from verifyit.grade import negative_candidate
from verifyit.modes.extract import extract_boxed
from verifyit.spec import ExactSpec, McqSpec, NumericSpec, PredictedActionSpec

from taskcompendium.grading import GradeResult, Outcome, grade_answer, resolve_verifier
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import CheckResult, CheckStatus, GraderReadiness
from taskcompendium.submission import AnswerFormat, FinalAction, SubmissionConvention
from taskcompendium.verifiers.base import GradingAttempt, Verifier
from taskcompendium.verifiers.dispatch import resolve_custom_verifier
from taskcompendium.verifiers.source_contract import SourceContractVerifier

PLAIN = SubmissionConvention(id="pipeline-plain", answer_format=AnswerFormat.PLAIN)


def grader_readiness(checks: list[CheckResult]) -> GraderReadiness:
    """Summarize control coverage independently of the static quality decision."""
    if any(check.status == CheckStatus.FAIL for check in checks):
        return GraderReadiness.FAILED
    if not checks or any(check.status != CheckStatus.PASS for check in checks):
        return GraderReadiness.UNVERIFIED
    return GraderReadiness.READY


def _control_verifier(task: TaskSpec) -> CandidateSpec | Verifier:
    if supports_candidate_mode(task.verifier.kind):
        return resolve_verifier(task.verifier)
    return resolve_custom_verifier(task.verifier)


def verify_task(task: TaskSpec) -> list[CheckResult]:
    """Check the answer grader and record unsupported runtime requirements."""
    if task.verifier.environment_requirements != EnvironmentRequirements():
        return [
            CheckResult(
                check="runtime", status=CheckStatus.UNSUPPORTED, detail="A private grading environment is required"
            )
        ]
    try:
        verifier = _control_verifier(task)
    except ValueError as error:
        return [CheckResult(check="verifier_contract", status=CheckStatus.FAIL, detail=str(error))]

    if isinstance(verifier, SourceContractVerifier):
        return [
            CheckResult(
                check="source_evaluator",
                status=CheckStatus.UNSUPPORTED,
                detail=f"{verifier.evaluator}@{verifier.source_revision} is unbound; requires "
                + "; ".join(verifier.runtime_requirements),
            )
        ]
    if task.environment_requirements != EnvironmentRequirements():
        return [CheckResult(check="runtime", status=CheckStatus.UNSUPPORTED, detail="An isolated runtime is required")]

    if isinstance(verifier, PredictedActionSpec):
        return _action_checks(task, verifier)
    if isinstance(verifier, NumericSpec):
        perturbed = negative_candidate(verifier)
        assert perturbed is not None
        negative = extract_boxed(perturbed) or perturbed
        positive = repr(verifier.expected)
    elif isinstance(verifier, McqSpec):
        positive = verifier.expected
        negative = "B" if positive != "B" else "A"
    elif isinstance(verifier, ExactSpec):
        positive = verifier.expected[0]
        negative = f"{positive}\n__incorrect_answer__"
    else:
        return [CheckResult(check="grader_controls", status=CheckStatus.UNSUPPORTED, detail=task.verifier.kind)]

    return answer_checks(
        task, verifier, (("empty", "", 0.0), ("reference", positive, 1.0), ("perturbed", negative, 0.0))
    )


def _grade_control(task: TaskSpec, verifier: CandidateSpec | Verifier, answer: str) -> GradeResult:
    events = (*task.context.events, TextMessage(role="assistant", content=answer))
    if isinstance(verifier, Verifier):
        return verifier.grade(GradingAttempt(PLAIN, events, None))
    return grade_answer(task, PLAIN, ConversationTrace(events=events))


def answer_checks(
    task: TaskSpec, verifier: CandidateSpec | Verifier, controls: tuple[tuple[str, str, float], ...]
) -> list[CheckResult]:
    """Grade plain-response controls while retaining unavailable graders as unsupported."""
    results = []
    for name, answer, expected in controls:
        result = _grade_control(task, verifier, answer)
        passed = (
            result.reward == expected
            if result.status == Outcome.GRADED
            else (expected == 0.0 and result.status == Outcome.EXTRACTION_ERROR)
        )
        status = CheckStatus.PASS if passed else CheckStatus.FAIL
        if result.status == Outcome.INFRA_ERROR:
            status = CheckStatus.UNSUPPORTED
        results.append(CheckResult(check=name, status=status, detail=f"{result.status}: reward={result.reward}"))
    return results


def verify_witness(task: TaskSpec, witness: str, negative: str) -> list[CheckResult]:
    """Check a separately supplied feasible answer and two failing submissions."""
    verifier = _control_verifier(task)
    return answer_checks(task, verifier, (("empty", "", 0.0), ("witness", witness, 1.0), ("negative", negative, 0.0)))


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
        grade = grade_answer(task, convention, ConversationTrace(events=(*task.context.events, response)))
        passed = grade.reward == expected or (expected == 0.0 and grade.status == Outcome.EXTRACTION_ERROR)
        results.append(
            CheckResult(
                check=name,
                status=CheckStatus.PASS if passed else CheckStatus.FAIL,
                detail=f"{grade.status}: reward={grade.reward}",
            )
        )
    return results
