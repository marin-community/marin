# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply TaskTrove-style positive and negative controls to semantic graders."""

import json

from verifyit.grade import negative_candidate
from verifyit.modes.extract import extract_boxed
from verifyit.spec import NumericSpec

from taskcompendium.grading import ExactAnswerVerifier, GradingAttempt, NumericAnswerVerifier, Outcome, Verifier
from taskcompendium.models import AssistantToolCalls, ConversationToolCall, TaskSpec, TextMessage
from taskcompendium.pipeline.models import CheckResult, CheckStatus, GraderReadiness
from taskcompendium.submission import AnswerFormat, SubmissionConvention
from taskcompendium.verifier_registry import resolve_verifier
from taskcompendium.verifiers.multiple_choice import MultipleChoiceVerifier
from taskcompendium.verifiers.predicted_action import PredictedActionVerifier
from taskcompendium.verifiers.source_contract import SourceContractVerifier

PLAIN = SubmissionConvention(id="pipeline-plain", answer_format=AnswerFormat.PLAIN)


def grader_readiness(checks: list[CheckResult]) -> GraderReadiness:
    """Summarize control coverage independently of the static quality decision."""
    if any(check.status == CheckStatus.FAIL for check in checks):
        return GraderReadiness.FAILED
    if not checks or any(check.status != CheckStatus.PASS for check in checks):
        return GraderReadiness.UNVERIFIED
    return GraderReadiness.READY


def verify_task(task: TaskSpec) -> list[CheckResult]:
    """Check the real answer grader, recording unsupported runtime requirements.

    These controls establish grader behavior, not correctness of the source key.
    They do not run a container or solve the task.
    """
    try:
        verifier = resolve_verifier(task.verifier)
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
    if task.environment_requirements.capabilities or task.environment_requirements.action_interfaces:
        return [CheckResult(check="runtime", status=CheckStatus.UNSUPPORTED, detail="An isolated runtime is required")]

    if isinstance(verifier, PredictedActionVerifier):
        return _action_checks(task, verifier)
    if isinstance(verifier, NumericAnswerVerifier):
        contract = NumericSpec(**json.loads(task.verifier.parameters_json))
        perturbed = negative_candidate(contract)
        assert perturbed is not None
        negative = extract_boxed(perturbed) or perturbed
        positive = repr(verifier.expected)
    elif isinstance(verifier, MultipleChoiceVerifier):
        positive = verifier.expected
        negative = "B" if positive != "B" else "A"
    elif isinstance(verifier, ExactAnswerVerifier):
        positive = verifier.expected
        negative = f"{positive}\n__incorrect_answer__"
    else:
        return [CheckResult(check="grader_controls", status=CheckStatus.UNSUPPORTED, detail=task.verifier.kind.value)]

    return _answer_checks(
        task, verifier, (("empty", "", 0.0), ("reference", positive, 1.0), ("perturbed", negative, 0.0))
    )


def _answer_checks(
    task: TaskSpec, verifier: Verifier, controls: tuple[tuple[str, str, float], ...]
) -> list[CheckResult]:
    results = []
    for name, answer, expected in controls:
        attempt = GradingAttempt(PLAIN, (*task.context.events, TextMessage(role="assistant", content=answer)), None)
        result = verifier.grade(attempt)
        # MCQA empty submissions fail extraction, which is a valid negative control.
        passed = (
            result.reward == expected
            if result.status == Outcome.GRADED
            else (expected == 0.0 and result.status == Outcome.EXTRACTION_ERROR)
        )
        status = CheckStatus.PASS if passed else CheckStatus.FAIL
        if result.status == Outcome.INFRA_ERROR:
            status = CheckStatus.INFRA_ERROR
        results.append(CheckResult(check=name, status=status, detail=f"{result.status}: reward={result.reward}"))
    return results


def verify_witness(task: TaskSpec, witness: str, negative: str) -> list[CheckResult]:
    """Check a separately supplied feasible answer and two failing submissions.

    A witness can prove formal feasibility without establishing content quality.
    It stays outside the model-visible task and never becomes its reference key.
    """
    verifier = resolve_verifier(task.verifier)
    return _answer_checks(task, verifier, (("empty", "", 0.0), ("witness", witness, 1.0), ("negative", negative, 0.0)))


def _action_checks(task: TaskSpec, verifier: PredictedActionVerifier) -> list[CheckResult]:
    convention = SubmissionConvention(id="pipeline-action", answer_format=AnswerFormat.FINAL_ACTION)
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
        grade = verifier.grade(GradingAttempt(convention, (*task.context.events, response), None))
        results.append(
            CheckResult(
                check=name,
                status=CheckStatus.PASS if grade.reward == expected else CheckStatus.FAIL,
                detail=f"{grade.status}: reward={grade.reward}",
            )
        )
    return results
