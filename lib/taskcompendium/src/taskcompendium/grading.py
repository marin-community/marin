# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade an attempt in process with the task's verifyit mode.

The task's answer format extracts one typed submission; this module adapts it to a verifyit
candidate and maps the verifyit reward to a ``GradeResult``. verifyit owns specification
validation and scoring.
"""

import json
import math

from openai import APIError
from verifyit.candidate import Candidate, grade_candidate
from verifyit.grade import GradingInfraError, InvalidTask, Reward, Status, empty_output_policy
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import (
    EmptyOutputPolicy,
    ExactSpec,
    JudgeSpec,
    NumericSpec,
    PredictedActionSpec,
    Spec,
    StructuredExactSpec,
)
from verifyit.spec import FunctionCall as CandidateCall

from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    ActionSubmission,
    AnswerType,
    AssistantToolCalls,
    GradingAttempt,
    JsonSubmission,
    PlainText,
    StateSubmission,
    Submission,
    SubmissionFailure,
    TaskSpec,
    TextMessage,
    TextSubmission,
    VerifyitGrader,
    verifyit_spec,
)
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import native_action_text, require_submission_compatibility


def grade_result(verifier: Spec, verdict: Reward) -> GradeResult:
    """Normalize verifyit outcomes independently of where the verifier ran."""
    if (
        not isinstance(verdict.status, Status)
        or not isinstance(verdict.detail, dict)
        or isinstance(verdict.reward, bool)
        or not isinstance(verdict.reward, int | float)
        or not 0 <= verdict.reward <= 1
        or not math.isfinite(verdict.reward)
        or (verdict.status != Status.SCORED and verdict.reward != 0)
    ):
        raise ValueError("Invalid verifier verdict")
    if verdict.status == Status.SCORED:
        if isinstance(verifier, NumericSpec) and verdict.detail.get("reason") == "invalid_numeric_candidate":
            if verdict.reward != 0:
                raise ValueError("Invalid numeric submissions cannot receive a positive reward")
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, verdict.detail.get("error"), verdict.detail)
        return GradeResult(Outcome.GRADED, float(verdict.reward), verdict.detail.get("error"), verdict.detail)
    status = Outcome.INVALID_TASK if verdict.status == Status.INVALID_TASK else Outcome.INFRA_ERROR
    return GradeResult(status, None, verdict.detail.get("error"), verdict.detail)


def parse_grade_result(verifier: Spec, data: bytes) -> GradeResult:
    """Decode a verdict file written by the verifyit command."""
    try:
        verdict = json.loads(data)
        if not isinstance(verdict, dict):
            raise ValueError("Verifier verdict must be an object")
        return grade_result(verifier, Reward(verdict["reward"], Status(verdict["status"]), verdict["detail"]))
    except (KeyError, TypeError, ValueError):
        return GradeResult(Outcome.INFRA_ERROR, None, "Invalid verifier verdict", failure=GradingFailure.INVALID_REWARD)


def answer_submission(task: TaskSpec, attempt: GradingAttempt) -> Submission:
    """The submission the task's answer type and format select from an attempt.

    Raises ``SubmissionFailure`` when the attempt carries no valid submission.
    """
    if task.answer_type in CONVERSATION_ANSWERS:
        return task.answer_format.extract(attempt)
    if task.answer_type == AnswerType.STATE:
        if attempt.state is None:
            raise SubmissionFailure("State submission requires captured state")
        return attempt.state
    raise TypeError(f"A {task.answer_type} answer is not extracted from the attempt")


def _candidate(verifier: Spec, submission: Submission) -> Candidate:
    match verifier, submission:
        case StructuredExactSpec(), JsonSubmission(value=value) | StateSubmission(value=value):
            return value
        case PredictedActionSpec(), ActionSubmission(message=final):
            if not isinstance(final, AssistantToolCalls):
                return ()
            return tuple(CandidateCall(call.name, call.object_arguments()) for call in final.calls)
        case ExactSpec(), JsonSubmission(value=value) | StateSubmission(value=value):
            if not isinstance(value, str):
                raise SubmissionFailure("Text verifier requires a string JSON value")
            return value
        case StructuredExactSpec() | PredictedActionSpec(), _:
            raise TypeError(f"{type(verifier).__name__} cannot grade a {type(submission).__name__}")
        case _, TextSubmission(value=value):
            return value
    raise TypeError(f"{type(verifier).__name__} cannot grade a {type(submission).__name__}")


def _gradeable_empty_text(task: TaskSpec, attempt: GradingAttempt, verifier: Spec) -> bool:
    """Whether a blank final text is a candidate under the verifier's explicit policy."""
    if not isinstance(task.answer_format, PlainText):
        return False
    try:
        policy = empty_output_policy(verifier)
    except InvalidTask:
        return False
    if policy is not EmptyOutputPolicy.GRADE:
        return False
    final = attempt.conversation.events[-1]
    return isinstance(final, TextMessage) and final.role == "assistant" and not final.content.strip()


def grade_answer(task: TaskSpec, attempt: GradingAttempt) -> GradeResult:
    """Extract the task's submission and score it with its in-process verifyit mode."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader) or grader.environment is not None:
        raise TypeError(f"In-process grading requires a verifyit grader without an environment, not {grader.kind}")
    verifier = verifyit_spec(grader)
    if task.answer_type in CONVERSATION_ANSWERS:
        require_submission_compatibility(task)
    try:
        candidate = _candidate(verifier, answer_submission(task, attempt))
    except SubmissionFailure as error:
        if not _gradeable_empty_text(task, attempt, verifier):
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
        candidate = ""
    resources = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    return grade_result(verifier, grade_candidate(verifier, candidate, resources))


def grade_remote_judge_answer(task: TaskSpec, attempt: GradingAttempt, connection: JudgeConnection) -> GradeResult:
    """Grade one judge submission through an explicit endpoint outside its network-denied sandbox."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader):
        raise TypeError(f"Remote judge grading requires a verifyit grader, not {grader.kind}")
    verifier = verifyit_spec(grader)
    if not isinstance(verifier, JudgeSpec):
        raise TypeError(f"Remote judge grading requires a judge verifier, not {grader.mode}")
    require_submission_compatibility(task)
    try:
        submission = answer_submission(task, attempt)
    except SubmissionFailure as error:
        if not _gradeable_empty_text(task, attempt, verifier):
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
        candidate = ""
    else:
        if isinstance(submission, TextSubmission):
            candidate = submission.value
        elif isinstance(submission, ActionSubmission):
            if isinstance(submission.message, TextMessage) and not submission.message.content.strip():
                return GradeResult(Outcome.GRADED, 0.0, detail={"reason": "empty_candidate"})
            candidate = native_action_text(submission.message)
        else:
            raise TypeError(f"A judge cannot grade a {type(submission).__name__}")
    if not candidate.strip() and verifier.empty_output is EmptyOutputPolicy.ZERO:
        return GradeResult(Outcome.GRADED, 0.0, detail={"reason": "empty_candidate"})
    resources = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    if verifier.context and verifier.context not in resources:
        return GradeResult(Outcome.INVALID_TASK, None, f"judge context {verifier.context!r} is not in task resources")
    context = resources.get(verifier.context, b"").decode(errors="replace")
    try:
        verdict = grade_judge_candidate(verifier, candidate, connection=connection, runtime=None, context=context)
    except InvalidTask as error:
        return GradeResult(Outcome.INVALID_TASK, None, str(error))
    except (APIError, GradingInfraError, RuntimeError, OSError) as error:
        return GradeResult(Outcome.INFRA_ERROR, None, str(error))
    return grade_result(verifier, verdict)
