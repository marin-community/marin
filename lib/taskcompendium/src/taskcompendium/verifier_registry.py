# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned verifier kinds without import-order-dependent registration."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ValidationError

from taskcompendium.grading import ExactAnswerVerifier, GradeResult, GradingAttempt, NumericAnswerVerifier, Verifier
from taskcompendium.models import ConversationTrace, TaskSpec, VerifierKind, VerifierSpec
from taskcompendium.submission import SubmissionConvention
from taskcompendium.verifiers.arc_injection import ArcGridVerifier, ArcTransformVerifier, IndirectInjectionVerifier
from taskcompendium.verifiers.atlas_answers import AbstentionAnswersVerifier, MathAnswerVerifier
from taskcompendium.verifiers.constraints import IfevalVerifier, JsonSchemaVerifier
from taskcompendium.verifiers.executable import TaskTroveExecutableVerifier
from taskcompendium.verifiers.multiple_choice import MultipleChoiceVerifier
from taskcompendium.verifiers.predicted_action import PredictedActionVerifier
from taskcompendium.verifiers.reasoning import PuzzleAnswerVerifier, ReasoningGymVerifier
from taskcompendium.verifiers.reference_answers import ReferenceAnswersVerifier
from taskcompendium.verifiers.runtime import CalendarStateVerifier, CaptureOutputVerifier
from taskcompendium.verifiers.schedule import ScheduleAnswerVerifier

VERIFIERS: Mapping[VerifierKind, type[Verifier]] = MappingProxyType(
    {
        VerifierKind.EXACT_ANSWER: ExactAnswerVerifier,
        VerifierKind.PREDICTED_ACTION: PredictedActionVerifier,
        VerifierKind.NUMERIC_ANSWER: NumericAnswerVerifier,
        VerifierKind.MCQ_ANSWER: MultipleChoiceVerifier,
        VerifierKind.CAPTURE_OUTPUT: CaptureOutputVerifier,
        VerifierKind.CALENDAR_STATE: CalendarStateVerifier,
        VerifierKind.IFEVAL: IfevalVerifier,
        VerifierKind.JSON_SCHEMA: JsonSchemaVerifier,
        VerifierKind.TASKTROVE_EXECUTABLE: TaskTroveExecutableVerifier,
        VerifierKind.REASONING_GYM: ReasoningGymVerifier,
        VerifierKind.PUZZLE_ANSWER: PuzzleAnswerVerifier,
        VerifierKind.SCHEDULE_ANSWER: ScheduleAnswerVerifier,
        VerifierKind.REFERENCE_ANSWERS: ReferenceAnswersVerifier,
        VerifierKind.MATH_ANSWER: MathAnswerVerifier,
        VerifierKind.ABSTENTION_ANSWERS: AbstentionAnswersVerifier,
        VerifierKind.ARC_GRID: ArcGridVerifier,
        VerifierKind.ARC_TRANSFORM: ArcTransformVerifier,
        VerifierKind.INDIRECT_INJECTION: IndirectInjectionVerifier,
    }
)


def resolve_verifier(specification: VerifierSpec) -> Verifier:
    verifier_type = VERIFIERS.get(specification.kind)
    if verifier_type is None:
        raise ValueError(f"Unknown verifier kind: {specification.kind!r}")
    try:
        return verifier_type.model_validate_json(specification.parameters_json)
    except ValidationError as error:
        raise ValueError(f"Invalid {specification.kind.value!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, conversation: ConversationTrace, environment: object
) -> GradeResult:
    verifier = resolve_verifier(specification.verifier)
    return verifier.grade(GradingAttempt(convention, conversation.events, environment))
