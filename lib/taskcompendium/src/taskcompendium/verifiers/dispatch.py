# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve TaskCompendium-specific verifier contracts."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ValidationError

from taskcompendium.models import EnvironmentRequirements, VerifierSpec
from taskcompendium.verifiers.arc_injection import ArcGridVerifier, ArcTransformVerifier, IndirectInjectionVerifier
from taskcompendium.verifiers.atlas_answers import AbstentionAnswersVerifier, MathAnswerVerifier
from taskcompendium.verifiers.base import Verifier, VerifierKind
from taskcompendium.verifiers.constraints import IfevalVerifier, JsonSchemaVerifier
from taskcompendium.verifiers.executable import TaskTroveExecutableVerifier
from taskcompendium.verifiers.preference import PreferenceEvidenceVerifier
from taskcompendium.verifiers.reasoning import PuzzleAnswerVerifier, ReasoningGymVerifier
from taskcompendium.verifiers.reference_answers import ReferenceAnswersVerifier
from taskcompendium.verifiers.repository_patch import RepositoryPatchVerifier
from taskcompendium.verifiers.rubric_judge import RubricJudgeVerifier
from taskcompendium.verifiers.runtime import CalendarStateVerifier, CaptureOutputVerifier
from taskcompendium.verifiers.schedule import ScheduleAnswerVerifier
from taskcompendium.verifiers.source_contract import SourceContractVerifier
from taskcompendium.verifiers.structured_fields import NamedFieldsVerifier

CUSTOM_VERIFIERS: Mapping[str, type[Verifier]] = MappingProxyType(
    {
        VerifierKind.CAPTURE_OUTPUT: CaptureOutputVerifier,
        VerifierKind.CALENDAR_STATE: CalendarStateVerifier,
        VerifierKind.IFEVAL: IfevalVerifier,
        VerifierKind.JSON_SCHEMA: JsonSchemaVerifier,
        VerifierKind.STRUCTURED_FIELDS: NamedFieldsVerifier,
        VerifierKind.TASKTROVE_EXECUTABLE: TaskTroveExecutableVerifier,
        VerifierKind.REASONING_GYM: ReasoningGymVerifier,
        VerifierKind.PUZZLE_ANSWER: PuzzleAnswerVerifier,
        VerifierKind.SCHEDULE_ANSWER: ScheduleAnswerVerifier,
        VerifierKind.REFERENCE_ANSWERS: ReferenceAnswersVerifier,
        VerifierKind.RUBRIC_JUDGE: RubricJudgeVerifier,
        VerifierKind.REPOSITORY_PATCH: RepositoryPatchVerifier,
        VerifierKind.SOURCE_CONTRACT: SourceContractVerifier,
        VerifierKind.PREFERENCE_EVIDENCE: PreferenceEvidenceVerifier,
        VerifierKind.MATH_ANSWER: MathAnswerVerifier,
        VerifierKind.ABSTENTION_ANSWERS: AbstentionAnswersVerifier,
        VerifierKind.ARC_GRID: ArcGridVerifier,
        VerifierKind.ARC_TRANSFORM: ArcTransformVerifier,
        VerifierKind.INDIRECT_INJECTION: IndirectInjectionVerifier,
    }
)


def resolve_custom_verifier(specification: VerifierSpec) -> Verifier:
    """Validate a source-specific or runtime verifier selected by task conversion."""
    if specification.environment_requirements != EnvironmentRequirements():
        raise NotImplementedError("Custom verifier requires an unbound private environment")
    verifier_type = CUSTOM_VERIFIERS.get(specification.kind)
    if verifier_type is None:
        raise ValueError(f"Unknown custom verifier kind: {specification.kind!r}")
    try:
        return verifier_type.model_validate_json(specification.parameters_json)
    except ValidationError as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error
