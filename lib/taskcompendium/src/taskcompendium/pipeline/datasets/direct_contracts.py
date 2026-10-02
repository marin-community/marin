# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build tasks with explicit private evaluator contracts and unbound readiness."""

from pydantic import JsonValue

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.verifiers.source_contract import SourceContractVerifier


def contract_task(
    row: RawRow,
    events: tuple[TextMessage, ...],
    evaluator: str,
    contract: dict[str, JsonValue],
    runtime_requirements: tuple[str, ...],
) -> TaskSpec:
    """Keep grader inputs private while preserving the source public conversation."""
    verifier = SourceContractVerifier(
        evaluator=evaluator,
        source_revision=row.source.revision,
        contract=contract,
        runtime_requirements=runtime_requirements,
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=events),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.SOURCE_CONTRACT, parameters_json=verifier.model_dump_json()),
    )
