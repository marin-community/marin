# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build tasks whose source evaluator is recorded but not runnable here."""

from pydantic import JsonValue

from taskcompendium.grader import GraderPackage
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import RawRow


def source_contract_package(
    evaluator: str,
    source_revision: str,
    contract: dict[str, JsonValue],
    runtime_requirements: tuple[str, ...],
) -> GraderPackage:
    """Record a source evaluator that needs ``runtime_requirements`` this repository cannot provide."""
    reason = f"Source evaluator {evaluator} requires: {'; '.join(runtime_requirements)}"
    data: dict[str, JsonValue] = {
        "evaluator": evaluator,
        "source_revision": source_revision,
        "contract": contract,
        "runtime_requirements": list(runtime_requirements),
    }
    return GraderPackage(NoGrader(reason=reason, contract=data))


def contract_task(
    row: RawRow,
    events: tuple[TextMessage, ...],
    evaluator: str,
    contract: dict[str, JsonValue],
    runtime_requirements: tuple[str, ...],
) -> TaskSpec:
    """Preserve the source conversation and record its unavailable evaluator."""
    package = source_contract_package(evaluator, row.source.revision, contract, runtime_requirements)
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=events),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
    )
