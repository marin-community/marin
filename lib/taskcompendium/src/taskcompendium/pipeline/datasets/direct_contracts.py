# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build tasks with explicit private evaluator contracts and unbound readiness."""

from pathlib import Path

from pydantic import JsonValue

from taskcompendium.grader import script_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.pipeline.models import RawRow

UNBOUND_SCRIPT = Path(__file__).with_name("unbound_grader.py").read_bytes()


def source_contract_package(
    evaluator: str,
    source_revision: str,
    contract: dict[str, JsonValue],
    runtime_requirements: tuple[str, ...],
) -> VerifierSpec:
    data = {
        "evaluator": evaluator,
        "source_revision": source_revision,
        "contract": contract,
        "runtime_requirements": runtime_requirements,
    }
    return script_package(UNBOUND_SCRIPT, data)


def contract_task(
    row: RawRow,
    events: tuple[TextMessage, ...],
    evaluator: str,
    contract: dict[str, JsonValue],
    runtime_requirements: tuple[str, ...],
) -> TaskSpec:
    """Keep grader inputs private while preserving the source public conversation."""
    package = source_contract_package(evaluator, row.source.revision, contract, runtime_requirements)
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=events),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=package,
    )
