# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert source rows to semantic tasks with private verifier inputs."""

from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue

from taskcompendium.chat import chat_input
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec, VerifierSpec


class ExternalVerifierSpec(BaseModel):
    """Private inputs for a verifier supplied by the execution application."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameters: dict[str, JsonValue]


def source_task(
    prompt: list[dict[str, Any]],
    extras: dict[str, Any],
    config: dict[str, Any],
    source: Source,
    *,
    environment: EnvironmentRequirements | None = None,
) -> TaskSpec:
    """Preserve source semantics without selecting a session or machine backend."""
    verifier = ExternalVerifierSpec(parameters={"extras": extras, "config": config})
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=chat_input(prompt),
        environment_requirements=EnvironmentRequirements() if environment is None else environment,
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind="external", parameters_json=verifier.model_dump_json()),
        source=source,
    )
