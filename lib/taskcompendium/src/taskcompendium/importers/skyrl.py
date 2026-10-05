# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert SkyRL source rows to tasks with private verifier inputs."""

from typing import Any

from taskcompendium.chat import chat_input
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, ExternalVerifierSpec
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec, VerifierKind, VerifierSpec


def source_task(
    prompt: list[dict[str, Any]],
    session: str,
    extras: dict[str, Any],
    config: dict[str, Any],
    source: Source,
    *,
    environment: EnvironmentSpec | None = None,
) -> TaskSpec:
    """Convert a source row to a self-contained task with private grading inputs."""
    verifier = ExternalVerifierSpec(parameters={"extras": extras, "config": config})
    resolved_environment = EnvironmentSpec(kind=EnvironmentKind.NULL) if environment is None else environment
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=chat_input(prompt),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.EXTERNAL, parameters_json=verifier.model_dump_json()),
        environment=resolved_environment.model_copy(update={"interaction": session}),
        source=source,
        metadata={"teacher_route": extras["teacher_route"]} if "teacher_route" in extras else {},
    )
