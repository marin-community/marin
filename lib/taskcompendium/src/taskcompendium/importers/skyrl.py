# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Gym source rows to tasks with private verifier inputs."""

from typing import Any

from taskcompendium.chat import chat_input
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, ExternalVerifierSpec
from taskcompendium.models import AnswerType, EnvironmentRequirements, Source, TaskSpec, VerifierKind, VerifierSpec

GYM_INTERACTION = "skyrl_gym"


def gym_task(
    prompt: list[dict[str, Any]],
    environment: str,
    extras: dict[str, Any],
    config: dict[str, Any],
    source: Source,
) -> TaskSpec:
    """Convert a source row to a self-contained task with private grading inputs."""
    verifier = ExternalVerifierSpec(name=environment, parameters={"extras": extras, "config": config})
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=chat_input(prompt),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.EXTERNAL, parameters_json=verifier.model_dump_json()),
        environment=EnvironmentSpec(kind=EnvironmentKind.NULL, interaction=GYM_INTERACTION),
        source=source,
        metadata={"teacher_route": extras["teacher_route"]} if "teacher_route" in extras else {},
    )
