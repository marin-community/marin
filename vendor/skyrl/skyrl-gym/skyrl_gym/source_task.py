"""Convert SkyRL source rows to tasks with private verifier inputs."""

from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue
from taskcompendium.chat import chat_input
from taskcompendium.grader import grader_config
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    SessionGrader,
    Source,
    TaskSpec,
)
from taskcompendium.runtime.resources import inline_resource


class SessionParameters(BaseModel):
    """Private inputs for a SkyRL task session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extras: dict[str, JsonValue]
    config: dict[str, JsonValue]


def session_parameters(task: TaskSpec) -> SessionParameters:
    return SessionParameters.model_validate(grader_config(task))


def source_task(
    prompt: list[dict[str, Any]],
    extras: dict[str, Any],
    config: dict[str, Any],
    source: Source,
    *,
    environment: EnvironmentRequirements | None = None,
) -> TaskSpec:
    """Preserve source semantics without selecting a session or machine backend."""
    parameters = SessionParameters(extras=extras, config=config)
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=chat_input(prompt),
        environment_requirements=EnvironmentRequirements() if environment is None else environment,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=SessionGrader(),
        resources=ResourceGroups(verifier=(inline_resource("config.json", parameters.model_dump_json().encode()),)),
        source=source,
    )
