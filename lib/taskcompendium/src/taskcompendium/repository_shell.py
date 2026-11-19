# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repository and workspace declared by a ``shell:v1`` provider.

The source's public setup instructions initialize the checkout. The provider
executes commands in that workspace without cloning or installing dependencies
on behalf of the source.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from taskcompendium.models import (
    EnvironmentRequirements,
    FunctionDefinition,
    ProviderRequirement,
    normalized_absolute_path,
)

SHELL_ACTION_INTERFACE = "shell:v1"


def shell_tool_definition(name: str, command_parameter: str) -> FunctionDefinition:
    return FunctionDefinition(
        name=name,
        description="Run a Bash command in the task workspace. Files persist between commands.",
        parameters={
            "type": "object",
            "properties": {command_parameter: {"type": "string"}},
            "required": [command_parameter],
            "additionalProperties": False,
        },
    )


SHELL_TOOL = shell_tool_definition("shell", "command")


class RepositoryShellState(BaseModel):
    """Public checkout identity and working directory, without an answer or patch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    workspace: str

    @field_validator("workspace")
    @classmethod
    def normalized_workspace(cls, value: str) -> str:
        return normalized_absolute_path(value)


def repository_shell_state(requirement: ProviderRequirement) -> RepositoryShellState:
    """Read the versioned provider without accepting a different tool interface."""
    if requirement.action_interface != SHELL_ACTION_INTERFACE:
        raise ValueError(f"Unsupported repository shell interface: {requirement.action_interface}")
    return RepositoryShellState.model_validate(requirement.initial_state)


def repository_shell_environment(requirements: EnvironmentRequirements) -> RepositoryShellState | None:
    """Bind the actor working directory without overriding a source declaration."""
    states = [
        repository_shell_state(requirement)
        for requirement in requirements.tool_providers.values()
        if requirement.action_interface == SHELL_ACTION_INTERFACE
    ]
    if not states:
        return None
    if len(states) != 1:
        raise ValueError("An actor machine can serve only one shell:v1 workspace")
    state = states[0]
    if requirements.working_directory not in (None, state.workspace):
        raise ValueError("shell:v1 workspace conflicts with the declared actor working directory")
    return state
