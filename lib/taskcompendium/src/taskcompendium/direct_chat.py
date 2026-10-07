# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate semantic requirements before presenting a task as chat."""

from taskcompendium.models import AnswerType, EnvironmentRequirements, TaskSpec


def unsupported_direct_chat_features(specification: TaskSpec) -> tuple[str, ...]:
    """List semantic requirements the direct-chat runtime cannot preserve."""
    requirements = specification.environment_requirements
    features = [
        name
        for name, value in (
            ("compatible_backends", requirements.compatible_backends),
            ("capabilities", requirements.capabilities),
            ("docker_image", requirements.docker_image),
            ("working_directory", requirements.working_directory),
            ("setup_commands", requirements.setup_commands),
            ("environment_variables", requirements.environment_variables),
            ("tool_providers", requirements.tool_providers),
        )
        if value
    ]
    if specification.interaction_tools:
        features.append("interaction_tools")
    if specification.output_paths:
        features.append("output_paths")
    if specification.output_directories:
        features.append("output_directories")
    if specification.verifier.environment_requirements != EnvironmentRequirements():
        features.append("verifier.environment_requirements")
    resources = specification.resources
    if resources.all or resources.worker or resources.oracle:
        features.append("resources")
    if specification.answer_type in {AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE}:
        features.append(specification.answer_type.value)
    return tuple(features)
