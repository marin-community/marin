# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate semantic requirements before presenting a task as chat."""

from taskcompendium.environment import EnvironmentKind
from taskcompendium.models import AnswerType, TaskSpec


def unsupported_direct_chat_features(specification: TaskSpec) -> tuple[str, ...]:
    """List semantic requirements the direct-chat runtime cannot preserve."""
    features = ["capabilities"] if specification.environment_requirements.capabilities else []
    if specification.interaction_tools:
        features.append("interaction_tools")
    if specification.output_paths:
        features.append("output_paths")
    if specification.verifier.environment_requirements.capabilities:
        features.append("verifier.environment_requirements")
    if specification.answer_type in {AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE}:
        features.append(specification.answer_type.value)
    if specification.environment.kind != EnvironmentKind.NULL or specification.environment.interaction is not None:
        features.append("runtime_environment")
    if specification.stages:
        features.append("stages")
    return tuple(features)
