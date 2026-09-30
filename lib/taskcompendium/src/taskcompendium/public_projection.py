# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reviewed agent-visible projection of a private TaskSpec."""

from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict

from taskcompendium.lowering import HarborEnvironmentConfig
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    EnvironmentRequirements,
    FunctionDefinition,
    ProviderRequirement,
    Source,
    TaskSpec,
    TextMessage,
    ToolResult,
)
from taskcompendium.submission import SubmissionConvention, submission_instruction

_PUBLIC_SPEC_FIELDS = frozenset(
    {
        "id",
        "context",
        "environment_requirements",
        "tool_providers",
        "final_tools",
        "answer_type",
        "verifier",
        "source",
        "tags",
        "schema_version",
    }
)
_PUBLIC_NESTED_FIELDS = MappingProxyType(
    {
        ConversationInput: frozenset({"events"}),
        TextMessage: frozenset({"type", "role", "content"}),
        AssistantToolCalls: frozenset({"type", "calls", "content"}),
        ConversationToolCall: frozenset({"call_id", "name", "arguments"}),
        ToolResult: frozenset({"type", "call_id", "content"}),
        EnvironmentRequirements: frozenset({"capabilities"}),
        ProviderRequirement: frozenset({"action_interface", "seed_sha256"}),
        FunctionDefinition: frozenset({"name", "parameters", "description", "strict"}),
        Source: frozenset({"dataset", "revision", "row", "importer_revision"}),
    }
)


class PublicTask(BaseModel):
    """Explicit agent-visible view; verifier fields have no representation here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_version: Literal[1] = 1
    id: str
    context: ConversationInput
    environment_requirements: EnvironmentRequirements
    tool_providers: dict[str, ProviderRequirement]
    final_tools: tuple[FunctionDefinition, ...]
    answer_type: AnswerType
    source: Source
    submission_instruction: str
    tags: tuple[str, ...] = ()
    source_category: str | None = None


def public_task(
    specification: TaskSpec,
    convention: SubmissionConvention,
    environment: HarborEnvironmentConfig,
    *,
    source_category: str | None = None,
) -> PublicTask:
    """Select only reviewed public fields from a private task and its binding."""
    if set(type(specification).model_fields) != _PUBLIC_SPEC_FIELDS:
        raise ValueError("TaskSpec fields changed; review the public projection before exporting")
    if any(set(model.model_fields) != fields for model, fields in _PUBLIC_NESTED_FIELDS.items()):
        raise ValueError("Nested task fields changed; review the public projection before exporting")
    if set(type(environment).model_fields) != {"tool_providers"}:
        raise ValueError("Environment fields changed; review the public projection before exporting")
    if not convention.supports(specification.answer_type):
        raise ValueError("Submission convention cannot carry the task answer type")
    if set(environment.tool_providers) != set(specification.tool_providers):
        raise ValueError("Selected tool bindings do not match task requirements")
    for name, requirement in specification.tool_providers.items():
        binding = environment.tool_providers[name]
        if binding.action_interface != requirement.action_interface or binding.seed_sha256 != requirement.seed_sha256:
            raise ValueError("Selected tool binding interface or seed differs from task requirements")
    if any(not tag for tag in specification.tags):
        raise ValueError("Source tags must be nonempty")
    return PublicTask(
        id=specification.id,
        context=specification.context,
        environment_requirements=specification.environment_requirements,
        tool_providers=specification.tool_providers,
        final_tools=specification.final_tools,
        answer_type=specification.answer_type,
        source=specification.source,
        submission_instruction=submission_instruction(convention),
        tags=specification.tags,
        source_category=source_category,
    )
