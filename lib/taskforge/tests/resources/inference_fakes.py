# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scripted completion backend and request shared by the inference and provider tests."""

from collections.abc import Mapping, Sequence
from typing import override

from taskforge.llm.client import Attempt, ModelRequestRejected, ModelUnavailable
from taskforge.llm.client import Completion as ProviderCompletion
from taskforge.llm.client import FinishReason as ProviderFinish
from taskforge.llm.client import ToolCall as ProviderToolCall
from taskforge.llm.client import Usage as ProviderUsage
from taskforge.llm.policy import LLMPolicy
from taskforge.model.llm import Message, Role, ToolCall, ToolSchema
from taskforge.resources.inference import CompletionBackend, InferenceRequest


class Backend(CompletionBackend):
    model = "test-model"

    def __init__(self, script: Sequence[ProviderCompletion | ModelRequestRejected | ModelUnavailable]) -> None:
        self.script = tuple(script)
        self.calls: list[tuple[Sequence[dict[str, object]], LLMPolicy, Mapping[str, object] | None]] = []

    @override
    async def complete(
        self,
        messages: Sequence[dict[str, object]],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> ProviderCompletion:
        value = self.script[len(self.calls)]
        self.calls.append((messages, policy, request_fields))
        if isinstance(value, ModelRequestRejected | ModelUnavailable):
            raise value
        return value


def provider_completion(attempts: tuple[Attempt, ...] = ()) -> ProviderCompletion:
    return ProviderCompletion(
        "answer",
        "reason",
        (ProviderToolCall("id", "shell", "{}"),),
        ProviderFinish.TOOL_CALLS,
        ProviderUsage(10, 20, 3, 4),
        1,
        0.1,
        0.9,
        2,
        attempts,
    )


def request() -> InferenceRequest:
    return InferenceRequest(
        (
            Message(Role.ASSISTANT, "prefix", (ToolCall("previous", "shell", "{}"),), reasoning="thought"),
            Message(Role.TOOL, "result", tool_call_id="previous"),
        ),
        (ToolSchema("shell", "Run command", {"type": "object"}),),
        LLMPolicy(max_tokens=100),
    )
