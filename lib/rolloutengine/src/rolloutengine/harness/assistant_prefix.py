# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One model reply continuing an explicitly selected assistant text prefix."""

from dataclasses import dataclass
from typing import Any

from taskcompendium.models import CONVERSATION_ANSWERS, PlainText, TaskSpec, TextMessage
from taskcompendium.submission import conversation_messages

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import Prompt, RolloutLimits

ASSISTANT_PREFIX_HARNESS = "assistant_prefix"


def assistant_prefix_prompt(task: TaskSpec) -> Prompt:
    """Keep the task's assistant prefix last and advertise schemas without execution bindings."""
    last = task.context.events[-1]
    if not isinstance(last, TextMessage) or last.role != "assistant":
        raise ValueError("An assistant-prefix task must end with an assistant text message")
    options: dict[str, Any] = {"extra_body": {"continue_final_message": True, "add_generation_prompt": False}}
    if task.final_tools:
        options["tools"] = [
            {"type": "function", "function": tool.model_dump(exclude_none=True)} for tool in task.final_tools
        ]
    return Prompt(tuple(conversation_messages(task.context.events)), options)


@dataclass(frozen=True)
class AssistantPrefixHarness:
    """Continue the final assistant message once, retaining raw text for grading."""

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        assistant_prefix_prompt(task)
        if task.answer_type not in CONVERSATION_ANSWERS or not isinstance(task.answer_format, PlainText):
            raise ValueError("An assistant-prefix task must submit a plain-text continuation")
        if (
            task.interaction_tools
            or task.tool_bindings
            or task.environment_requirements.tool_providers
            or task.environment_requirements.capabilities
        ):
            raise ValueError("An assistant-prefix task cannot execute tools or require agent capabilities")

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        turn = await session.start(assistant_prefix_prompt(context.task))
        if (
            turn.message.get("reasoning_content")
            or turn.message.get("reasoning")
            or turn.metadata.get("reasoning")
            or turn.message.get("tool_calls")
        ):
            raise ValueError("The assistant-prefix continuation was parsed into another channel; use raw text output")
        session.record()
