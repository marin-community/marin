# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One completion of a ``tool_free_single`` task: its system and user text, no tools, no follow-up."""

from dataclasses import dataclass

from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import ContextRecovery, Prompt, RolloutLimits

TOOL_FREE_HARNESS = "tool_free_single"


def tool_free_prompt(task: TaskSpec) -> Prompt:
    """The source's own messages with no tool advertisement."""
    return Prompt(tuple(conversation_messages(task.context.events)), {})


@dataclass(frozen=True)
class ToolFreeHarness:
    """Request exactly one reply; tool calls in it are graded as the host-captured reply status records them."""

    context_recovery = ContextRecovery.NEVER

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        if task.completion_protocol != TOOL_FREE_HARNESS:
            raise ValueError("The tool-free harness serves only tool_free_single tasks")

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        await session.start(tool_free_prompt(context.task))
        session.record()
