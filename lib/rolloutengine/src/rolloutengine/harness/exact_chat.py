# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One chat generation using the benchmark's exact conversation prefix."""

from dataclasses import dataclass, field
from typing import Any

from taskcompendium.direct_chat import unsupported_direct_chat_features
from taskcompendium.models import CONVERSATION_ANSWERS, TaskSpec
from taskcompendium.submission import conversation_messages

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import Prompt, RolloutLimits

EXACT_CHAT_HARNESS = "exact_chat"


@dataclass(frozen=True)
class ExactChatHarness:
    """Keep the source prompt unchanged; sampling is supplied by the evaluation protocol."""

    options: dict[str, Any] = field(default_factory=dict)

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        unsupported = tuple(
            feature for feature in unsupported_direct_chat_features(task) if feature != "grader.environment"
        )
        if unsupported:
            raise ValueError(f"Exact chat cannot satisfy: {', '.join(unsupported)}")
        if task.answer_type not in CONVERSATION_ANSWERS:
            raise ValueError("Exact chat requires a conversation answer")
        if task.final_tools or task.interaction_tools or task.tool_bindings:
            raise ValueError("Exact chat evaluation does not execute tools")

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        await session.start(Prompt(tuple(conversation_messages(context.task.context.events)), dict(self.options)))
        session.record()
