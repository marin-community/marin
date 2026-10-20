# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One model completion of a conversation, optionally continuing an assistant prefill."""

from taskcompendium.models import CONVERSATION_ANSWERS, EnvironmentRequirements, TaskSpec
from taskcompendium.submission import completion_request

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import Prompt, RolloutLimits


class CompletionHarness:
    """Answer a conversation in one model turn, using the task's answer format."""

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        if task.answer_type not in CONVERSATION_ANSWERS:
            raise NotImplementedError("CompletionHarness requires a conversation answer")
        if (
            task.environment_requirements != EnvironmentRequirements()
            or task.interaction_tools
            or task.tool_bindings
            or task.output_paths
            or task.resources.all
            or task.resources.worker
        ):
            raise NotImplementedError(
                "CompletionHarness cannot run task environments, interactive tools, or workspace outputs"
            )
        completion_request(task)

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        request = completion_request(context.task)
        await session.start(Prompt(messages=tuple(request.pop("messages")), options=request))
