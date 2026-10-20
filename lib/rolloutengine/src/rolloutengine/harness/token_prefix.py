# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a task's exact token prefix in one model turn without rewriting it."""

from dataclasses import dataclass

from taskcompendium.models import AnswerType, EnvironmentRequirements, PlainText, TaskSpec, TokenizedInput
from taskcompendium.submission import conversation_messages

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import Prompt, RolloutContractError, RolloutLimits


@dataclass(frozen=True, kw_only=True)
class TokenPrefixHarness:
    """Complete a tokenized text task with an explicitly configured inference tokenizer.

    The tokenizer fields must describe the configured model transport. Equality
    with a task's pin checks configuration, not the remote server's tokenizer.
    The session separately verifies that the transport served the exact tokens.
    """

    tokenizer_model: str
    tokenizer_revision: str

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        context = task.context
        if not isinstance(context, TokenizedInput):
            raise ValueError("TokenPrefixHarness requires TokenizedInput")
        if (context.tokenizer_model, context.tokenizer_revision) != (self.tokenizer_model, self.tokenizer_revision):
            raise ValueError("Task tokenizer pin differs from the configured inference tokenizer")
        if task.answer_type != AnswerType.TEXT or not isinstance(task.answer_format, PlainText):
            raise NotImplementedError("TokenPrefixHarness requires a plain-text continuation")
        if (
            task.environment_requirements != EnvironmentRequirements()
            or task.final_tools
            or task.interaction_tools
            or task.tool_bindings
            or task.output_paths
            or task.resources.all
            or task.resources.worker
        ):
            raise NotImplementedError("TokenPrefixHarness cannot serve task environments, tools, or workspace outputs")

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        task_input = context.task.context
        if not isinstance(task_input, TokenizedInput):
            raise ValueError("TokenPrefixHarness requires TokenizedInput")
        turn = await session.start(
            Prompt(
                messages=tuple(conversation_messages(task_input.events)),
                options={"skip_special_tokens": False},
                token_ids=task_input.token_ids,
            )
        )
        if turn.message != {"role": "assistant", "content": turn.text}:
            raise RolloutContractError(
                "Token-prefix transport must preserve raw completion text as the assistant content"
            )
