# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scoped tool loop with explicit limits and no transport retry policy."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import override

from forge.effects.context import Context
from forge.effects.issues import Hold
from forge.harness.events import ToolInvoked, ToolReturned
from forge.harness.protocol import (
    FinishRequest,
    Harness,
    Returned,
    ReturnRequest,
    Stopped,
    StopReason,
    SubmitRequest,
    Submitted,
    Tool,
    ToolResult,
)
from forge.model.llm import FinishReason, Message, Sampling
from forge.model.stage import Exit
from forge.model.workpiece import Conversation
from forge.resources.env import Env
from forge.resources.inference import InferenceClient, InferenceRequest, complete


@dataclass(frozen=True, slots=True)
class Limits:
    max_turns: int | None
    max_prompt_tokens: int | None
    soft_prompt_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AgenticHarness(Harness):
    limits: Limits
    sampling: Sampling
    nudge: str
    wrap_up: str
    output_chars: int

    def __post_init__(self) -> None:
        if self.output_chars < 1:
            raise ValueError("Tool output cap must be positive")

    @override
    async def run(
        self, ctx: Context, session: Conversation, env: Env, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        turns, prompt_tokens, nudged = 0, 0, False
        warned = Message("user", self.wrap_up) in session.messages
        schemas = tuple(tool.schema for tool in tools.values())
        session = Conversation(session.messages, schemas)
        async with ctx.scope("harness"):
            ctx.resource(client, client.fingerprint)
            ctx.handle(InferenceRequest, partial(complete, notify=ctx.emit), uses=(client,))
            while True:
                await ctx.check(Hold)
                if self.limits.max_turns is not None and turns >= self.limits.max_turns:
                    return Stopped(StopReason.TURN_LIMIT, session)
                if self.limits.max_prompt_tokens is not None and prompt_tokens >= self.limits.max_prompt_tokens:
                    return Stopped(StopReason.CONTEXT_LIMIT, session)
                if (
                    not warned
                    and self.limits.soft_prompt_tokens is not None
                    and prompt_tokens >= self.limits.soft_prompt_tokens
                ):
                    session = Conversation((*session.messages, Message("user", self.wrap_up)), schemas)
                    warned = True
                result = await ctx.emit(InferenceRequest(session.messages, schemas, self.sampling))
                turns += 1
                # Provider prompt_tokens already includes the cached portion.
                prompt_tokens = result.usage.prompt_tokens
                session = Conversation(
                    (
                        *session.messages,
                        Message("assistant", result.content, result.tool_calls, reasoning=result.reasoning),
                    ),
                    schemas,
                )
                for call in result.tool_calls:
                    await ctx.emit(ToolInvoked(call.id, call.name, call.arguments))
                    try:
                        args = json.loads(call.arguments)
                        if not isinstance(args, dict):
                            raise ValueError("Tool arguments must be a JSON object")
                    except ValueError as error:
                        outcome = ToolResult(f"Invalid arguments: {error}")
                    else:
                        tool = tools.get(call.name)
                        outcome = (
                            ToolResult(f"Unknown tool: {call.name}") if tool is None else await tool(ctx, args, env)
                        )
                    if isinstance(outcome, SubmitRequest):
                        return Submitted(outcome.submission, session)
                    if isinstance(outcome, ReturnRequest):
                        return Returned(outcome.tool, outcome.payload, session)
                    if isinstance(outcome, FinishRequest):
                        return Stopped(StopReason.FINISHED, session)
                    content = outcome.content
                    if len(content) > self.output_chars:
                        marker = "\n[output truncated]"
                        content = content[: max(0, self.output_chars - len(marker))] + marker[: self.output_chars]
                    await ctx.emit(ToolReturned(call.id, call.name, content))
                    session = Conversation((*session.messages, Message("tool", content, tool_call_id=call.id)), schemas)
                if not result.tool_calls and result.finish_reason == FinishReason.STOP:
                    if nudged:
                        return Stopped(StopReason.FINISHED, session)
                    nudged = True
                    session = Conversation((*session.messages, Message("user", self.nudge)), schemas)
