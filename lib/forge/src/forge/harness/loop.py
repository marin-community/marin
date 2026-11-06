# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scoped tool loop with explicit limits and no transport retry policy."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath
from typing import override

from taskforge.llm.policy import LLMPolicy

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
from forge.model.content_hash import digest
from forge.model.llm import FinishReason, Message, Role
from forge.model.stage import Exit
from forge.model.workpiece import Conversation
from forge.resources.env import WriteFile
from forge.resources.inference import ContextExceeded, InferenceClient, InferenceRequest, complete

# Mark's policy: nudge until an explicit exit; repeated empty completions indicate provider failure.
MAX_EMPTY_COMPLETIONS = 3


@dataclass(frozen=True, slots=True)
class AgenticHarness(Harness):
    max_prompt_tokens: int
    sampling: LLMPolicy
    nudge: str
    output_chars: int

    def __post_init__(self) -> None:
        if self.output_chars < 1:
            raise ValueError("Tool output cap must be positive")

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        prompt_tokens, empty_completions, turns = 0, 0, 0
        schemas = tuple(tool.schema for tool in tools.values())
        session = Conversation(session.messages, schemas)
        async with ctx.scope("harness"):
            ctx.resource(client, client.fingerprint)
            ctx.handle(InferenceRequest, partial(complete, notify=ctx.emit), uses=(client,))
            while True:
                await ctx.check(Hold)
                if prompt_tokens >= self.max_prompt_tokens:
                    return Stopped(
                        StopReason.CONTEXT_LIMIT,
                        session,
                        f"estimated prompt {prompt_tokens} tokens reaches model window {self.max_prompt_tokens}",
                        turns,
                    )
                turns += 1
                try:
                    result = await ctx.emit(InferenceRequest(session.messages, schemas, self.sampling))
                except ContextExceeded as error:
                    return Stopped(StopReason.CONTEXT_LIMIT, session, str(error), turns)
                # Provider prompt_tokens already includes the cached portion.
                prompt_tokens = result.usage.prompt_tokens + result.usage.completion_tokens
                session = Conversation(
                    (
                        *session.messages,
                        Message(Role.ASSISTANT, result.content, result.tool_calls, reasoning=result.reasoning),
                    ),
                    schemas,
                )
                empty_completions = empty_completions + 1 if not result.content and not result.tool_calls else 0
                if empty_completions == MAX_EMPTY_COMPLETIONS:
                    raise Hold(f"provider returned {MAX_EMPTY_COMPLETIONS} empty completions")
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
                        outcome = ToolResult(f"Unknown tool: {call.name}") if tool is None else await tool(ctx, args)
                    if len(result.tool_calls) > 1 and isinstance(outcome, (SubmitRequest, ReturnRequest, FinishRequest)):
                        outcome = ToolResult("Session-returning tools must be called alone. Retry this call by itself.")
                    if isinstance(outcome, SubmitRequest):
                        return Submitted(outcome.submission, session)
                    if isinstance(outcome, ReturnRequest):
                        return Returned(outcome.tool, outcome.payload, session)
                    if isinstance(outcome, FinishRequest):
                        return Stopped(StopReason.FINISHED, session, outcome.reason, turns)
                    content = outcome.content
                    if len(content) > self.output_chars:
                        path = PurePosixPath(".tool-output") / f"{digest((call.id, content))}.txt"
                        data = content.encode()
                        await ctx.emit(WriteFile(path, data))
                        marker = (
                            f"\n[Full output saved to {path}: {len(content)} characters, {len(data)} bytes. "
                            "Use read_file with byte offset and limit to read slices.]"
                        )
                        content = content[: self.output_chars] + marker
                    # Estimate only appended text; the provider counted the prior prompt and reply.
                    prompt_tokens += (len(content) + 3) // 4
                    await ctx.emit(ToolReturned(call.id, call.name, content))
                    session = Conversation(
                        (*session.messages, Message(Role.TOOL, content, tool_call_id=call.id)), schemas
                    )
                if not result.tool_calls and result.finish_reason == FinishReason.STOP:
                    prompt_tokens += (len(self.nudge) + 3) // 4
                    session = Conversation((*session.messages, Message(Role.USER, self.nudge)), schemas)
