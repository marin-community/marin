# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scoped tool loop with no transport retry policy.

By design there is no turn, token or wall-clock bound: the model's context window bounds a session,
which otherwise ends through an exit tool or repeated empty or degenerate completions.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import PurePosixPath
from typing import override

from taskforge.effects.context import Context
from taskforge.effects.issues import ContextExceeded, Hold, OutputBudgetExhausted
from taskforge.harness.events import ToolInvoked, ToolReturned
from taskforge.harness.protocol import (
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
from taskforge.llm.policy import LLMPolicy
from taskforge.model.content_hash import digest
from taskforge.model.llm import Completion, FinishReason, Message, Role
from taskforge.model.stage import Exit
from taskforge.model.workpiece import Conversation
from taskforge.resources.env import WriteFile
from taskforge.resources.execution import Notify
from taskforge.resources.inference import InferenceClient, InferenceRequest, complete

# A session ends only through an exit tool, so an empty completion gets a nudge. This many empty
# completions in a row mean the provider is failing.
MAX_EMPTY_COMPLETIONS = 3


async def settled(event: InferenceRequest, client: InferenceClient, *, notify: Notify) -> Completion:
    """The harness's inference: a provider's context refusal settles the visit, so it is a recorded result (with the
    retries before it) that a restarted visit replays, never a raise that leaves the request to be asked again."""
    try:
        return await complete(event, client, notify=notify)
    except ContextExceeded as error:
        return Completion(str(error), finish_reason=FinishReason.CONTEXT_EXCEEDED, retries=error.retries)


@dataclass(frozen=True, slots=True)
class AgenticHarness(Harness):
    max_prompt_tokens: int
    sampling: LLMPolicy
    nudge: str
    output_chars: int
    replay_arguments: Callable[[str], str]
    """The model profile's repair of sampled tool-call arguments, applied before they enter the conversation."""
    spill: bool = True
    """Whether a reply longer than ``output_chars`` is saved to a workspace file; a visit without a machine has none,
    and its replies are clipped instead. The role runner sets it per visit."""

    def __post_init__(self) -> None:
        if self.output_chars < 1:
            raise ValueError("Tool output cap must be positive")

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        prompt_tokens, empty_completions, degenerate_completions = 0, 0, 0
        turns = sum(message.role == Role.ASSISTANT for message in session.messages)
        previous: InferenceRequest | None = None
        consumed = 0
        schemas = tuple(tool.schema for tool in tools.values())
        session = Conversation(session.messages, schemas)
        async with ctx.scope("harness"):
            ctx.handle(InferenceRequest, partial(settled, notify=ctx.notify), client)
            while True:
                await ctx.check()
                if prompt_tokens >= self.max_prompt_tokens:
                    return Stopped(
                        StopReason.CONTEXT_LIMIT,
                        session,
                        f"estimated prompt {prompt_tokens} tokens reaches model window {self.max_prompt_tokens}",
                        turns,
                    )
                turns += 1
                request = InferenceRequest(
                    session.messages[consumed:], schemas, self.sampling, previous, sample=degenerate_completions
                )
                result = await ctx.emit(request)
                if result.finish_reason is FinishReason.CONTEXT_EXCEEDED:
                    return Stopped(StopReason.CONTEXT_LIMIT, session, result.content, turns)
                previous, consumed = request, len(session.messages)
                if result.finish_reason == FinishReason.LENGTH and not result.content and not result.tool_calls:
                    degenerate_completions += 1
                    if degenerate_completions == 2:
                        return Stopped(
                            StopReason.DEGENERATE_OUTPUT,
                            session,
                            f"{result.usage.reasoning_tokens} reasoning tokens, finish=length",
                            turns,
                        )
                    continue
                degenerate_completions = 0
                # Provider prompt_tokens already includes the cached portion.
                prompt_tokens = result.usage.prompt_tokens + result.usage.completion_tokens
                calls = tuple(
                    replace(call, arguments=self.replay_arguments(call.arguments)) for call in result.tool_calls
                )
                message = Message(Role.ASSISTANT, result.content, calls, reasoning=result.reasoning)
                session = replace(session, messages=(*session.messages, message))
                empty_completions = empty_completions + 1 if not result.content and not result.tool_calls else 0
                if empty_completions == MAX_EMPTY_COMPLETIONS:
                    raise Hold(f"provider returned {MAX_EMPTY_COMPLETIONS} empty completions")
                for call in result.tool_calls:
                    ctx.notify(ToolInvoked(call.id, call.name, call.arguments))
                    try:
                        args = json.loads(call.arguments)
                        if not isinstance(args, dict):
                            raise ValueError("Tool arguments must be a JSON object")
                    except ValueError as error:
                        outcome = ToolResult(
                            f"{call.name} arguments were not a valid JSON object: {error}; call {call.name} again."
                        )
                    else:
                        tool = tools.get(call.name)
                        try:
                            outcome = ToolResult(f"Unknown tool: {call.name}") if tool is None else await tool(ctx, args)
                        except OutputBudgetExhausted as error:
                            error.session = session  # for a caller that records where the session ended
                            raise
                    if len(result.tool_calls) > 1 and isinstance(outcome, (SubmitRequest, ReturnRequest, FinishRequest)):
                        outcome = ToolResult("Session-returning tools must be called alone. Retry this call by itself.")
                    if isinstance(outcome, SubmitRequest):
                        return Submitted(outcome.submission, session)
                    if isinstance(outcome, ReturnRequest):
                        return Returned(outcome.tool, outcome.payload, session)
                    if isinstance(outcome, FinishRequest):
                        return Stopped(StopReason.FINISHED, session, outcome.reason, turns)
                    content = outcome.content
                    if len(content) > self.output_chars and not self.spill:  # every reply passes here, diagnostics too
                        cut = (
                            f"\n[reply cut at {self.output_chars} characters: "
                            "this session has no machine to hold the rest]"
                        )
                        content = content[: max(self.output_chars - len(cut), 0)] + cut[: self.output_chars]
                    elif len(content) > self.output_chars:
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
                    ctx.notify(ToolReturned(call.id, call.name, content))
                    session = replace(
                        session, messages=(*session.messages, Message(Role.TOOL, content, tool_call_id=call.id))
                    )
                if not result.tool_calls and result.finish_reason == FinishReason.STOP:
                    prompt_tokens += (len(self.nudge) + 3) // 4
                    session = replace(session, messages=(*session.messages, Message(Role.USER, self.nudge)))
