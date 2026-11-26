# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an agent that owns its prompts and tool loop, and that sends each model request through the model session."""

import asyncio
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from taskcompendium.models import TaskSpec

from rolloutengine.cleanup import _retain_task
from rolloutengine.harness.core import HarnessContext, ModelSession, reply_options_match
from rolloutengine.harness.shell import task_prompt
from rolloutengine.models import LENGTH_STOP_REASON, ContextRecovery, ModelTurn, Prompt, RolloutEnded, RolloutLimits


class AgentCompletion(Protocol):
    async def __call__(self, prompt: Prompt) -> ModelTurn: ...


AgentRun = Callable[[HarnessContext, AgentCompletion], Coroutine[Any, Any, None]]
"""An agent: it gets the attempt's context and a completion function, and sends one prompt at a time."""


class LengthStopPolicy(StrEnum):
    """What a prompt that extends a turn cut by its output limit does. The session never replies to that turn."""

    END = "end"
    """The prompt ends the attempt with stop reason ``"length"``."""
    NEW_CONVERSATION = "new_conversation"
    """The prompt starts a new conversation with its full history."""


class ReplyMatch(StrEnum):
    """Which prompts that repeat the conversation so far continue it with a reply."""

    EXACT = "exact"
    """Only a prompt whose messages and options repeat the conversation exactly; the model gets what the agent sent."""
    NORMALIZED = "normalized"
    """Also a prompt whose messages match after dropping empty fields and normalizing tool arguments, with any
    output limits, so that an agent that re-serializes its history keeps one conversation."""


def _messages_equal(left: tuple[dict[str, Any], ...], right: tuple[dict[str, Any], ...]) -> bool:
    """Compare messages after normalizing empty fields and tool argument JSON."""
    return len(left) == len(right) and all(
        a == b or _normalized_message(a) == _normalized_message(b) for a, b in zip(left, right, strict=True)
    )


def _normalized_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in message.items()
        if value is not None
        and (key not in {"reasoning", "reasoning_content", "content", "annotations"} or value not in ("", []))
    }
    if result.get("tool_calls"):
        result["tool_calls"] = [_normalized_tool_call(call) for call in result["tool_calls"]]
    return result


def _normalized_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    try:
        arguments = json.dumps(json.loads(function["arguments"]), sort_keys=True)
    except json.JSONDecodeError:
        return call
    return {**call, "function": {**function, "arguments": arguments}}


async def complete_agent_prompt(
    session: ModelSession,
    prompt: Prompt,
    last: tuple[Prompt, ModelTurn] | None,
    length_stop: LengthStopPolicy = LengthStopPolicy.END,
    reply_match: ReplyMatch = ReplyMatch.NORMALIZED,
) -> ModelTurn:
    """Serve an external agent's prompt as a repeat, reply, or new conversation.

    A prompt that extends the last turn exactly is a reply. Under ``ReplyMatch.NORMALIZED``, so is one whose
    messages match it after normalization, unless that turn was cut by its output limit. ``length_stop``
    decides whether a prompt that extends a cut turn exactly is a reply, which ends the attempt, or a new
    conversation.
    """
    if last is not None:
        previous, turn = last
        if prompt == previous:
            return turn
        history = (*previous.messages, turn.message)
        prefix = prompt.messages[: len(history)]
        cut = turn.stop_reason == LENGTH_STOP_REASON
        exact_reply = (
            prefix == history
            and prompt.options == previous.options
            and not (cut and length_stop is LengthStopPolicy.NEW_CONVERSATION)
        )
        text_reply = (
            reply_match is ReplyMatch.NORMALIZED
            and prompt.token_ids is None
            and not cut
            and reply_options_match(previous.options, prompt.options)
            and _messages_equal(prefix, history)
        )
        if exact_reply or text_reply:
            return await session.reply(
                prompt.messages[len(history) :], token_ids=prompt.token_ids, options=prompt.options
            )
    return await session.start(prompt)


@dataclass(frozen=True)
class AgentHarness:
    """A harness for an agent that owns its prompts and tool loop.

    ``length_stop`` decides what a prompt that extends a turn cut by its output limit does, and ``reply_match``
    which prompts continue the conversation. When the attempt
    ends, calls wait until the cleanup, which cancels the agent before the engine grades. See the task rollouts
    reference for conversation, cancellation, and shutdown rules.
    """

    context_recovery = ContextRecovery.ALWAYS

    agent: AgentRun
    length_stop: LengthStopPolicy = LengthStopPolicy.END
    reply_match: ReplyMatch = ReplyMatch.NORMALIZED

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        # The agent does not get the tools that the task's own prompt advertises. A shell binding is exempt,
        # because the agent uses its own shell.
        advertised = task_prompt(task, None, ()).options.get("tools", ())
        tools = [tool["function"]["name"] for tool in advertised if tool["function"]["name"] not in task.tool_bindings]
        tools.extend(provider.action_interface for provider in task.environment_requirements.tool_providers.values())
        if tools:
            raise NotImplementedError(f"The agent harness does not give these task tools to the agent: {tools}")

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        completions = _Completions(session, self.length_stop, self.reply_match)
        agent = asyncio.create_task(self.agent(context, completions.complete))
        context.cleanup.push("agent_stop", lambda: completions.stop(agent))
        try:
            await asyncio.wait((agent, completions.ended), return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Later calls, also with the last prompt, wait for the cleanup.
            completions.end()
        if agent.done():
            completions.reported = True
            # The engine cancels the agent only in the cleanup, after this. Thus an agent that ended cancelled, for
            # example from a cancelled child task or from its own cancel, failed.
            if agent.cancelled():
                raise RuntimeError("The agent ended with a CancelledError before the engine stopped it")
            agent.result()


class _Completions:
    """Answer the agent's completion calls with the model session, each call in its caller's task."""

    def __init__(self, session: ModelSession, length_stop: LengthStopPolicy, reply_match: ReplyMatch):
        self.session = session
        self.length_stop = length_stop
        self.reply_match = reply_match
        self.last: tuple[Prompt, ModelTurn] | None = None
        self.ended: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.released = asyncio.Event()
        self.reported = False

    def end(self) -> None:
        if not self.ended.done():
            self.ended.set_result(None)

    async def complete(self, prompt: Prompt) -> ModelTurn:
        # Each call yields, so that an agent that calls in a loop cannot block a deadline.
        await asyncio.sleep(0)
        if not self.ended.done():
            try:
                turn = await complete_agent_prompt(self.session, prompt, self.last, self.length_stop, self.reply_match)
            except RolloutEnded:
                self.end()
            else:
                self.last = prompt, turn
                return turn
        await self.released.wait()
        raise RolloutEnded("The rollout ended")

    async def stop(self, agent: asyncio.Task[None]) -> None:
        # Waiting calls fail first, so that an agent that waits for its handlers can stop.
        self.released.set()
        agent.cancel()
        try:
            await asyncio.wait((agent,))
        except asyncio.CancelledError:
            # The cleanup deadline expired. Keep the agent until it ends, and log a failure.
            _retain_task(agent)
            raise
        # A failure that the harness did not report, for example during grading, is a cleanup error.
        if not self.reported and not agent.cancelled():
            error = agent.exception()
            if error is not None and not isinstance(error, RolloutEnded):
                raise error
