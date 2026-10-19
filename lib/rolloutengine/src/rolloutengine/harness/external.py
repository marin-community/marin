# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an agent that owns its prompts and tool loop, and that sends each model request through the recorder."""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

from taskcompendium.models import TaskSpec

from rolloutengine.cleanup import _retain_task
from rolloutengine.contracts import ModelTurn, Prompt, RolloutEnded
from rolloutengine.harness.core import HarnessContext, RolloutRecorder
from rolloutengine.harness.shell import task_prompt
from rolloutengine.spec import RolloutLimits


class AgentCompletion(Protocol):
    async def __call__(self, prompt: Prompt) -> ModelTurn: ...


AgentRun = Callable[[HarnessContext, AgentCompletion], Coroutine[Any, Any, None]]
"""An agent: it gets the attempt's context and a completion function, and sends one prompt at a time."""


@dataclass(frozen=True)
class AgentHarness:
    """A harness for an agent that owns its prompts and tool loop.

    Each completion call sends one prompt and returns the exact ``ModelTurn``, also from tasks that the agent
    does not await, such as HTTP handlers. A prompt that extends the last prompt with the model's message,
    under the same options, is a reply; any other prompt starts a new conversation. The last prompt again gets
    the same turn. One model request can be open at a time. When the attempt ends, calls wait until the
    cleanup, after grading; the cleanup then cancels the agent. ``check`` rejects tools that the agent does
    not get, such as an answer tool or a tool provider. The task rollout reference gives the rules for
    cancelled calls and for the shutdown of the agent.
    """

    agent: AgentRun

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        # The agent does not get the tools that the task's own prompt advertises. A shell binding is exempt,
        # because the agent uses its own shell.
        advertised = task_prompt(task, None, ()).options.get("tools", ())
        tools = [tool["function"]["name"] for tool in advertised if tool["function"]["name"] not in task.tool_bindings]
        tools.extend(provider.action_interface for provider in task.environment_requirements.tool_providers.values())
        if tools:
            raise NotImplementedError(f"The agent harness does not give these task tools to the agent: {tools}")

    async def run(self, context: HarnessContext, recorder: RolloutRecorder) -> None:
        completions = _Completions(recorder)
        agent = asyncio.create_task(self.agent(context, completions.complete))
        context.cleanup.push("agent_stop", lambda: completions.stop(agent))
        try:
            await asyncio.wait((agent, completions.ended), return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Later calls, also with the last prompt, wait for the cleanup.
            completions.end()
        if agent.done():
            completions.reported = True
            agent.result()


@dataclass(frozen=True)
class _Answered:
    prompt: Prompt
    turn: ModelTurn


class _Completions:
    """Answer the agent's completion calls with the recorder, each call in its caller's task."""

    def __init__(self, recorder: RolloutRecorder):
        self.recorder = recorder
        self.last: _Answered | None = None
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
            last = self.last
            if last is not None and prompt == last.prompt:
                return last.turn
            history = () if last is None else (*last.prompt.messages, last.turn.message)
            try:
                if (
                    last is not None
                    and prompt.options == last.prompt.options
                    and prompt.messages[: len(history)] == history
                ):
                    turn = await self.recorder.reply(prompt.messages[len(history) :], token_ids=prompt.token_ids)
                else:
                    turn = await self.recorder.start(prompt)
            except RolloutEnded:
                self.end()
            else:
                self.last = _Answered(prompt, turn)
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
