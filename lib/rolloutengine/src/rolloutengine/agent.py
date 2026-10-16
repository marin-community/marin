# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an agent that owns its prompts and tool loop, while the engine samples each model turn."""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

from taskcompendium.grading_result import GradeResult

from rolloutengine.contracts import ModelTurn, Prompt, RolloutContractError, RolloutData, Transition
from rolloutengine.harness import Episode, Harness, HarnessContext
from rolloutengine.shell import task_prompt
from rolloutengine.spec import RolloutSpec


class AgentCompletion(Protocol):
    async def __call__(self, prompt: Prompt) -> ModelTurn: ...


AgentRun = Callable[[HarnessContext, AgentCompletion], Coroutine[Any, Any, None]]
"""An agent: it gets the attempt's context and a completion function, and sends one prompt at a time."""


TURN_UNDELIVERED = "turn_undelivered"
"""``Transition.metrics`` key of a turn that the agent never received; training must exclude its step."""


class RolloutEnded(Exception):
    """The rollout ended before the model served this request."""


def _check_task_tools(spec: RolloutSpec) -> None:
    """Reject the tools that a task's own prompt advertises, because the bridge does not give them to the agent.

    A shell binding is exempt: the agent uses its own shell.
    """
    task = spec.task
    advertised = task_prompt(task, None, ()).options.get("tools", ())
    tools = [tool["function"]["name"] for tool in advertised if tool["function"]["name"] not in task.tool_bindings]
    tools.extend(requirement.action_interface for requirement in task.environment_requirements.tool_providers.values())
    if tools:
        raise NotImplementedError(f"The agent harness does not give these task tools to the agent: {tools}")


def agent_harness(
    run: AgentRun,
    *,
    check: Callable[[RolloutSpec], None] = _check_task_tools,
    grade: Callable[[RolloutData], Awaitable[GradeResult]] | None = None,
) -> Harness[Episode]:
    """A harness for an agent that owns its prompts and tool loop.

    Each completion call sends one prompt and returns the engine's exact ``ModelTurn``. The calls can
    come from tasks that the agent does not await, such as an HTTP server's handlers. A prompt that
    extends the previous one with the model's message, under the same options, adds observations (with
    its pinned token IDs, if any); any other prompt resets the conversation and starts a new training
    record. An identical prompt is a retry, for example after an HTTP client disconnect, and receives the
    same turn without a new generation; a turn that the agent never receives is marked with
    ``TURN_UNDELIVERED``, and training must exclude its step. When the rollout ends, a request that is
    still open fails with ``RolloutEnded``. ``grade`` grades the record of a ``SessionGrader`` task;
    without it, the engine grades the task. The default ``check`` rejects a task that declares tools other
    than a shell binding, such as an answer tool or a tool provider; an agent that serves them passes its
    own ``check``. The agent must release its own subprocesses and network resources when it is cancelled;
    the engine owns the task machine's lifetime.
    """

    def open_episode(context: HarnessContext) -> Episode:
        return _AgentEpisode(context, run)

    if grade is None:
        return Harness(open_episode, check)

    async def grade_record(_episode: Episode, record: RolloutData) -> GradeResult:
        return await grade(record)

    return Harness(open_episode, check, grade_record)


@dataclass
class _PendingRequest:
    prompt: Prompt
    response: asyncio.Future[ModelTurn]
    received: bool = False
    """Whether the caller resumed with the turn; a caller cancelled after ``set_result`` never does."""


class _AgentEpisode:
    """Bridge the agent's completion calls to the engine's turn loop."""

    def __init__(self, context: HarnessContext, run: AgentRun):
        self.context = context
        self.run = run
        # At most one live request; queued requests of cancelled callers are pruned at the next admission.
        self.requests: asyncio.Queue[_PendingRequest] = asyncio.Queue(maxsize=1)
        self.agent: asyncio.Task[None] | None = None
        self.pending: _PendingRequest | None = None
        self.in_flight: asyncio.Future[ModelTurn] | None = None
        self.closed = False

    async def _complete(self, prompt: Prompt) -> ModelTurn:
        if self.closed:
            raise RolloutEnded("The rollout ended before this request")
        # A cancelled caller's future is done, so its replacement request is not a concurrent one.
        if self.in_flight is not None and not self.in_flight.done():
            raise RolloutContractError("An agent sends only one model request at a time")
        while not self.requests.empty():
            # Only a request whose caller was cancelled before the engine took it can still be queued.
            self.requests.get_nowait()
        request = _PendingRequest(prompt, asyncio.get_running_loop().create_future())
        self.in_flight = request.response
        self.requests.put_nowait(request)
        turn = await request.response
        request.received = True
        return turn

    async def _next_request(self) -> _PendingRequest | None:
        """The agent's next request whose caller still waits, or ``None`` when the agent exits."""
        assert self.agent is not None
        while True:
            receive = asyncio.create_task(self.requests.get())
            try:
                completed, _ = await asyncio.wait((receive, self.agent), return_when=asyncio.FIRST_COMPLETED)
                if self.agent in completed:
                    await self.agent
                    if receive in completed and not receive.result().response.done():
                        raise RolloutContractError("The agent exited with an unanswered model request")
                    return None
                request = receive.result()
            finally:
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
            if not request.response.done():
                return request

    async def prepare(self) -> Prompt:
        if self.agent is not None:
            raise RuntimeError("The agent episode was already started")
        self.agent = asyncio.create_task(self.run(self.context, self._complete))
        self.pending = await self._next_request()
        if self.pending is None:
            raise RolloutContractError("The agent exited before requesting a model response")
        return self.pending.prompt

    async def advance(self, turn: ModelTurn) -> Transition:
        assert self.pending is not None
        prompt = self.pending.prompt
        served: list[_PendingRequest] = []
        # An identical prompt is a retry, for example after an HTTP client disconnect; it receives the same turn.
        while self.pending is not None and self.pending.prompt == prompt:
            if not self.pending.response.done():
                self.pending.response.set_result(turn)
                served.append(self.pending)
            self.pending = await self._next_request()
        # A caller cancelled after the turn arrived never resumes with it, so receipt decides delivery.
        delivered = any(request.received for request in served)
        undelivered = {} if delivered else {TURN_UNDELIVERED: 1.0}
        if self.pending is None:
            return Transition(done=True, metrics=undelivered)
        following = self.pending.prompt
        prefix = (*prompt.messages, turn.message)
        if delivered and following.options == prompt.options and following.messages[: len(prefix)] == prefix:
            return Transition(done=False, observations=following.messages[len(prefix) :], token_ids=following.token_ids)
        # The agent never received the turn, compacted its history, or changed its tools.
        return Transition(done=False, reset=following, metrics=undelivered)

    async def close(self) -> None:
        self.closed = True
        if self.agent is not None:
            # Cancelling the agent also cancels a request that the agent itself waits for.
            self.agent.cancel()
        if self.in_flight is not None and not self.in_flight.done():
            # A caller that the agent does not await, such as an HTTP handler, gets an answer before the
            # engine waits for the agent, so an agent that drains its handlers can stop.
            self.in_flight.set_exception(RolloutEnded("The rollout ended before the model served this request"))
        if self.agent is not None:
            try:
                await self.agent
            except asyncio.CancelledError:
                pass
