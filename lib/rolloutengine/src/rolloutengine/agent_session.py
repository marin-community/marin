# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Let an external agent request generations from the shared rollout loop."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from taskcompendium.grading_result import GradeResult

from rolloutengine.contracts import ModelTurn, RolloutContractError, SessionStart, Transition


class AgentCompletion(Protocol):
    async def __call__(self, request: SessionStart) -> ModelTurn: ...


class AgentGrader(Protocol):
    async def __call__(self, messages: tuple[dict[str, Any], ...], *, stop_reason: str) -> GradeResult: ...


@dataclass(frozen=True)
class _PendingRequest:
    request: SessionStart
    response: asyncio.Future[ModelTurn]


class ExternalAgentSession:
    """Run a sequential agent while the engine owns each model generation.

    The agent adapter translates its wire protocol to ``SessionStart`` and calls
    the supplied completion function. It receives the engine's exact ``ModelTurn``.
    The agent coroutine must release its own subprocesses and network resources
    on success, failure, or cancellation; the engine owns the task machine's lifetime.
    """

    def __init__(
        self,
        run_agent: Callable[[AgentCompletion], Awaitable[None]],
        grader: AgentGrader,
    ):
        self.run_agent = run_agent
        self.grader = grader
        self.requests: asyncio.Queue[_PendingRequest] = asyncio.Queue(maxsize=1)
        self.agent: asyncio.Task[None] | None = None
        self.pending: _PendingRequest | None = None
        self.in_flight: asyncio.Future[ModelTurn] | None = None
        self.closed = False

    async def _complete(self, request: SessionStart) -> ModelTurn:
        if self.closed:
            raise RuntimeError("The agent session is closed")
        if self.in_flight is not None:
            raise RolloutContractError("An agent session supports only one model request at a time")
        response = asyncio.get_running_loop().create_future()
        self.in_flight = response
        try:
            self.requests.put_nowait(_PendingRequest(request, response))
            return await response
        finally:
            self.in_flight = None

    async def _next_request(self) -> _PendingRequest | None:
        assert self.agent is not None
        receive = asyncio.create_task(self.requests.get())
        try:
            completed, _ = await asyncio.wait((receive, self.agent), return_when=asyncio.FIRST_COMPLETED)
            if self.agent in completed:
                await self.agent
                if receive in completed:
                    raise RolloutContractError("The agent exited with an unanswered model request")
                return None
            return receive.result()
        finally:
            receive.cancel()
            await asyncio.gather(receive, return_exceptions=True)

    async def prepare(self) -> SessionStart:
        if self.agent is not None:
            raise RuntimeError("The agent session was already started")
        self.agent = asyncio.create_task(self.run_agent(self._complete))
        self.pending = await self._next_request()
        if self.pending is None:
            raise RolloutContractError("The agent exited before requesting a model response")
        return self.pending.request

    async def advance(self, turn: ModelTurn) -> Transition:
        assert self.pending is not None
        previous = self.pending
        previous.response.set_result(turn)
        self.pending = await self._next_request()
        if self.pending is None:
            return Transition(done=True)
        request = self.pending.request
        prefix = (*previous.request.messages, turn.message)
        # Native agents may compact history or change the available tools.
        # Both require a new exact-token prefix instead of appending observations.
        if request.messages[: len(prefix)] != prefix or request.options != previous.request.options:
            return Transition(done=False, reset_conversation=request.messages, next_options=request.options)
        return Transition(done=False, observations=request.messages[len(prefix) :])

    async def grade(self, messages: tuple[dict[str, Any], ...], *, stop_reason: str) -> GradeResult:
        return await self.grader(messages, stop_reason=stop_reason)

    async def close(self) -> None:
        self.closed = True
        if self.agent is None:
            return
        if not self.agent.done():
            self.agent.cancel()
        try:
            await self.agent
        except asyncio.CancelledError:
            pass
