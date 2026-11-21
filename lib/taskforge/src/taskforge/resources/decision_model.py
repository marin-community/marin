# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision model behind ``decision_model`` graders, read through recorded ``DecisionModelRead`` effects.

A decision model compares a final answer with a reference under a rubric and returns a level; it is not the LLM judge
(``resources/judge.py``), and the two share no client, policy or code. A read is keyed by TaskCompendium's whole
``DecisionModelRequest`` (versioned model identity, question, levels, rubric, reference, deliverable, candidate), so an
equivalent read replays without a call and another model identity misses.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, cast, override

import httpx
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from taskcompendium.decision_model import DecisionModelRequest

from taskforge.effects.context import Context
from taskforge.effects.events import Event, Fingerprint, ReplayPolicy, Retrying, digest, event_digest
from taskforge.effects.issues import DecisionModelUnavailable
from taskforge.llm.glm_provider import EndpointSource, EndpointUnavailable, retry_after
from taskforge.resources.execution import Notify
from taskforge.resources.rollouts import Engine

CLEF_MODEL, CLEF_QUESTION = "clef", "whole"
DOWN_STATUSES = frozenset({404, 408, 429})
"""Besides any 5xx, the gateway statuses that mean Clef cannot answer now, such as its 503 ``clef_unavailable``."""
HEALTH_TIMEOUT = 5.0


class DecisionModelHealth(StrEnum):
    HEALTHY = "healthy"
    DOWN = "down"
    UNSUPPORTED = "unsupported"
    """The host has no health route (404 or 405), such as an older gateway."""


class DecisionModelDown(Exception):
    """The host cannot answer this read now; ``retry_after`` is the wait it asked for."""

    def __init__(self, reason: str, retry_after: float = 0.0) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


class OffScale(Exception):
    """A reader returned the argument, no level of its request: it is never recorded, so a resumed run reads again,
    and TaskCompendium grades it ``INFRA_ERROR`` with its own bounded diagnostic."""


class DecisionModelReader(Protocol):
    """One host's decision model: ``read`` returns a level or raises ``DecisionModelDown``; ``health`` is the host's
    own readiness answer and never a read."""

    async def read(self, request: DecisionModelRequest) -> float: ...

    async def health(self) -> DecisionModelHealth: ...


@dataclass(frozen=True, slots=True)
class ClefReader(DecisionModelReader):
    """The library's default reader: Clef behind the GLM gateway that ``source`` resolves (``POST /systemone``)."""

    source: EndpointSource
    http: httpx.AsyncClient

    async def _send(self, method: str, route: str, body: object = None) -> httpx.Response:
        """The authorized request to ``route``; an unresolvable or unreachable host is down. The caller bounds it."""
        try:
            endpoint = await asyncio.to_thread(self.source.current)
            url, headers = f"{endpoint.base_url}{route}", {"Authorization": f"Bearer {endpoint.token}"}
            return await self.http.request(method, url, headers=headers, json=body, timeout=None)
        except EndpointUnavailable as error:
            raise DecisionModelDown(str(error)) from error
        except httpx.TransportError as error:
            raise DecisionModelDown(f"unreachable: {type(error).__name__}") from error

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        state = {"task_deliverable": request.deliverable, "rubric": request.rubric}
        state |= {"y_reference_answer": request.reference, "z_candidate_answer": request.candidate}
        question = {"type": "score", "instructions": request.question, "criteria": list(request.levels)}
        body = {"model": CLEF_MODEL, "state": state, "questions": {CLEF_QUESTION: question}}
        response = await self._send("POST", "/systemone", body)  # the resource's request deadline bounds it
        if (status := response.status_code) in DOWN_STATUSES or status >= 500:
            raise DecisionModelDown(f"HTTP {status}: {response.text[:200]}", retry_after(response))
        if status != 200:
            raise RuntimeError(f"Decision model refused the read with HTTP {status}: {response.text[:300]}")
        probabilities = response.json()["answers"][CLEF_QUESTION]["probabilities"]
        weights = [float(probabilities[str(level)]) for level in range(len(request.levels))]
        if min(weights) < 0 or not sum(weights) > 0:
            raise RuntimeError(f"Clef probabilities are not a distribution: {weights}")
        return sum(level * weight for level, weight in enumerate(weights)) / sum(weights)

    @override
    async def health(self) -> DecisionModelHealth:
        try:
            async with asyncio.timeout(HEALTH_TIMEOUT):  # relay resolution included
                response = await self._send("GET", "/systemone/health")
        except (DecisionModelDown, TimeoutError):
            return DecisionModelHealth.DOWN
        if response.status_code == 200:
            return DecisionModelHealth.HEALTHY
        return DecisionModelHealth.UNSUPPORTED if response.status_code in (404, 405) else DecisionModelHealth.DOWN


@dataclass(frozen=True, slots=True)
class DecisionModelPolicy:
    """``decision_model.*``: ``model`` is the versioned identity every read records; the rest bound each read."""

    model: str
    concurrency: int
    request_timeout: float
    unavailable_wait: float
    backoff_initial: float
    backoff_maximum: float
    retry_interval: float
    """How often a driver reads the host's health while decision-model work waits."""


@dataclass(frozen=True, slots=True)
class DecisionModelRead(Event[float]):
    kind = "decision_model.read"
    request: DecisionModelRequest


@dataclass(slots=True)
class DecisionModel:
    """At most ``policy.concurrency`` reads at once, each retried with backoff while the host is down; a read that has
    no level ``unavailable_wait`` after it was asked for, its wait for a read slot included, raises
    ``DecisionModelUnavailable``."""

    reader: DecisionModelReader
    policy: DecisionModelPolicy
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    gate: asyncio.Semaphore = field(init=False)
    down: str | None = field(default=None, init=False)
    """Why the latest attempt failed, until one succeeds: what a read still waiting for a slot reports."""
    answered: bool | None = field(default=None, init=False)
    """The latest physical outcome of a read in this process, in order of occurrence: a valid answer (True), or a
    failure or a dispatched read still unanswered at its deadline (False); a read that only waited for a slot is none."""
    reading: dict[str, "Flight"] = field(default_factory=dict, init=False)
    """Reads in flight by event digest, so that equal requests share one paid read."""

    def __post_init__(self) -> None:
        self.gate = asyncio.Semaphore(self.policy.concurrency)

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("decision_model", digest(self.policy.model), ReplayPolicy.CACHE)

    async def level(self, event: DecisionModelRead, notify: Notify) -> float:
        policy = self.policy
        delay, attempt, why = policy.backoff_initial, 1, f"none of its {policy.concurrency} read slots came free"
        try:
            async with asyncio.timeout(policy.unavailable_wait) as deadline, self.gate:
                while True:
                    why, request_deadline = "its read went unanswered", asyncio.timeout(policy.request_timeout)
                    try:  # the one physical call; its outcome is recorded here, whatever ends it
                        async with request_deadline:
                            level: object = await self.reader.read(event.request)
                        top = len(event.request.levels) - 1  # TaskCompendium's level contract, checked before recording
                        if isinstance(level, bool) or not isinstance(level, int | float) or not 0 <= level <= top:
                            raise OffScale(level)
                    except BaseException as error:
                        # Only the caller leaving is no evidence: a cancellation with neither deadline expired. Once
                        # either has, a cancellation arriving during the read's cleanup is still that read failing.
                        if (
                            not isinstance(error, asyncio.CancelledError)
                            or request_deadline.expired()
                            or deadline.expired()
                        ):
                            self.answered = False
                        if not isinstance(error, DecisionModelDown | TimeoutError):
                            raise  # classified as before: a bug, an off-scale level, the caller's cancellation
                        if isinstance(error, DecisionModelDown):
                            reason, asked = str(error), error.retry_after
                        else:
                            reason, asked = f"no answer within {policy.request_timeout:g}s", 0.0
                    else:
                        self.down, self.answered = None, True
                        return level
                    self.down = reason
                    notify(Retrying(event, attempt, reason))
                    await self.sleep(max(asked, delay))
                    delay, attempt = min(delay * 2, policy.backoff_maximum), attempt + 1
        except TimeoutError:
            last = why if self.down is None else f"last {self.down}"
            raise DecisionModelUnavailable(
                f"{policy.model} gave no read within {policy.unavailable_wait:g}s: {last}"
            ) from None


@dataclass(slots=True)
class Flight:
    """A paid read, findable until a waiter takes its result (recorded before any other task runs) or a failure, or
    until its last waiter leaves; a waiter cancelled meanwhile only leaves, and the last to leave early cancels and
    joins it."""

    task: asyncio.Task[float]
    waiters: int = 0


async def read_level(event: DecisionModelRead, model: DecisionModel, *, ctx: Context) -> float:
    if (flight := model.reading.get(key := event_digest(event))) is None:
        # In the caller's context (its Iris client) outside every caller's scope: its retries are root rows.
        read = asyncio.get_running_loop().create_task(model.level(event, ctx.notify), context=ctx.detached())
        flight = model.reading[key] = Flight(read)
    flight.waiters += 1
    try:
        return await asyncio.shield(flight.task)
    finally:
        flight.waiters -= 1
        taken = flight.task.done() and not cast(asyncio.Task[float], asyncio.current_task()).cancelling()
        if (taken or not flight.waiters) and model.reading.get(key) is flight:
            del model.reading[key]
        if not flight.task.done() and not flight.waiters:
            flight.task.cancel()
            await asyncio.gather(flight.task, return_exceptions=True)


@dataclass(frozen=True, slots=True)
class DecisionModelEngine:
    """Bind each decision-model spec's read, at grade setup, to recorded ``DecisionModelRead`` effects."""

    engine: Engine
    ctx: Context

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        if (connection := spec.decision_model) is not None:

            async def read(request: DecisionModelRequest) -> float:
                try:
                    return await self.ctx.emit(DecisionModelRead(request))
                except OffScale as error:
                    return cast(float, error.args[0])  # graded INFRA_ERROR; the level was never recorded

            spec = spec.model_copy(update={"decision_model": replace(connection, read=read)})
        return await self.engine.run_task(spec, model, run_id=run_id)
