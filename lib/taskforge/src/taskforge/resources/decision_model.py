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
from typing import Protocol, override

import httpx
from rolloutengine.models import GradedRollout, Model, TaskRolloutSpec
from taskcompendium.decision_model import DecisionModelRequest

from taskforge.effects.context import Context
from taskforge.effects.events import Event, Fingerprint, ReplayPolicy, Retrying, digest
from taskforge.effects.issues import DecisionModelUnavailable
from taskforge.llm.glm_provider import EndpointSource, EndpointUnavailable, GlmEndpoint
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

    async def _endpoint(self) -> GlmEndpoint:
        try:
            return await asyncio.to_thread(self.source.current)
        except EndpointUnavailable as error:
            raise DecisionModelDown(str(error)) from error

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        endpoint = await self._endpoint()
        state = {"task_deliverable": request.deliverable, "rubric": request.rubric}
        state |= {"y_reference_answer": request.reference, "z_candidate_answer": request.candidate}
        question = {"type": "score", "instructions": request.question, "criteria": list(request.levels)}
        body = {"model": CLEF_MODEL, "state": state, "questions": {CLEF_QUESTION: question}}
        headers = {"Authorization": f"Bearer {endpoint.token}"}
        try:  # the resource's request deadline bounds the call
            response = await self.http.post(f"{endpoint.base_url}/systemone", json=body, headers=headers, timeout=None)
        except httpx.TransportError as error:
            raise DecisionModelDown(f"unreachable: {type(error).__name__}") from error
        if (status := response.status_code) in DOWN_STATUSES or status >= 500:
            seconds = response.headers.get("retry-after", "")  # an HTTP-date form is ignored
            asked = float(seconds) if seconds.replace(".", "", 1).isdigit() else 0.0
            raise DecisionModelDown(f"HTTP {status}: {response.text[:200]}", asked)
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
            endpoint = await self._endpoint()
            async with asyncio.timeout(HEALTH_TIMEOUT):
                response = await self.http.get(
                    f"{endpoint.base_url}/systemone/health", headers={"Authorization": f"Bearer {endpoint.token}"}
                )
        except (DecisionModelDown, httpx.TransportError, TimeoutError):
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

    def __post_init__(self) -> None:
        self.gate = asyncio.Semaphore(self.policy.concurrency)

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("decision_model", digest(self.policy.model), ReplayPolicy.CACHE)

    async def read(self, event: DecisionModelRead, *, notify: Notify) -> float:
        policy = self.policy
        delay, attempt, why = policy.backoff_initial, 1, f"none of its {policy.concurrency} read slots came free"
        try:
            async with asyncio.timeout(policy.unavailable_wait), self.gate:
                while True:
                    why = "its read went unanswered"
                    try:
                        async with asyncio.timeout(policy.request_timeout):
                            level = await self.reader.read(event.request)
                        self.down = None
                        return level
                    except DecisionModelDown as error:
                        reason, asked = str(error), error.retry_after
                    except TimeoutError:
                        reason, asked = f"no answer within {policy.request_timeout:g}s", 0.0
                    self.down = reason
                    notify(Retrying(event, attempt, reason))
                    await self.sleep(max(asked, delay))
                    delay, attempt = min(delay * 2, policy.backoff_maximum), attempt + 1
        except TimeoutError:
            last = why if self.down is None else f"last {self.down}"
            raise DecisionModelUnavailable(
                f"{policy.model} gave no read within {policy.unavailable_wait:g}s: {last}"
            ) from None


async def read_level(event: DecisionModelRead, model: DecisionModel, *, notify: Notify) -> float:
    return await model.read(event, notify=notify)


@dataclass(frozen=True, slots=True)
class DecisionModelEngine:
    """Bind each decision-model spec's read, at grade setup, to recorded ``DecisionModelRead`` effects."""

    engine: Engine
    ctx: Context

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        if (connection := spec.decision_model) is not None:

            async def read(request: DecisionModelRequest) -> float:
                return await self.ctx.emit(DecisionModelRead(request))

            spec = spec.model_copy(update={"decision_model": replace(connection, read=read)})
        return await self.engine.run_task(spec, model, run_id=run_id)
