# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The decision model behind ``decision_model`` graders, as recorded ``DecisionModelRead`` effects.

A decision model makes a lightly semantic comparison of an answer with a reference answer against a rubric; it
is not the LLM judge, which grades high-judgement or taste deliverables through VerifyIt's judge mode
(``taskcompendium.models.DecisionModelGrader``). This module shares nothing with the judge path.

The host's decision model is Clef, behind the GLM gateway at ``POST /v1/systemone`` (model ``clef``). One read
asks ``DECISION_QUESTION`` on ``DECISION_LEVELS`` about the state the 2026-10-16 form study measured
(deliverable, rubric, reference, candidate) and returns the expected level. The cut is not part of the read, so
an author who moves the cut replays its logged reads.

Clef is one preemptible pod. While it is unavailable (transport errors, 404, 408, 429, any 5xx such as the
gateway's 503 ``clef_unavailable``) a read waits out ``retry-after`` with backoff; after
``decision_model.unavailable_wait`` it raises ``DecisionModelUnavailable`` so the workpiece is held and its slot
freed. Other refusals are bugs and raise ``RuntimeError``.
"""

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Protocol, override

import httpx
from rolloutengine.models import GradedRollout, Model, RolloutInterrupted, TaskRolloutSpec
from taskcompendium.grading import DecisionModelConnection
from taskcompendium.models import DECISION_LEVELS, DECISION_QUESTION, DecisionModelGrader
from taskforge.llm.glm_client import EndpointSource

from forge.effects.context import Context
from forge.effects.events import Event, Fingerprint, ReplayPolicy, Retrying, digest
from forge.effects.issues import DecisionModelUnavailable, EffectIssue
from forge.resources.execution import Notify
from forge.resources.rollouts import Engine

CLEF_MODEL = "clef"
QUESTION_ID = "whole"
UNAVAILABLE_MARKER = "DECISION_MODEL_UNAVAILABLE"
"""Starts every ``DecisionModelUnavailable`` reason; watchers count held workpieces by it."""
UNAVAILABLE_STATUSES = frozenset({404, 408, 429})
RETRY_DELAY_MIN, RETRY_DELAY_MAX = 1.0, 60.0


@dataclass(frozen=True, slots=True)
class DecisionModelLevel:
    level: float
    input_tokens: int
    retries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionModelRead(Event[DecisionModelLevel]):
    deliverable: str
    rubric: str
    reference: str
    answer: str


class DecisionModelClient(Protocol):
    def fingerprint(self) -> Fingerprint: ...
    async def read(self, event: DecisionModelRead) -> DecisionModelLevel: ...


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("retry-after", 0)))
    except ValueError:
        return 0.0


def expected_level(probabilities: Mapping[str, Any]) -> float:
    weights = [float(probabilities[str(level)]) for level in range(len(DECISION_LEVELS))]
    if any(not math.isfinite(w) or w < 0 for w in weights) or not sum(weights) > 0:
        raise ValueError(f"Clef probabilities are not a distribution: {weights}")
    return sum(index * weight for index, weight in enumerate(weights)) / sum(weights)


class ClefDecisionModel(DecisionModelClient):
    """Clef behind the gateway ``source`` resolves; at most ``max_inflight`` reads, waiting included."""

    def __init__(
        self,
        source: EndpointSource,
        http: httpx.AsyncClient,
        *,
        request_timeout: float,
        unavailable_wait: float,
        max_inflight: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if request_timeout <= 0 or unavailable_wait <= 0 or max_inflight < 1:
            raise ValueError("Decision-model timeouts must be positive and max_inflight at least 1")
        self.source, self.http, self.request_timeout, self.unavailable_wait = source, http, request_timeout, unavailable_wait
        self.slots = asyncio.Semaphore(max_inflight)
        self.clock, self.sleep = clock, sleep

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("decision_model", digest((CLEF_MODEL, DECISION_QUESTION, DECISION_LEVELS)), ReplayPolicy.CACHE)

    @override
    async def read(self, event: DecisionModelRead) -> DecisionModelLevel:
        body = {
            "model": CLEF_MODEL,
            "state": {
                "task_deliverable": event.deliverable,
                "rubric": event.rubric,
                "y_reference_answer": event.reference,
                "z_candidate_answer": event.answer,
            },
            "questions": {
                QUESTION_ID: {"type": "score", "instructions": DECISION_QUESTION, "criteria": list(DECISION_LEVELS)}
            },
        }
        retries: list[str] = []
        delay = RETRY_DELAY_MIN
        async with self.slots:
            deadline = self.clock() + self.unavailable_wait
            while True:
                answer = await self._post(body)
                if isinstance(answer, DecisionModelLevel):
                    return replace(answer, retries=tuple(retries))
                reason, retry_after = answer
                retries.append(reason)
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise DecisionModelUnavailable(
                        f"{UNAVAILABLE_MARKER}: no decision-model answer for {self.unavailable_wait:g}s; last {reason}",
                        retries=tuple(retries),
                    )
                wait = min(max(retry_after, delay), remaining)
                logging.warning("%s: %s; retrying in %.0fs", UNAVAILABLE_MARKER, reason, wait)
                await self.sleep(wait)
                delay = min(delay * 2, RETRY_DELAY_MAX)

    async def _post(self, body: dict[str, object]) -> DecisionModelLevel | tuple[str, float]:
        """One read, or why the decision model is unavailable and how long it asked us to wait."""
        try:
            endpoint = await asyncio.to_thread(self.source.current)
            response = await self.http.post(
                f"{endpoint.base_url}/systemone",
                json=body,
                headers={"Authorization": f"Bearer {endpoint.token}"},
                timeout=self.request_timeout,
            )
        except (httpx.TransportError, RuntimeError) as error:  # RuntimeError: the relay has no endpoint yet
            return f"unreachable: {type(error).__name__}", 0.0
        status = response.status_code
        if status in UNAVAILABLE_STATUSES or status >= 500:
            return f"HTTP {status}: {response.text[:200]}", _retry_after(response)
        if status != 200:
            raise RuntimeError(f"Decision model refused the read with HTTP {status}: {response.text[:300]}")
        try:
            payload = response.json()
            return DecisionModelLevel(
                expected_level(payload["answers"][QUESTION_ID]["probabilities"]), int(payload["usage"]["input_tokens"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Decision model answered 200 without a score read: {error}") from error


async def read(event: DecisionModelRead, client: DecisionModelClient, *, notify: Notify) -> DecisionModelLevel:
    result = await client.read(event)
    for index, cause in enumerate(result.retries, 1):
        notify(Retrying(event, index, cause))
    return result


@dataclass(frozen=True)
class DecisionModelEngine:
    """Bind decision_model graders to recorded ``DecisionModelRead`` effects; other graders pass through."""

    engine: Engine
    ctx: Context
    client: DecisionModelClient

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        if not isinstance(spec.task.grader, DecisionModelGrader):
            return await self.engine.run_task(spec, model, run_id=run_id)
        failure: EffectIssue | None = None

        async def level(grader: DecisionModelGrader, answer: str) -> float:
            nonlocal failure
            event = DecisionModelRead(grader.deliverable, grader.rubric, grader.reference, answer)
            async with self.ctx.scope("decision_model"):
                self.ctx.resource(self.client, self.client.fingerprint)
                self.ctx.handle(DecisionModelRead, partial(read, notify=self.ctx.emit), uses=(self.client,))
                try:
                    return (await self.ctx.emit(event)).level
                except EffectIssue as error:
                    failure = error
                    raise

        spec = spec.model_copy(update={"decision_model": DecisionModelConnection(level)})
        try:
            return await self.engine.run_task(spec, model, run_id=run_id)
        except RolloutInterrupted as error:
            # The engine wraps grading failures; the effect decision (hold, restart) belongs to the decision model.
            if failure is not None:
                raise failure from error
            raise
