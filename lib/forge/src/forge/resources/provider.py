# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A wall-clock availability window around the frozen provider transport."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import override

from taskforge.llm.client import Attempt, Completion, ModelUnavailable
from taskforge.llm.glm_client import GlmClient, _Segment, request_kwargs
from taskforge.llm.policy import LLMPolicy

from forge.resources.inference import CompletionBackend


class EndpointLimitedClient(GlmClient):
    """The serving endpoint alone sets output capacity; no max_tokens is sent."""

    # Source: frozen taskforge/llm/glm_client.py::_segment, without its output cap/heuristic.
    @override
    async def _segment(
        self,
        messages: Sequence[dict[str, object]],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        segment: int,
        attempts: list[Attempt],
    ) -> _Segment:
        fields = {
            key: value for key, value in request_fields.items() if key not in ("max_tokens", "max_completion_tokens")
        }
        kwargs = request_kwargs(self.model, messages, 0, policy, fields)
        kwargs.pop("max_tokens")

        async def send(events: list[str]) -> _Segment:
            return await self._stream(kwargs, events)

        # Zero records an unspecified client output allowance in transport telemetry.
        return await self._retrying(send, segment, 0, attempts)


class WaitingBackend(CompletionBackend):
    def __init__(
        self,
        client: CompletionBackend,
        unavailable_wait: float,
        initial: float,
        maximum: float,
        factor: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if unavailable_wait <= 0 or initial <= 0 or maximum <= 0 or factor < 1:
            raise ValueError("Provider availability window and backoff must be positive; factor must be at least 1")
        self.client, self.unavailable_wait = client, unavailable_wait
        self.initial, self.maximum, self.factor = initial, min(maximum, 60.0), factor
        self.clock, self.sleep = clock, sleep

    @property
    @override
    def model(self) -> str:
        return self.client.model

    @override
    async def complete(
        self,
        messages: Sequence[dict[str, object]],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> Completion:
        started = self.clock()
        deadline = started + self.unavailable_wait
        delay = min(self.initial, self.maximum)
        attempts: list[Attempt] = []
        while (remaining := deadline - self.clock()) > 0:
            try:
                # The deadline bounds unavailable retries, not a healthy streaming response.
                result = await self.client.complete(messages, policy, request_fields)
            except ModelUnavailable as error:
                attempts.extend(error.attempts)
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise ModelUnavailable(f"GLM unavailable for {self.unavailable_wait}s", attempts) from error
                wait = min(delay, remaining)
                logging.warning("GLM unavailable; %.1fs left in availability window, retrying in %.1fs", remaining, wait)
                await self.sleep(wait)
                delay = min(delay * self.factor, self.maximum)
            else:
                return replace(result, attempts=(*attempts, *result.attempts), wall_time=self.clock() - started)
        raise ModelUnavailable(f"GLM unavailable for {self.unavailable_wait}s", attempts)
