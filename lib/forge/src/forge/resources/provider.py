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
from taskforge.llm.policy import LLMPolicy

from forge.resources.inference import CompletionBackend


class WaitingBackend(CompletionBackend):
    def __init__(
        self,
        client: CompletionBackend,
        unavailable_wait: float,
        initial: float,
        maximum: float,
        factor: float,
        *,
        role: str,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if unavailable_wait <= 0 or initial <= 0 or maximum <= 0 or factor < 1:
            raise ValueError("Provider availability window and backoff must be positive; factor must be at least 1")
        self.client, self.unavailable_wait, self.role = client, unavailable_wait, role
        self.initial, self.maximum, self.factor = initial, maximum, factor
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
                # The window bounds time unavailable: neither a healthy stream nor one that failed after streaming.
                result = await self.client.complete(messages, policy, request_fields)
            except ModelUnavailable as error:
                attempts.extend(error.attempts)
                if any(attempt.events for attempt in error.attempts):
                    # The provider streamed before failing (e.g. a replica reclaimed mid-stream), so it was
                    # available: restart the window rather than charging healthy streaming time to it.
                    deadline = self.clock() + self.unavailable_wait
                    delay = min(self.initial, self.maximum)
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise ModelUnavailable(
                        f"{self._name()} unavailable for {self.unavailable_wait}s", attempts
                    ) from error
                wait = min(delay, remaining)
                logging.warning(
                    "%s unavailable; %.1fs left in availability window, retrying in %.1fs", self._name(), remaining, wait
                )
                await self.sleep(wait)
                delay = min(delay * self.factor, self.maximum)
            else:
                return replace(result, attempts=(*attempts, *result.attempts), wall_time=self.clock() - started)
        raise ModelUnavailable(f"{self._name()} unavailable for {self.unavailable_wait}s", attempts)

    def _name(self) -> str:
        return f"MODEL role={self.role} model={self.model}"
