# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry provider reads and idempotent named lifecycle operations."""

import asyncio
import logging
from asyncio import get_running_loop, sleep
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum

from daytona import DaytonaError
from daytona.common.errors import DaytonaConnectionError, DaytonaTimeoutError
from rigging.timing import ExponentialBackoff, active_timeout, suspend_active_time

logger = logging.getLogger(__name__)
PROVIDER_RETRY_MAX_DELAY = 120
PROVIDER_REQUEST_TIMEOUT = 60
PENDING_SANDBOX_IP_MESSAGE = "failed to resolve container ip after 3 attempts"


@dataclass
class _RetryScope:
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


_RETRY_SCOPE: ContextVar[_RetryScope | None] = ContextVar("provider_retry_scope", default=None)


def cancel_provider_waits() -> None:
    """Disable retries in the current provider wait scope."""
    scope = _RETRY_SCOPE.get()
    if scope is not None:
        scope.cancel()


@asynccontextmanager
async def provider_wait_scope() -> AsyncIterator[Callable[[], None]]:
    """Bind provider recovery and expose cancellation for an owned child operation."""
    scope = _RETRY_SCOPE.get()
    if scope is not None:
        yield scope.cancel
        return
    scope = _RetryScope()
    token = _RETRY_SCOPE.set(scope)
    try:
        yield scope.cancel
    finally:
        _RETRY_SCOPE.reset(token)


class ProviderWaitPolicy(StrEnum):
    PHASE = "phase"
    JOB = "job"

    def timeout(self, seconds: float | None) -> AbstractAsyncContextManager[asyncio.Timeout]:
        return active_timeout(seconds) if self is ProviderWaitPolicy.JOB else asyncio.timeout(seconds)

    def wait(self) -> AbstractContextManager[None]:
        return suspend_active_time() if self is ProviderWaitPolicy.JOB else nullcontext()


async def retry_provider_call[T](
    operation: str,
    request: Callable[[], Awaitable[T]],
    *,
    deadline: float | None = None,
    wait_policy: ProviderWaitPolicy = ProviderWaitPolicy.PHASE,
    request_timeout: float | None = None,
) -> T:
    """Retry transient failures until success or the caller cancels the operation.

    Use only for idempotent provider operations. Source command execution must
    not be replayed after an ambiguous response. An absolute loop-time deadline
    preserves the last SDK error when another retry would exceed the budget.
    """
    scope = _RETRY_SCOPE.get()
    if scope is not None and scope.cancelled:
        return await request()
    try:
        with wait_policy.wait():
            return await _retry_provider_call(operation, request, deadline, request_timeout)
    except asyncio.CancelledError:
        cancel_provider_waits()
        raise


async def _retry_provider_call[T](
    operation: str, request: Callable[[], Awaitable[T]], deadline: float | None, request_timeout: float | None
) -> T:
    backoff = ExponentialBackoff(initial=1, maximum=PROVIDER_RETRY_MAX_DELAY, factor=2, jitter=0.5)
    attempt = 0
    while True:
        try:
            try:
                async with asyncio.timeout(request_timeout):
                    return await request()
            except TimeoutError as error:
                raise DaytonaTimeoutError(f"Provider request timed out: {operation}") from error
        except DaytonaError as error:
            scope = _RETRY_SCOPE.get()
            if scope is not None and scope.cancelled:
                raise
            status = error.status_code
            transient = (
                isinstance(error, (DaytonaConnectionError, DaytonaTimeoutError))
                or status in (408, 429)
                or (status is not None and 500 <= status < 600)
                or (status == 400 and PENDING_SANDBOX_IP_MESSAGE in str(error).lower())
            )
            if not transient:
                raise
            attempt += 1
            retry_after = max(
                (
                    int(value)
                    for name, value in error.headers.items()
                    if name.lower().startswith("retry-after") and str(value).isdecimal()
                ),
                default=0,
            )
            delay = min(PROVIDER_RETRY_MAX_DELAY, max(retry_after, backoff.next_interval()))
            if deadline is not None and get_running_loop().time() + delay >= deadline:
                raise
            logger.warning(
                "DAYTONA_PROVIDER_RETRY operation=%s attempt=%d status=%s delay=%.3f",
                operation,
                attempt,
                status,
                delay,
                extra={
                    "provider_operation": operation,
                    "provider_retry_attempt": attempt,
                    "provider_status_code": status,
                },
            )
            await sleep(delay)
