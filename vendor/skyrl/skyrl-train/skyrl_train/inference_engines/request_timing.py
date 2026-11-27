"""Bounded per-request engine evidence, independent of completed response retention."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass


@dataclass
class RequestTiming:
    session_id: str | None
    queued_at: float | None = None
    scheduled_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    last_report_at: float | None = None
    generated_tokens: int = 0
    prompt_tokens: int | None = None


class RequestTimings:
    """Emit queue, first-token and periodic decode evidence without text or credentials."""

    def __init__(self, emit: Callable[[dict], None]):
        self.emit = emit
        self.requests: dict[str, RequestTiming] = {}
        self._scheduled: dict[str, asyncio.Event] = {}

    @staticmethod
    def request_key(request_id: str) -> str:
        return request_id.removeprefix("chatcmpl-").removeprefix("cmpl-")

    def scheduled_event(self, request_id: str) -> asyncio.Event:
        """Register before submission so even a fast request cannot lose admission."""
        return self._scheduled.setdefault(self.request_key(request_id), asyncio.Event())

    def forget_schedule(self, request_id: str) -> None:
        self._scheduled.pop(self.request_key(request_id), None)

    def start(self, request_id: str, session_id: str | None) -> None:
        key = self.request_key(request_id)
        self.scheduled_event(key)
        self.requests[key] = RequestTiming(session_id)
        self.emit({"event": "submitted", "request_id": key, "session_id": session_id})

    def observe(
        self,
        request_id: str,
        events: Iterable[tuple[str, float]],
        token_count: int,
        sampled_at: float,
        *,
        prompt_tokens: int,
    ) -> None:
        key = self.request_key(request_id)
        timing = self.requests.get(key)
        if timing is None:
            return
        timing.prompt_tokens = prompt_tokens
        previously_scheduled = timing.scheduled_at is not None
        for event, timestamp in events:
            if event == "QUEUED" and timing.queued_at is None:
                timing.queued_at = timestamp
            elif event == "SCHEDULED" and timing.scheduled_at is None:
                timing.scheduled_at = timestamp
        if timing.scheduled_at is not None or token_count:
            self.scheduled_event(key).set()
        if timing.scheduled_at is not None and not previously_scheduled:
            self.report(key, "scheduled", timing)
        if not token_count:
            return
        first = timing.first_token_at is None
        if first:
            timing.first_token_at = sampled_at
        timing.generated_tokens += token_count
        timing.last_token_at = sampled_at
        if first or timing.last_report_at is None or sampled_at - timing.last_report_at >= 60:
            self.report(key, "first_token" if first else "decode_progress", timing)
            timing.last_report_at = sampled_at

    def finish(self, request_id: str) -> None:
        key = self.request_key(request_id)
        timing = self.requests.pop(key, None)
        self._scheduled.pop(key, None)
        if timing is not None:
            self.report(key, "ended", timing)

    def report(self, request_id: str, event: str, timing: RequestTiming) -> None:
        self.emit(
            {
                "event": event,
                "request_id": request_id,
                **vars(timing),
                "context_tokens": (
                    None if timing.prompt_tokens is None else timing.prompt_tokens + timing.generated_tokens
                ),
                "queue_wait": (
                    timing.scheduled_at - timing.queued_at
                    if timing.scheduled_at is not None and timing.queued_at is not None
                    else None
                ),
            }
        )


@dataclass(frozen=True)
class RequestScheduled:
    """The engine has observed SCHEDULED or a first token, never just submission."""


@dataclass(frozen=True)
class RequestCompleted[T]:
    value: T


async def scheduled_results[T](
    pending: Awaitable[T], scheduled: asyncio.Event
) -> AsyncIterator[RequestScheduled | RequestCompleted[T]]:
    """Stream a result and any observed admission notice on one cancellable RPC."""
    task = asyncio.ensure_future(pending)
    admission = asyncio.create_task(scheduled.wait())
    try:
        await asyncio.wait({task, admission}, return_when=asyncio.FIRST_COMPLETED)
        if scheduled.is_set():
            yield RequestScheduled()
        yield RequestCompleted(await task)
    finally:
        admission.cancel()
        task.cancel()
        await asyncio.gather(admission, task, return_exceptions=True)
