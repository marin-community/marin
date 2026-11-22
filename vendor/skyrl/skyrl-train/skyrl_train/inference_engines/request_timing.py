"""Bounded per-request engine evidence, independent of completed response retention."""

from collections.abc import Callable, Iterable
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

    @staticmethod
    def request_key(request_id: str) -> str:
        return request_id.removeprefix("chatcmpl-").removeprefix("cmpl-")

    def start(self, request_id: str, session_id: str | None) -> None:
        key = self.request_key(request_id)
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
        for event, timestamp in events:
            if event == "QUEUED" and timing.queued_at is None:
                timing.queued_at = timestamp
            elif event == "SCHEDULED" and timing.scheduled_at is None:
                timing.scheduled_at = timestamp
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
