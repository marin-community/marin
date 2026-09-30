"""Bounded, credential-free retry policy for provider sandbox provisioning."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any

MAX_ATTEMPTS = 4
MAX_DELAY_SECONDS = 60.0
BASE_DELAY_SECONDS = 5.0


class ProviderRateLimitExhausted(RuntimeError):
    def __init__(self, attempts: list[dict[str, Any]]):
        self.attempts = attempts
        super().__init__(
            "provider sandbox provisioning rate limit exhausted: "
            + json.dumps(attempts, sort_keys=True, separators=(",", ":"))
        )


def _retry_after(error: Exception, fallback: float) -> float:
    headers = getattr(error, "headers", {})
    if not isinstance(headers, Mapping):
        headers = {}
    values = {str(key).lower(): value for key, value in headers.items()}
    value = values.get("retry-after")
    delay = None
    if type(value) in (int, float):
        delay = float(value)
    elif isinstance(value, str):
        try:
            delay = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                delay = retry_at.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                delay = None
    if delay is None or not math.isfinite(delay) or delay < 0:
        delay = fallback
    return min(MAX_DELAY_SECONDS, delay)


def provision_with_rate_limit_retry(
    create: Callable[[], Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[Any, list[dict[str, Any]]]:
    """Retry only a sandbox-create call; never retry candidate execution/grading."""
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be between one and {MAX_ATTEMPTS}")
    attempts = []
    for attempt in range(1, max_attempts + 1):
        try:
            value = create()
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status != 429 and type(error).__name__ != "DaytonaRateLimitError":
                raise
            record = {
                "attempt": attempt,
                "state": "rate_limited",
                "status_code": status,
                "error_code": getattr(error, "error_code", None),
            }
            if attempt == max_attempts:
                attempts.append(record)
                raise ProviderRateLimitExhausted(attempts) from error
            delay = _retry_after(
                error,
                min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1))),
            )
            record["retry_after_seconds"] = delay
            attempts.append(record)
            sleep(delay)
        else:
            attempts.append({"attempt": attempt, "state": "created"})
            return value, attempts
    raise AssertionError("bounded provider retry loop did not terminate")
